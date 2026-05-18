"""LaMI training entry point.

Trains the LateFusionMixin's ``mm_proj`` and ``fusion_layer`` on top of a
frozen base LLM and frozen CLIP vision encoder. Images are either loaded from
disk or generated on the fly with SDXL-Turbo for text-only batches (WikiText).
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
from typing import Optional

import datasets
import torch
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoPipelineForText2Image
from nltk.tokenize import sent_tokenize
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms.functional import crop as crop_image
from tqdm.auto import tqdm
from transformers import SchedulerType, get_scheduler
from transformers.utils.versions import require_version

from load_datasets import load_cropped_vg_regions, load_laion_220, load_wiki
from models import build_fusion_model, build_tokenizer

logger = get_logger(__name__)
require_version("datasets>=2.14", "Install dependencies via requirements.txt.")

to_tensor = transforms.ToTensor()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LaMI late-fusion model.")

    parser.add_argument("--run_name", type=str, default="test")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Comma- or substring-matched list of {wiki, laion_220, cropped_vg}.")
    parser.add_argument("--output_dir", type=str, required=True)

    # data
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_elements", type=int, default=None,
                        help="Per-dataset cap (after concatenation each loader contributes up to this many).")
    parser.add_argument("--shuffle_data", action="store_true")
    parser.add_argument("--preprocessing_num_workers", type=int, default=4)
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--dataset_cache_dir", type=str, default="datasets")
    parser.add_argument("--laion_image_root", type=str, default=None,
                        help="Root dir for LAION-220 image files (required when --dataset_name includes laion_220).")
    parser.add_argument("--vg_cache_dir", type=str, default="datasets/visual_genome_regions",
                        help="Where to materialize Visual Genome images that lack a filename.")
    parser.add_argument("--wiki_num", type=str, default="103")
    parser.add_argument("--wiki_block_size", type=int, default=447)

    # optimization
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear",
                        choices=list(SchedulerType))
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # misc
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--low_cpu_mem_usage", action="store_true")
    parser.add_argument("--pretrained_model", type=str, default=None)
    parser.add_argument("--run_bf16", action="store_true")
    parser.add_argument("--checkpointing_steps", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--with_tracking", action="store_true")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--trust_remote_code", action="store_true")

    # diffusion
    parser.add_argument("--sd_model", type=str, default="stabilityai/sdxl-turbo")
    parser.add_argument("--sd_steps", type=int, default=1)
    parser.add_argument("--sd_guidance", type=float, default=0.0)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def _truncate_to_active_tokens(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                               tokenizer, model_family: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop the trailing padding tokens. Behavior matches the original per-model collates."""
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    if model_family == "gemma":
        # Gemma sequences are right-padded with 0; the first non-zero is the BOS shift.
        non_zero_start = (input_ids != 0).long().argmax(dim=1)
        min_index = max(int(non_zero_start.min().item()) - 1, 0)
        input_ids = input_ids[:, min_index:]
        attention_mask = attention_mask[:, min_index:]
        if input_ids.shape[1] >= 2:
            input_ids[input_ids[:, 1] != 0, 1] = 2
            input_ids[:, 0] = 0
            attention_mask[:, 0] = 0
        return input_ids, attention_mask

    if model_family == "llama":
        input_ids = input_ids[:, :100]
        attention_mask = attention_mask[:, :100]
        sentinel = eos_id if eos_id is not None else pad_id
        first_pad = (input_ids == sentinel).long().argmax(dim=1)
        max_index = max(int(first_pad.max().item()), 1)
        return input_ids[:, :max_index], attention_mask[:, :max_index]

    if model_family in {"gpt2", "opt"}:
        sentinel = eos_id if model_family == "gpt2" else 1  # OPT sentinel is BOS=1
        first = (input_ids == sentinel).long().argmax(dim=1)
        max_index = max(int(first.max().item()), 1)
        return input_ids[:, :max_index], attention_mask[:, :max_index]

    return input_ids, attention_mask


def _mask_for_bert(input_ids: torch.Tensor, mlm_probability: float = 0.20):
    """Standard BERT MLM masking: 80% [MASK], 10% random, 10% unchanged."""
    labels = input_ids.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special = ((input_ids == 101) | (input_ids == 102) | (input_ids == 0))
    probability_matrix.masked_fill_(special, value=0.0)
    masked = torch.bernoulli(probability_matrix).bool()
    labels[~masked] = -100
    replace = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked
    input_ids[replace] = 103
    rand = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked & ~replace
    input_ids[rand] = torch.randint(30522, labels.shape, dtype=torch.long)[rand]
    return input_ids, labels


def make_collate_fn(tokenizer, model_family: str):
    """One collate function that adapts to the model family using tokenizer ids."""

    def collate(examples):
        image_paths = [e["image_path"] for e in examples]
        input_ids = torch.tensor([e["input_ids"] for e in examples], dtype=torch.long)
        attention_mask = torch.tensor([e["attention_mask"] for e in examples], dtype=torch.long)

        if model_family == "bert":
            input_ids = input_ids[:, :100]
            attention_mask = attention_mask[:, :100]
            input_ids, labels = _mask_for_bert(input_ids)

            # Ensure every example has at least one masked position to train against.
            real = (input_ids != 101) & (input_ids != 102) & (input_ids != 0)
            has_mask = torch.any(input_ids == 103, dim=1)
            if (~has_mask).any():
                rows = torch.nonzero(~has_mask, as_tuple=False).squeeze(1)
                cols = torch.randint(2, min(10, input_ids.shape[1]), (rows.shape[0],))
                # Only replace where it's a real token
                for r, c in zip(rows.tolist(), cols.tolist()):
                    if real[r, c]:
                        labels[r, c] = input_ids[r, c]
                        input_ids[r, c] = 103
        else:
            input_ids, attention_mask = _truncate_to_active_tokens(
                input_ids, attention_mask, tokenizer, model_family
            )
            labels = input_ids

        batch = {
            "image_path": image_paths,
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
        if "crop" in examples[0]:
            batch["crop"] = [e["crop"] for e in examples]
        return batch

    return collate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_family(model_name: str) -> str:
    name = model_name.lower()
    for key in ("llama", "gemma", "gpt2", "opt", "bert"):
        if key in name:
            return key
    raise ValueError(f"Unrecognized model family for '{model_name}'.")


def _load_pretrained_weights(model: torch.nn.Module, path: str) -> None:
    state_dict = torch.load(f"{path}.bin", map_location="cpu")
    consume_prefix_in_state_dict_if_present(state_dict, "module.")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Loaded {path}.bin (missing={len(missing)}, unexpected={len(unexpected)})")


def _build_datasets(args, tokenizer):
    parts = []
    if "wiki" in args.dataset_name:
        parts.append(load_wiki(args, tokenizer))
    if "laion_220" in args.dataset_name:
        parts.append(load_laion_220(args, tokenizer, short="short" in args.dataset_name))
    if "cropped_vg" in args.dataset_name:
        parts.append(load_cropped_vg_regions(args, tokenizer))
    if not parts:
        raise ValueError(f"--dataset_name '{args.dataset_name}' matched nothing.")

    if args.max_elements is not None:
        capped = []
        for ds in parts:
            ds_view = ds.shuffle(seed=42) if args.shuffle_data else ds
            capped.append(ds_view.select(range(min(len(ds_view), args.max_elements))))
        parts = capped
    return datasets.concatenate_datasets(parts)


def _prepare_images(batch, pipe, model_dtype, device, processor, tokenizer, args):
    """Read disk images and synthesize the missing ones via SDXL-Turbo."""
    images: list[Optional[torch.Tensor]] = []
    missing_indices: list[int] = []
    for i, path in enumerate(batch["image_path"]):
        if path is None:
            missing_indices.append(i)
            images.append(None)
            continue
        image = read_image(path, mode=ImageReadMode.RGB)
        if "crop" in batch and batch["crop"][i] is not None:
            x, y, w, h = (max(0, v) for v in batch["crop"][i])
            image = crop_image(image, top=y, left=x, height=h, width=w)
        images.append(image)

    if missing_indices:
        prompts = tokenizer.batch_decode(batch["input_ids"][missing_indices], skip_special_tokens=True)
        prompts = [
            random.choice(sent_tokenize(p)) if sent_tokenize(p) else "random"
            for p in prompts
        ]
        generated = pipe(prompts, num_inference_steps=args.sd_steps, guidance_scale=args.sd_guidance).images
        for idx, img in zip(missing_indices, generated):
            tensor = (to_tensor(img) * 255).to(torch.uint8)
            images[idx] = tensor

    pixel_values = processor(images=images, return_tensors="pt").pixel_values
    return pixel_values.to(device=device, dtype=model_dtype)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    family = _detect_family(args.model_name_or_path)

    accelerator_kwargs = {}
    if args.with_tracking:
        accelerator_kwargs["log_with"] = args.report_to
        accelerator_kwargs["project_dir"] = args.output_dir
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_kwargs
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    tokenizer = build_tokenizer(
        args.model_name_or_path, use_fast=True, trust_remote_code=args.trust_remote_code
    )
    model_dtype = torch.bfloat16 if args.run_bf16 else torch.float
    model = build_fusion_model(
        args.model_name_or_path,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        trust_remote_code=args.trust_remote_code,
    )
    processor = model.processor

    if args.pretrained_model is not None:
        _load_pretrained_weights(model, args.pretrained_model)

    train_dataset = _build_datasets(args, tokenizer)
    collate = make_collate_fn(tokenizer, family)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        collate_fn=collate,
        num_workers=args.num_workers,
        shuffle=args.shuffle_data,
    )

    # SDXL-Turbo for text-only batches (e.g. WikiText)
    needs_diffusion = "wiki" in args.dataset_name
    pipe = None
    if needs_diffusion:
        pipe = AutoPipelineForText2Image.from_pretrained(
            args.sd_model, torch_dtype=torch.float16
        ).to(accelerator.device)

    # Freeze everything except mm_proj + fusion_layer.
    model.requires_grad_(False)
    trainable = []
    model.mm_proj.requires_grad_(True)
    trainable += list(model.mm_proj.parameters())
    model.fusion_layer.requires_grad_(True)
    trainable += list(model.fusion_layer.parameters())

    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    overrode_max_train_steps = args.max_train_steps is None
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    if args.with_tracking:
        experiment_config = vars(args).copy()
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("lami", experiment_config)

    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    resume_step = None

    if args.resume_from_checkpoint:
        accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)
        marker = os.path.splitext(os.path.basename(args.resume_from_checkpoint))[0]
        if "epoch" in marker:
            starting_epoch = int(marker.replace("epoch_", "")) + 1
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            resume_step = int(marker.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)
    progress_bar.update(completed_steps)

    unwrapped = accelerator.unwrap_model(model)
    device, dtype = accelerator.device, unwrapped.dtype

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        active = train_dataloader
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            active = accelerator.skip_first_batches(train_dataloader, resume_step)

        for batch in active:
            with accelerator.accumulate(model):
                pixel_values = _prepare_images(batch, pipe, dtype, device, processor, tokenizer, args)
                model_inputs = {k: v for k, v in batch.items() if k not in ("image_path", "crop")}
                model_inputs["pixel_values"] = pixel_values

                outputs = model(**model_inputs)
                loss = outputs.loss

                accelerator.backward(loss)
                if accelerator.sync_gradients and args.grad_clip > 0:
                    accelerator.clip_grad_norm_(trainable, args.grad_clip)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if args.with_tracking:
                    accelerator.log(
                        {"train_loss": loss.item(), "epoch": epoch, "step": completed_steps},
                        step=completed_steps,
                    )

            if accelerator.sync_gradients:
                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.detach().item())
                completed_steps += 1

            if isinstance(checkpointing_steps, int) and completed_steps % checkpointing_steps == 0:
                save_dir = os.path.join(args.output_dir, f"step_{completed_steps}")
                accelerator.save_state(save_dir)

            if completed_steps >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            ckpt_path = os.path.join(args.output_dir, f"{args.run_name}_{epoch}.bin")
            torch.save(accelerator.unwrap_model(model).state_dict(), ckpt_path)


if __name__ == "__main__":
    main()
