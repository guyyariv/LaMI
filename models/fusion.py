"""Late multi-image fusion mixins for HuggingFace causal-LM and masked-LM models.

Each LaMI variant is a small subclass that:
  1. Inherits from the stock HF ``XForCausalLM`` (or ``XForMaskedLM``) class.
  2. Inherits from the appropriate mixin in this module.
  3. Sets ``fusion_model_cls`` and ``fusion_config_cls`` class attributes.
  4. Calls ``self._init_fusion_modules(config)`` at the end of ``__init__``.

The mixin overrides ``forward`` to run the base model on the text, then — if
``pixel_values`` is supplied — encode the images with CLIP, prepend the
projected visual tokens to the base hidden states, and pass them through a
1-layer fusion transformer. Logits come from the inherited LM head and are
optionally re-weighted per image against the text-only path via ``scores``.
"""

from __future__ import annotations

from typing import Optional, Tuple, Type, Union

import torch
from torch import nn
from transformers import AutoProcessor, CLIPVisionModel
from transformers.modeling_outputs import CausalLMOutputWithPast, MaskedLMOutput

CLIP_NAME = "openai/clip-vit-base-patch32"


class MMProj(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.linear1 = nn.Linear(input_size, output_size)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(output_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.gelu(self.linear1(x)))


def _encode_images(mm_model: nn.Module, mm_proj: MMProj, pixel_values: torch.Tensor) -> torch.Tensor:
    """CLIP penultimate-layer features (minus CLS), projected to LM hidden size."""
    image_outputs = mm_model(pixel_values, output_hidden_states=True)
    feat = image_outputs.hidden_states[-2][:, 1:, :]
    return mm_proj(feat)


def _repeat_along_batch(x: Optional[torch.Tensor], times: int) -> Optional[torch.Tensor]:
    if x is None or times == 1:
        return x
    return x.repeat(times, *([1] * (x.dim() - 1)))


class _FusionInit:
    fusion_model_cls: Optional[Type[nn.Module]] = None
    fusion_config_cls: Optional[Type] = None

    def _init_fusion_modules(self, config, clip_name: str = CLIP_NAME) -> None:
        if self.fusion_model_cls is None or self.fusion_config_cls is None:
            raise RuntimeError(
                f"{type(self).__name__} must set fusion_model_cls and fusion_config_cls"
            )
        self.processor = AutoProcessor.from_pretrained(clip_name)
        self.mm_model = CLIPVisionModel.from_pretrained(clip_name)
        self.mm_proj = MMProj(self.mm_model.config.hidden_size, config.hidden_size)
        fusion_cfg = self.fusion_config_cls.from_dict(config.to_dict())
        fusion_cfg.num_hidden_layers = 1
        self.fusion_layer = self.fusion_model_cls(fusion_cfg)
        self._fusion_kv = None


class CausalLMFusionMixin(_FusionInit):
    """Late multi-image fusion for any HF ``XForCausalLM``.

    Subclasses must set ``base_attr`` to the name of the base-decoder attribute
    on the inherited class (``"model"`` for Llama/Gemma, ``"transformer"`` for
    GPT-2). The fusion layer is a 1-layer copy of the same architecture.
    """

    base_attr: str = "model"

    @property
    def _base(self) -> nn.Module:
        return getattr(self, self.base_attr)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        scores: Optional[torch.FloatTensor] = None,
        **base_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        base_outputs = self._base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=False,
            return_dict=True,
            **base_kwargs,
        )
        hidden_states = base_outputs.last_hidden_state
        base_logits = self.lm_head(hidden_states)
        logits = base_logits

        prompt_step = input_ids is not None and input_ids.shape[1] > 1
        decode_step = (
            input_ids is not None
            and input_ids.shape[1] == 1
            and self._fusion_kv is not None
        )

        if pixel_values is not None and prompt_step:
            self._fusion_kv = None  # reset for a fresh prompt
            k = pixel_values.shape[0]
            batch = hidden_states.shape[0]
            if k != batch:
                if k % batch != 0:
                    raise ValueError(
                        f"pixel_values batch ({k}) must be a multiple of text batch ({batch})"
                    )
                times = k // batch
                hidden_states = _repeat_along_batch(hidden_states, times)
                attention_mask = _repeat_along_batch(attention_mask, times)
                labels = _repeat_along_batch(labels, times)

            vision_features = _encode_images(self.mm_model, self.mm_proj, pixel_values)

            img_attention = torch.ones(
                vision_features.shape[:2], device=attention_mask.device, dtype=attention_mask.dtype
            )
            fused_attention = torch.cat([img_attention, attention_mask], dim=1)
            fused_inputs = torch.cat([vision_features, hidden_states], dim=1)
            if labels is not None:
                label_pad = torch.full(
                    vision_features.shape[:2], -100, device=labels.device, dtype=labels.dtype
                )
                labels = torch.cat([label_pad, labels], dim=1)

            fusion_outputs = self.fusion_layer(
                attention_mask=fused_attention,
                inputs_embeds=fused_inputs,
                use_cache=bool(use_cache),
                output_hidden_states=False,
                return_dict=True,
            )
            if use_cache:
                self._fusion_kv = fusion_outputs.past_key_values
            logits = self.lm_head(fusion_outputs.last_hidden_state).float()

        elif decode_step:
            cached_len = self._fusion_kv[0][0].shape[2]
            fused_attention = torch.cat(
                [attention_mask, torch.ones_like(attention_mask)], dim=1
            )[:, : cached_len + 1]
            fusion_outputs = self.fusion_layer(
                attention_mask=fused_attention,
                inputs_embeds=hidden_states,
                past_key_values=self._fusion_kv,
                use_cache=True,
                return_dict=True,
            )
            self._fusion_kv = fusion_outputs.past_key_values
            logits = self.lm_head(fusion_outputs.last_hidden_state).float()

        loss = None
        if scores is not None and (prompt_step or decode_step):
            seq_len = base_logits.shape[1]
            logits = logits[:, -seq_len:, :]
            w = scores.view(-1, 1, 1)
            logits = w * logits + (1 - w) * base_logits.repeat(logits.shape[0], 1, 1)
            logits = logits.mean(dim=0, keepdim=True)
            if labels is not None:
                labels = labels[:1, -seq_len:]

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1).to(shift_logits.device),
                ignore_index=-100,
            )

        if not return_dict:
            output = (logits,) + tuple(v for v in (base_outputs.past_key_values,) if v is not None)
            return (loss, *output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=base_outputs.past_key_values,
            hidden_states=base_outputs.hidden_states,
            attentions=base_outputs.attentions,
        )


class MaskedLMFusionMixin(_FusionInit):
    """Late multi-image fusion for HF ``XForMaskedLM`` (BERT-family).

    Assumes the subclass exposes ``self.bert`` (the encoder) and ``self.cls``
    (the MLM head). The fusion layer is a 1-layer encoder built from a copy of
    the same config.
    """

    base_attr: str = "bert"
    head_attr: str = "cls"

    @property
    def _base(self) -> nn.Module:
        return getattr(self, self.base_attr)

    @property
    def _head(self) -> nn.Module:
        return getattr(self, self.head_attr)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        **base_kwargs,
    ) -> Union[Tuple, MaskedLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        base_outputs = self._base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=False,
            return_dict=True,
            **base_kwargs,
        )
        hidden_states = base_outputs.last_hidden_state

        if pixel_values is not None:
            k = pixel_values.shape[0]
            batch = hidden_states.shape[0]
            if k != batch:
                if k % batch != 0:
                    raise ValueError(
                        f"pixel_values batch ({k}) must be a multiple of text batch ({batch})"
                    )
                times = k // batch
                hidden_states = _repeat_along_batch(hidden_states, times)
                attention_mask = _repeat_along_batch(attention_mask, times)
                labels = _repeat_along_batch(labels, times)

            vision_features = _encode_images(self.mm_model, self.mm_proj, pixel_values)
            img_attention = torch.ones(
                vision_features.shape[:2], device=attention_mask.device, dtype=attention_mask.dtype
            )
            fused_attention = torch.cat([img_attention, attention_mask], dim=1)
            fused_inputs = torch.cat([vision_features, hidden_states], dim=1)
            if labels is not None:
                label_pad = torch.full(
                    vision_features.shape[:2], -100, device=labels.device, dtype=labels.dtype
                )
                labels = torch.cat([label_pad, labels], dim=1)

            fusion_outputs = self.fusion_layer(
                inputs_embeds=fused_inputs,
                attention_mask=fused_attention,
                return_dict=True,
            )
            hidden_states = fusion_outputs.last_hidden_state

        prediction_scores = self._head(hidden_states)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        if not return_dict:
            return (loss, prediction_scores) if loss is not None else (prediction_scores,)
        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=base_outputs.hidden_states,
            attentions=base_outputs.attentions,
        )
