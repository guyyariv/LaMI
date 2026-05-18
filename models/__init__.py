"""Factory for LaMI late-fusion model variants.

Usage:
    from models import build_fusion_model
    model = build_fusion_model("meta-llama/Meta-Llama-3-8B", torch_dtype=torch.bfloat16)
"""

from __future__ import annotations

from typing import Optional

from transformers import AutoConfig, AutoTokenizer

from .bert import BertFusionForMaskedLM
from .fusion import CausalLMFusionMixin, MaskedLMFusionMixin, MMProj
from .gemma import GemmaFusionForCausalLM
from .gpt2 import GPT2FusionLMHeadModel
from .llama3 import LlamaFusionForCausalLM
from .opt import OPTFusionForCausalLM

__all__ = [
    "BertFusionForMaskedLM",
    "GPT2FusionLMHeadModel",
    "GemmaFusionForCausalLM",
    "LlamaFusionForCausalLM",
    "MMProj",
    "OPTFusionForCausalLM",
    "build_fusion_model",
    "build_tokenizer",
]


_MODEL_REGISTRY = [
    ("llama", LlamaFusionForCausalLM),
    ("gemma", GemmaFusionForCausalLM),
    ("gpt2", GPT2FusionLMHeadModel),
    ("opt", OPTFusionForCausalLM),
    ("bert", BertFusionForMaskedLM),
]


def _select_class(model_name: str):
    name = model_name.lower()
    for key, cls in _MODEL_REGISTRY:
        if key in name:
            return cls
    raise ValueError(
        f"No LaMI variant registered for '{model_name}'. "
        f"Supported keys: {[k for k, _ in _MODEL_REGISTRY]}."
    )


def build_fusion_model(model_name: str, config=None, **from_pretrained_kwargs):
    """Instantiate the LaMI variant matching ``model_name`` from a HF checkpoint."""
    cls = _select_class(model_name)
    if config is None:
        config = AutoConfig.from_pretrained(model_name)
    if config.pad_token_id is None and "gemma" in model_name.lower():
        config.pad_token_id = 0
    if not hasattr(config, "ignore_index"):
        config.ignore_index = -100
    return cls.from_pretrained(model_name, config=config, **from_pretrained_kwargs)


def build_tokenizer(model_name: str, use_fast: bool = True, trust_remote_code: bool = False):
    """Tokenizer with the small LaMI-specific tweaks (image token for Gemma, pad=eos for Llama)."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=use_fast, trust_remote_code=trust_remote_code
    )
    name = model_name.lower()
    if "gemma" in name:
        tokenizer.add_tokens("<image>")
    if "llama" in name or "gpt2" in name:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
