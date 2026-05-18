from transformers import GPT2Config, GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Model

from .fusion import CausalLMFusionMixin


class GPT2FusionLMHeadModel(CausalLMFusionMixin, GPT2LMHeadModel):
    fusion_model_cls = GPT2Model
    fusion_config_cls = GPT2Config
    base_attr = "transformer"

    def __init__(self, config):
        super().__init__(config)
        self._init_fusion_modules(config)
