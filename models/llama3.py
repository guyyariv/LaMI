from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaModel

from .fusion import CausalLMFusionMixin


class LlamaFusionForCausalLM(CausalLMFusionMixin, LlamaForCausalLM):
    fusion_model_cls = LlamaModel
    fusion_config_cls = LlamaConfig
    base_attr = "model"

    def __init__(self, config):
        super().__init__(config)
        self._init_fusion_modules(config)
