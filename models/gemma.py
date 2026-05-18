from transformers import GemmaConfig, GemmaForCausalLM
from transformers.models.gemma.modeling_gemma import GemmaModel

from .fusion import CausalLMFusionMixin


class GemmaFusionForCausalLM(CausalLMFusionMixin, GemmaForCausalLM):
    fusion_model_cls = GemmaModel
    fusion_config_cls = GemmaConfig
    base_attr = "model"

    def __init__(self, config):
        super().__init__(config)
        self._init_fusion_modules(config)
