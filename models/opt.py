from transformers import OPTConfig, OPTForCausalLM
from transformers.models.opt.modeling_opt import OPTModel

from .fusion import CausalLMFusionMixin


class OPTFusionForCausalLM(CausalLMFusionMixin, OPTForCausalLM):
    fusion_model_cls = OPTModel
    fusion_config_cls = OPTConfig
    base_attr = "model"

    def __init__(self, config):
        super().__init__(config)
        self._init_fusion_modules(config)
