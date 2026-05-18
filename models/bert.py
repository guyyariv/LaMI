from transformers import BertConfig, BertForMaskedLM
from transformers.models.bert.modeling_bert import BertModel

from .fusion import MaskedLMFusionMixin


class BertFusionForMaskedLM(MaskedLMFusionMixin, BertForMaskedLM):
    fusion_model_cls = BertModel
    fusion_config_cls = BertConfig
    base_attr = "bert"
    head_attr = "cls"

    def __init__(self, config):
        super().__init__(config)
        self._init_fusion_modules(config)
