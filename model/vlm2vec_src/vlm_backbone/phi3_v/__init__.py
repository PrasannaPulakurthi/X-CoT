from transformers.models.auto.processing_auto import PROCESSOR_MAPPING
from .configuration_phi3_v import Phi3VConfig
from .processing_phi3_v import Phi3VProcessor

PROCESSOR_MAPPING[Phi3VConfig] = Phi3VProcessor

# 还需要添加到 AUTO_MAPPING
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
CONFIG_MAPPING["phi3_v"] = Phi3VConfig