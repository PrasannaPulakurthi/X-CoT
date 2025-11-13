from config.base_config import Config
from model.clip_baseline import CLIPBaseline
from model.clip_transformer import CLIPTransformer
try:
    from model.qwen2vl_vlm2vec import Qwen2VLAdvancePromptRetrieval
except:
    None
class ModelFactory:
    @staticmethod
    def get_model(config: Config):
        if config.arch == 'clip_baseline':
            return CLIPBaseline(config)
        elif config.arch == 'clip_transformer':
            return CLIPTransformer(config)
        elif config.arch == 'qwen2vl_vlm2vec':
            return Qwen2VLAdvancePromptRetrieval(config)
        else:
            raise NotImplemented
