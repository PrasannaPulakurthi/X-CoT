import os
import torch
import random
import numpy as np
from config.all_config import AllConfig
from torch.utils.tensorboard.writer import SummaryWriter
from datasets.data_factory import DataFactory
from model.model_factory import ModelFactory
from modules.metrics import t2v_metrics, v2t_metrics
from modules.loss import LossFactory
from trainer.trainer_xpool import Trainer
import time
import gdown

def main():
    config = AllConfig()
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    if not config.no_tensorboard:
        writer = SummaryWriter(log_dir=config.tb_log_dir)
    else:
        writer = None


    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if config.huggingface:
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", TOKENIZERS_PARALLELISM=False)
    else:
        from modules.tokenization_clip import SimpleTokenizer
        tokenizer = SimpleTokenizer()

    test_data_loader  = DataFactory.get_data_loader(config, split_type='test')
    model = ModelFactory.get_model(config)
    
    if config.metric == 't2v':
        metrics = t2v_metrics
    elif config.metric == 'v2t':
        metrics = v2t_metrics
    else:
        raise NotImplemented
    
    loss = LossFactory.get_loss(config)

    trainer = Trainer(model, loss, metrics, None,
                      config=config,
                      train_data_loader=None,
                      valid_data_loader=test_data_loader,
                      lr_scheduler=None,
                      writer=writer,
                      tokenizer=tokenizer)

    if config.load_epoch is not None:
        # Check if the model exists
        if not os.path.exists(os.path.join(config.model_path, "model_best.pth")):
            print(f"Downloading X-Pool Model for {config.dataset_name} dataset.")
            if config.dataset_name == "MSRVTT":
                gdown.download("https://drive.google.com/file/d/1porWLljGecfExL3H3-ucl_4-z7mY-ZKL/view", "outputs/MSRVTT/model_best.pth", quiet=False, fuzzy=True)
            elif config.dataset_name == "MSVD":
                gdown.download("https://drive.google.com/file/d/1IdCAmUBo8ScKtLKGnLfvz77RSfDZOBAu/view", "outputs/MSVD/model_best.pth", quiet=False)
            elif config.dataset_name == "DiDeMo":
                gdown.download("https://drive.google.com/file/d/1Hr47o5Wb0e2jZj-mR16SIcy312NjlWFO/view", "outputs/DiDeMo/model_best.pth", quiet=False)
            elif config.dataset_name == "LSMDC":
                gdown.download("https://drive.google.com/file/d/12f7jJ63OnywTAg431I4eLRRrJRo7zJI7/view", "outputs/LSMDC/model_best.pth", quiet=False)
            else:
                print("X-Pool model for {config.dataset_name} dataset is not avilable.")
        if config.load_epoch > 0:
            trainer.load_checkpoint("checkpoint-epoch{}.pth".format(config.load_epoch))
        else:
            trainer.load_checkpoint("model_best.pth")    
    start_time = time.perf_counter()  # <-- start timer
    trainer.validate()

    end_time = time.perf_counter()  # <-- end timer
    elapsed = end_time - start_time

    print(elapsed)

if __name__ == '__main__':
    main()

