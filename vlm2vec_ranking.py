import os
import time
import torch
import random
import numpy as np
from config.all_config import AllConfig
# from torch.utils.tensorboard.writer import SummaryWriter
from datasets.data_factory import DataFactory
from model.model_factory import ModelFactory
from modules.metrics import t2v_metrics, v2t_metrics
from modules.loss import LossFactory
from trainer.trainer_vlm2vec import Trainer
from config.all_config import gen_log
# @WJM: solve num_workers
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def main():
    config = AllConfig()
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    # if not config.no_tensorboard:
    #     writer = SummaryWriter(log_dir=config.tb_log_dir)
    # else:
    writer = None

    if config.gpu is not None and config.gpu != '99':
        print('set GPU')
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if not torch.cuda.is_available():
            raise Exception('NO GPU!')

    # GPU
    # print(f"Visible devices: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}") # 1
    # print(f"Available GPUs: {torch.cuda.device_count()}") # 1

    # @WJM: for logging on RC/local
    msg = f'model pth = {config.model_path}'
    gen_log(model_path=config.model_path, log_name='log_trntst', msg=msg)
    msg = f'\nconfig={config.__dict__}\n'
    gen_log(model_path=config.model_path, log_name='log_trntst', msg=msg)
    gen_log(model_path=config.model_path, log_name='log_trntst', msg='\nrecord all training and testing results\n')

    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    test_data_loader  = DataFactory.get_data_loader(config, split_type='test')
    model = ModelFactory.get_model(config)
    print(f"Model initial device: {next(model.parameters()).device}") # 1
    
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
                      tokenizer=None)
    print(f"Model device after trainer init: {next(trainer.model.parameters()).device}")

    # if config.load_epoch is not None:
    #     if config.load_epoch > 0:
    #         trainer.load_checkpoint("checkpoint-epoch{}.pth".format(config.load_epoch))
    #     else:
    #         trainer.load_checkpoint("model_best.pth")
    start_time = time.time()
    trainer.validate()
    end_time   = time.time()
    msg = (f'>>>Total time usage of testing: {(end_time - start_time)/60.0} mins')
    gen_log(model_path=config.model_path, log_name='log_trntst', msg=msg)



if __name__ == '__main__':
    main()

