import sys
import os
from os.path import join as pjoin
from options.train_options import TrainOptions
from utils.plot_script import *

from models import build_models
from utils.ema import ExponentialMovingAverage
from trainers import DDPMTrainer
from motion_loader import get_dataset_loader

from accelerate.utils import set_seed
from accelerate import Accelerator
import torch

if __name__ == '__main__':
    accelerator = Accelerator()
    
    parser = TrainOptions()
    opt = parser.parse(accelerator)
    set_seed(opt.seed)
    torch.autograd.set_detect_anomaly(True)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')

    if accelerator.is_main_process:
        os.makedirs(opt.model_dir, exist_ok=True)
        os.makedirs(opt.meta_dir, exist_ok=True)

    train_datasetloader = get_dataset_loader(opt,  batch_size = opt.batch_size, split='train', accelerator=accelerator, mode='train') # 7169


    accelerator.print('\nInitializing model ...' )
    encoder = build_models(opt)
    model_ema = None
    if opt.model_ema:
        # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        adjust = 106_667 * opt.model_ema_steps / opt.num_train_steps
        alpha = 1.0 - opt.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        print('EMA alpha:',alpha)
        model_ema = ExponentialMovingAverage(encoder, decay=1.0 - alpha)
    accelerator.print('Finish building Model.\n')

    trainer = DDPMTrainer(opt, encoder,accelerator, model_ema)

    trainer.train(train_datasetloader)


