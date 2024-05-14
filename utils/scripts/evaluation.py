import sys
import torch
from motion_loader import get_dataset_loader, get_motion_loader
from datasets import get_dataset
from models import build_models
from eval import EvaluatorModelWrapper,evaluation
from utils.utils import *
from utils.model_load import load_model_weights
import os
from os.path import join as pjoin

from models.gaussian_diffusion import DiffusePipeline
from accelerate.utils import set_seed

from options.evaluate_options import TestOptions



if __name__ == '__main__':
    parser = TestOptions()
    opt = parser.parse()
    set_seed(0)

    device_id = opt.gpu_id
    device = torch.device('cuda:%d' % device_id if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    opt.device = device

    # load evaluator
    eval_wrapper = EvaluatorModelWrapper(opt)

    # load dataset
    gt_loader = get_dataset_loader(opt, opt.batch_size, mode='gt_eval',split='test')
    gen_dataset = get_dataset(opt, mode='eval',split='test')

    # load model
    model = build_models(opt)
    ckpt_path = pjoin(opt.model_dir, opt.which_ckpt + '.tar')  
    load_model_weights(model, ckpt_path, use_ema=not opt.no_ema, device=device)

    # Create a pipeline for generation in diffusion model framework
    pipeline = DiffusePipeline(
        opt = opt,
        model = model, 
        diffuser_name = opt.diffuser_name, 
        device=device,
        num_inference_steps=opt.num_inference_steps,
        torch_dtype=torch.float32 if opt.no_fp16 else torch.float16)

    eval_motion_loaders = {
        'text2motion': lambda: get_motion_loader(
            opt,
            opt.batch_size,
            pipeline,
            gen_dataset,
            opt.mm_num_samples,
            opt.mm_num_repeats,
        )
    }

    save_dir = pjoin(opt.save_root,'eval') 
    os.makedirs(save_dir, exist_ok=True)
    if opt.no_ema:
        log_file = pjoin(save_dir,opt.diffuser_name)+f'_{str(opt.num_inference_steps)}setps.log'
    else:
        log_file = pjoin(save_dir,opt.diffuser_name)+f'_{str(opt.num_inference_steps)}steps_ema.log'
    
    if not os.path.exists(log_file):
        config_dict = dict(pipeline.scheduler.config)
        config_dict['no_ema'] = opt.no_ema
        with open(log_file, 'wt') as f:
            f.write('------------ Options -------------\n')
            for k, v in sorted(config_dict.items()):
                f.write('%s: %s\n' % (str(k), str(v)))
            f.write('-------------- End ----------------\n')

    all_metrics = evaluation(eval_wrapper, gt_loader, eval_motion_loaders, log_file, opt.replication_times, opt.diversity_times, opt.mm_num_times, run_mm=True)

