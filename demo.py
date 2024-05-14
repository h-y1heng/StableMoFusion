import gradio as gr
import torch
import numpy as np
from os.path import join as pjoin
import utils.paramUtil as paramUtil
from utils.plot_script import *

from utils.utils import *
from utils.motion_process import recover_from_ric
from accelerate.utils import set_seed
from models.gaussian_diffusion_w_footskate_cleanup import DiffusePipeline
from options.generate_options import GenerateOptions
from utils.model_load import load_model_weights
from motion_loader import get_dataset_loader
from models import build_models
from argparse import Namespace
from options.get_opt import get_opt
# 设置临时目录路径
os.environ['GRADIO_TEMP_DIR'] = '/data/yiheng_huang/StableMoFusion/tmp'

class GradioModel:
    def __init__(self) -> None:
        opt = Namespace()
        get_opt(opt, './checkpoints/t2m/t2m_condunet1d_batch64/opt.txt')
        model = build_models(opt)
        ckpt_path = pjoin(opt.model_dir, 'latest.tar')  
        niter = load_model_weights(model, ckpt_path, use_ema=True)
        device = torch.device('cuda:%d' % 0 if torch.cuda.is_available() else 'cpu')
        self.pipeline = DiffusePipeline(
            opt = opt,
            model = model, 
            diffuser_name = 'dpmsolver', 
            device=device,
            num_inference_steps=10,
            torch_dtype=torch.float16,)
        self.mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
        self.std = np.load(pjoin(opt.meta_dir, 'std.npy'))
    
    def generate_motion(self, texts, motion_lens,footskate_cleanup):
        pred_motions = self.pipeline.generate(texts, torch.LongTensor([int(x) for x in motion_lens]),footskate_cleanup=footskate_cleanup)

        
        for i, motion in enumerate(pred_motions):
            motion = motion.cpu().numpy() * self.std + self.mean
            # 1. recover 3d joints representation by ik
            motion = recover_from_ric(torch.from_numpy(motion).float(), 22)
            # 2. put on Floor (Y axis)
            floor_height = motion.min(dim=0)[0].min(dim=0)[0][1]
            motion[:, :, 1] -= floor_height
            motion = motion.numpy()
            # 3. remove jitter
            motion = motion_temporal_filter(motion, sigma=1)
            # 4. visualize
            kinematic_tree = paramUtil.t2m_kinematic_chain
            plot_3d_motion( f"./tmp/sample{i}_pred.mp4", kinematic_tree, motion, title="", fps=20, radius=4)
            print('success')

        
        


gradio_model = GradioModel()
def generate_motion(text_prompt, motion_length,footskate_cleanup):
    texts = [text_prompt]
    motion_lens = [motion_length * 20]
    gradio_model.generate_motion(texts, torch.LongTensor([int(x) for x in motion_lens]),footskate_cleanup=footskate_cleanup)
    return './tmp/sample0_pred.mp4'  # 返回视频文件的路径


if __name__ == "__main__":
   

    demo = gr.Interface(
        fn=generate_motion, 
        inputs=[
            gr.Textbox(label="Text Prompt"), 
            gr.Number(value=4,label="Motion Length (Seconds)", precision=0,
                minimum=2,
                maximum=10,),
            gr.Checkbox(label='cleanup footskate (this needs more time)')
        ], 
        outputs=gr.Video(label="Generated Motion Video"),
        title="Text-to-Motion Demo",
        description="Enter a text prompt and a motion length to generate a motion video. "
    )
    demo.launch()




