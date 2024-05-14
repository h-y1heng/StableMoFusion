from diffusers import  DPMSolverMultistepScheduler, DDPMScheduler, DDIMScheduler, PNDMScheduler, DEISMultistepScheduler
import torch
import yaml
import math
import tqdm
from utils.footskate_clean import footskate_clean
from UnderPressure import models
from models import vGRFmodel
import os
from utils.paramUtil import HumanML3D_JOINT_NAMES, KIT_JOINT_NAMES

class DiffusePipeline(object):
    
    def __init__(self, opt, model, diffuser_name, num_inference_steps, device, torch_dtype=torch.float16):
        self.device = device
        self.torch_dtype = torch_dtype
        self.diffuser_name = diffuser_name
        self.num_inference_steps = num_inference_steps
        if self.torch_dtype == torch.float16:
            model = model.half()       
        self.model = model.to(device)
        self.opt=opt

        # Load footskate cleanup model
        self.load_footskate_cleanup_model()

        # Load parameters from YAML file
        with open('./config/diffuser_params.yaml', 'r') as yaml_file:
            diffuser_params = yaml.safe_load(yaml_file)

        # Select diffusion'parameters based on diffuser_name
        if diffuser_name in diffuser_params:
            params = diffuser_params[diffuser_name]
            scheduler_class_name = params['scheduler_class']
            additional_params = params['additional_params']

            # align training parameters
            additional_params['num_train_timesteps'] = opt.diffusion_steps
            additional_params['beta_schedule'] = opt.beta_schedule
            additional_params['prediction_type'] = opt.prediction_type

            try:
                scheduler_class = globals()[scheduler_class_name]
            except KeyError:
                raise ValueError(f"Class '{scheduler_class_name}' not found.")

            self.scheduler = scheduler_class(**additional_params)
        else:
            raise ValueError(f"Unsupported diffuser_name: {diffuser_name}")

    def load_footskate_cleanup_model(self):
        checkpoints_dir = os.path.join(os.getcwd(),"checkpoints", "footskate")
        underpressure_model_path = os.path.join(checkpoints_dir, "underpressure_pretrained.tar")
        underpressure_model = models.DeepNetwork(state_dict=torch.load(underpressure_model_path)["model"]).eval()
        underpressure_model.requires_grad_(False)
        mogen_model_path = os.path.join(checkpoints_dir, f'{self.opt.dataset_name}_pretrained.tar')
        mogen_model = vGRFmodel.DeepNetwork(joints_num=self.opt.joints_num, state_dict=torch.load(mogen_model_path)["model"]).eval()
        mogen_model.requires_grad_(False)
        if self.opt.dataset_name == 't2m':
            JOINT_NAMES = HumanML3D_JOINT_NAMES
        else:  # kit
            JOINT_NAMES = KIT_JOINT_NAMES
        self.underpressure_model = underpressure_model
        self.mogen_model = mogen_model
        self.JOINT_NAMES = JOINT_NAMES

    def generate_batch(self, caption, m_lens,footskate_cleanup=False):
        B = len(caption)
        T = m_lens.max()
        shape = (B, T, self.model.input_feats)

        # random sampling noise x_T
        sample = torch.randn(shape,device=self.device, dtype=self.torch_dtype)

        # set timesteps
        self.scheduler.set_timesteps(self.num_inference_steps, self.device)
        timesteps = [ torch.tensor([t] * B, device=self.device).long() for t in self.scheduler.timesteps]
        
        # cache text_embedded 
        enc_text = self.model.encode_text(caption, self.device)
            
        for i, t in enumerate(timesteps):
            # 1. model predict 
            with torch.no_grad():
                if  getattr(self.model, 'cond_mask_prob', 0) > 0 :
                    predict = self.model.forward_with_cfg(sample,t,enc_text=enc_text)
                else:
                    predict = self.model(sample, t, enc_text=enc_text)

            # 2. compute less noisy motion and set x_t -> x_t-1
            sample = self.scheduler.step(predict, t[0], sample).prev_sample

            # 3. footskate cleanup in the last 10% step
            if footskate_cleanup and i >= int(self.num_inference_steps * 0.9):  
                footskate_clean_sample = []
                for j in range(B):
                    n_guide_steps = 1000
                    clean_sample = footskate_clean(sample[j], self.opt, n_guide_steps,
                                                       self.underpressure_model, self.mogen_model, self.JOINT_NAMES)
                    footskate_clean_sample.append(clean_sample)
                footskate_clean_sample = torch.cat(footskate_clean_sample, dim=0)
                sample = footskate_clean_sample

        return sample

    def generate(self, caption, m_lens, batch_size=32,footskate_cleanup=False):
        N = len(caption)
        infer_mode = ''
        if  getattr(self.model, 'cond_mask_prob', 0) > 0:
            infer_mode = 'classifier-free-guidance'
        print(f'\nUsing {self.diffuser_name} diffusion scheduler to {infer_mode} generate {N} motions, sampling {self.num_inference_steps} steps.')
        self.model.eval()

        all_output = []
        cur_idx=0
        for bacth_idx in tqdm.tqdm(range(math.ceil(N/batch_size))):
            if cur_idx + batch_size >= N:
                batch_caption = caption[cur_idx:]
                batch_m_lens = m_lens[cur_idx:]
            else:
                batch_caption = caption[cur_idx: cur_idx + batch_size]
                batch_m_lens = m_lens[cur_idx: cur_idx + batch_size]

            output = self.generate_batch(batch_caption, batch_m_lens,footskate_cleanup=footskate_cleanup)
        
            # Crop motion with gt/predicted motion length
            B = output.shape[0]
            for i in range(B):
                all_output.append(output[i,:batch_m_lens[i]])

            cur_idx += batch_size

        return all_output

