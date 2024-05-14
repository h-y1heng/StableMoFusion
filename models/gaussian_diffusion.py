from diffusers import  DPMSolverMultistepScheduler, DDPMScheduler, DDIMScheduler, PNDMScheduler, DEISMultistepScheduler
import torch
import yaml
import math
import tqdm
import time

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
        
        # Load parameters from YAML file
        with open('config/diffuser_params.yaml', 'r') as yaml_file:
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

    def generate_batch(self, caption, m_lens):
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

        return sample

    def generate(self, caption, m_lens, batch_size=32):
        N = len(caption)
        infer_mode = ''
        if  getattr(self.model, 'cond_mask_prob', 0) > 0:
            infer_mode = 'classifier-free-guidance'
        print(f'\nUsing {self.diffuser_name} diffusion scheduler to {infer_mode} generate {N} motions, sampling {self.num_inference_steps} steps.')
        self.model.eval()

        all_output = []
        t_sum = 0
        cur_idx=0
        for bacth_idx in tqdm.tqdm(range(math.ceil(N/batch_size))):
            if cur_idx + batch_size >= N:
                batch_caption = caption[cur_idx:]
                batch_m_lens = m_lens[cur_idx:]
            else:
                batch_caption = caption[cur_idx: cur_idx + batch_size]
                batch_m_lens = m_lens[cur_idx: cur_idx + batch_size]
            torch.cuda.synchronize() 
            start_time = time.time()
            output = self.generate_batch(batch_caption, batch_m_lens)
            torch.cuda.synchronize() 
            now_time = time.time()
            
            # The average inference time is calculated after GPU warm-up in the first 50 steps.
            if (bacth_idx+1) * self.num_inference_steps >= 50:
                t_sum += now_time-start_time

            # Crop motion with gt/predicted motion length
            B = output.shape[0]
            for i in range(B):
                all_output.append(output[i,:batch_m_lens[i]])

            cur_idx += batch_size

        # calcalate average inference time
        t_eval = t_sum/(bacth_idx-1)
        print('The average generation time of a batch motion (bs=%d) is %f seconds'%(batch_size,t_eval))
        return all_output, t_eval

