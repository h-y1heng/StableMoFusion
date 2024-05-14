import torch
from utils.word_vectorizer import WordVectorizer
from torch.utils.data import Dataset, DataLoader
from os.path import join as pjoin
from tqdm import tqdm
import numpy as np
from eval.evaluator_modules import *

from torch.utils.data._utils.collate import default_collate


class GeneratedDataset(Dataset):
    """
    opt.dataset_name
    opt.max_motion_length
    opt.unit_length
    """

    def __init__(self, opt, pipeline, dataset, w_vectorizer, mm_num_samples, mm_num_repeats):
        assert mm_num_samples < len(dataset)
        self.dataset=dataset
        dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)
        generated_motion = []
        min_mov_length = 10 if opt.dataset_name == 't2m' else 6

        # Pre-process all target captions
        mm_generated_motions = []
        if mm_num_samples > 0:
            mm_idxs = np.random.choice(len(dataset), mm_num_samples, replace=False)
            mm_idxs = np.sort(mm_idxs)

        all_caption = []
        all_m_lens = []
        all_data = []
        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader)):
                word_emb, pos_ohot, caption, cap_lens, motions, m_lens, tokens = data
                all_data.append(data)
                tokens = tokens[0].split('_')
                mm_num_now = len(mm_generated_motions)
                is_mm = True if ((mm_num_now < mm_num_samples) and (i == mm_idxs[mm_num_now])) else False
                repeat_times = mm_num_repeats if is_mm else 1
                m_lens = max(torch.div(m_lens,opt.unit_length, rounding_mode='trunc') * opt.unit_length, min_mov_length * opt.unit_length)
                m_lens = min(m_lens, opt.max_motion_length)
                if isinstance(m_lens, int):
                    m_lens = torch.LongTensor([m_lens]).to(opt.device)
                else:
                    m_lens = m_lens.to(opt.device)
                for t in range(repeat_times):
                    all_m_lens.append(m_lens)
                    all_caption.extend(caption)
                if is_mm:
                    mm_generated_motions.append(0)
        all_m_lens = torch.stack(all_m_lens)
        
        # Generate all sequences
        with torch.no_grad():
            all_pred_motions,t_eval = pipeline.generate(all_caption, all_m_lens)
        self.eval_generate_time = t_eval
        
        cur_idx = 0
        mm_generated_motions = []
        with torch.no_grad():
            for i, data_dummy in tqdm(enumerate(dataloader)):
                data = all_data[i]
                word_emb, pos_ohot, caption, cap_lens, motions, m_lens, tokens = data
                tokens = tokens[0].split('_')
                mm_num_now = len(mm_generated_motions)
                is_mm = True if ((mm_num_now < mm_num_samples) and (i == mm_idxs[mm_num_now])) else False
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                for t in range(repeat_times):
                    pred_motions = all_pred_motions[cur_idx] 
                    cur_idx += 1
                    if t == 0:
                        sub_dict = {'motion': pred_motions.cpu().numpy(),
                                    'length': pred_motions.shape[0], #m_lens[0].item(), # 
                                    'caption': caption[0],
                                    'cap_len': cap_lens[0].item(),
                                    'tokens': tokens}
                        generated_motion.append(sub_dict)

                    if is_mm:
                        mm_motions.append({
                            'motion': pred_motions.cpu().numpy(),
                            'length': pred_motions.shape[0] , #m_lens[0].item(), #m_lens[0].item()
                        })
                if is_mm:
                    mm_generated_motions.append({'caption': caption[0],
                                                 'tokens': tokens,
                                                 'cap_len': cap_lens[0].item(),
                                                 'mm_motions': mm_motions})
        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.opt = opt
        self.w_vectorizer = w_vectorizer


    def __len__(self):
        return len(self.generated_motion)


    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
        sent_len = data['cap_len']

        # This step is needed because T2M evaluators expect their norm convention
        normed_motion = motion
        denormed_motion = self.dataset.inv_transform(normed_motion)
        renormed_motion = (denormed_motion - self.dataset.mean_for_eval) / self.dataset.std_for_eval  # according to T2M norms
        motion = renormed_motion
          
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)
        length=len(motion)
        if length < self.opt.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.opt.max_motion_length - length, motion.shape[1]))
                                     ], axis=0)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


class MMGeneratedDataset(Dataset):
    def __init__(self, opt, motion_dataset, w_vectorizer):
        self.opt = opt
        self.dataset = motion_dataset.mm_generated_motion
        self.w_vectorizer = w_vectorizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        mm_motions = data['mm_motions']
        m_lens = []
        motions = []
        for mm_motion in mm_motions:
            m_lens.append(mm_motion['length'])
            motion = mm_motion['motion']
            if len(motion) < self.opt.max_motion_length:
                motion = np.concatenate([motion,
                                         np.zeros((self.opt.max_motion_length - len(motion), motion.shape[1]))
                                         ], axis=0)
            motion = motion[None, :]
            motions.append(motion)
        m_lens = np.array(m_lens, dtype=np.int32)
        motions = np.concatenate(motions, axis=0)
        sort_indx = np.argsort(m_lens)[::-1].copy()

        m_lens = m_lens[sort_indx]
        motions = motions[sort_indx]
        return motions, m_lens

def get_motion_loader(opt, batch_size, pipeline, ground_truth_dataset, mm_num_samples, mm_num_repeats):

    # Currently the configurations of two datasets are almost the same
    if opt.dataset_name == 't2m' or opt.dataset_name == 'kit':
        w_vectorizer = WordVectorizer(opt.glove_dir, 'our_vab')
    else:
        raise KeyError('Dataset not recognized!!')

    dataset = GeneratedDataset(opt, pipeline, ground_truth_dataset, w_vectorizer, mm_num_samples, mm_num_repeats)
    mm_dataset = MMGeneratedDataset(opt, dataset, w_vectorizer)

    motion_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, drop_last=True, num_workers=4)
    mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=1)

    return motion_loader, mm_motion_loader,dataset.eval_generate_time
