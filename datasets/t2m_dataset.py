import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm.auto import tqdm
from utils.word_vectorizer import WordVectorizer, POS_enumerator
from utils.motion_process import recover_from_ric
class Text2MotionDataset(data.Dataset):
    """Dataset for Text2Motion generation task.

    """
    data_root = ''
    min_motion_len=40
    joints_num = None
    dim_pose = None
    max_motion_length = 196
    def __init__(self, opt, split, mode='train', accelerator=None):
        self.max_text_len = getattr(opt, 'max_text_len', 20)
        self.unit_length = getattr(opt, 'unit_length', 4)
        self.mode = mode
        motion_dir = pjoin(self.data_root, 'new_joint_vecs')
        text_dir = pjoin(self.data_root, 'texts')

        if mode not in ['train', 'eval','gt_eval','xyz_gt','hml_gt']:
            raise ValueError(f"Mode '{mode}' is not supported. Please use one of: 'train', 'eval', 'gt_eval', 'xyz_gt','hml_gt'.")
        
        mean, std = None, None
        if mode == 'gt_eval':
            print(pjoin(opt.eval_meta_dir, f'{opt.dataset_name}_std.npy'))
            # used by T2M models (including evaluators)
            mean = np.load(pjoin(opt.eval_meta_dir, f'{opt.dataset_name}_mean.npy'))
            std = np.load(pjoin(opt.eval_meta_dir, f'{opt.dataset_name}_std.npy'))
        elif mode in ['eval']:
            print(pjoin(opt.meta_dir, 'std.npy'))
            # used by our models during inference
            mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
            std = np.load(pjoin(opt.meta_dir, 'std.npy'))
        else:
            # used by our models during train
            mean = np.load(pjoin(self.data_root, 'Mean.npy'))
            std = np.load(pjoin(self.data_root, 'Std.npy'))
            
        if mode == 'eval':
            # used by T2M models (including evaluators)
            # this is to translate ours norms to theirs
            self.mean_for_eval = np.load(pjoin(opt.eval_meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std_for_eval = np.load(pjoin(opt.eval_meta_dir, f'{opt.dataset_name}_std.npy'))
        if mode in ['gt_eval','eval']:
            self.w_vectorizer = WordVectorizer(opt.glove_dir, 'our_vab')
        
        data_dict = {}
        id_list = []
        split_file = pjoin(self.data_root, f'{split}.txt')
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list,disable=not accelerator.is_local_main_process if accelerator is not None else False):
            try:
                motion = np.load(pjoin(motion_dir, name + '.npy'))
                if (len(motion)) < self.min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag
                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                            if (len(n_motion)) < self.min_motion_len or (len(n_motion) >= 200):
                                continue
                            new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                            while new_name in data_dict:
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                            data_dict[new_name] = {'motion': n_motion,
                                                    'length': len(n_motion),
                                                    'text':[text_dict]}
                            new_name_list.append(new_name)
                            length_list.append(len(n_motion))
                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text':text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                # Some motion may not exist in KIT dataset
                pass


        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        if mode=='train':
            if opt.dataset_name != 'amass':
                joints_num = self.joints_num
                # root_rot_velocity (B, seq_len, 1)
                std[0:1] = std[0:1] / opt.feat_bias
                # root_linear_velocity (B, seq_len, 2)
                std[1:3] = std[1:3] / opt.feat_bias
                # root_y (B, seq_len, 1)
                std[3:4] = std[3:4] / opt.feat_bias
                # ric_data (B, seq_len, (joint_num - 1)*3)
                std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
                # rot_data (B, seq_len, (joint_num - 1)*6)
                std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
                            joints_num - 1) * 9] / 1.0
                # local_velocity (B, seq_len, joint_num*3)
                std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
                                                                                        4 + (joints_num - 1) * 9: 4 + (
                                                                                                    joints_num - 1) * 9 + joints_num * 3] / 1.0
                # foot contact (B, seq_len, 4)
                std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
                                                                4 + (joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias

                assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            
            if accelerator is not None and accelerator.is_main_process:
                np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
                np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        self.data_dict = data_dict
        self.name_list = name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']

        # Randomly select a caption
        text_data = random.choice(text_list)
        caption = text_data['caption']

        "Z Normalization"
        if self.mode not in['xyz_gt','hml_gt']:
            motion = (motion - self.mean) / self.std

        "crop motion"
        if self.mode in ['eval','gt_eval']:
            # Crop the motions in to times of 4, and introduce small variations
            if self.unit_length < 10:
                coin2 = np.random.choice(['single', 'single', 'double'])
            else:
                coin2 = 'single'
            if coin2 == 'double':
                m_length = (m_length // self.unit_length - 1) * self.unit_length
            elif coin2 == 'single':
                m_length = (m_length // self.unit_length) * self.unit_length
            idx = random.randint(0, len(motion) - m_length)
            motion = motion[idx:idx+m_length]
        elif m_length >= self.max_motion_length:
            idx = random.randint(0, len(motion) - self.max_motion_length)
            motion = motion[idx: idx + self.max_motion_length]
            m_length = self.max_motion_length
        
        "pad motion"
        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                        np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                        ], axis=0)
        assert len(motion) == self.max_motion_length


        if self.mode in ['gt_eval', 'eval']:
            "word embedding for text-to-motion evaluation"
            tokens = text_data['tokens']
            if len(tokens) < self.max_text_len:
                # pad with "unk"
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
                tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
            else:
                # crop
                tokens = tokens[:self.max_text_len]
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb, pos_oh = self.w_vectorizer[token]
                pos_one_hots.append(pos_oh[None, :])
                word_embeddings.append(word_emb[None, :])
            pos_one_hots = np.concatenate(pos_one_hots, axis=0)
            word_embeddings = np.concatenate(word_embeddings, axis=0)
            return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)
        elif self.mode in ['xyz_gt']:
            "Convert motion hml representation to skeleton points xyz"
            # 1. Use kn to get the keypoints position (the padding position after kn is all zero)
            motion = torch.from_numpy(motion).float()
            pred_joints = recover_from_ric(motion, self.joints_num)  # (nframe, njoints, 3)  

            # 2. Put on Floor (Y axis)
            floor_height = pred_joints.min(dim=0)[0].min(dim=0)[0][1]
            pred_joints[:, :, 1] -= floor_height
            return pred_joints
    
        
        return caption, motion, m_length

class HumanML3D(Text2MotionDataset):
    def __init__(self, opt, split="train", mode='train', accelerator=None):
        self.data_root = '/data/yiheng_huang/data/HumanML3D'
        self.min_motion_len = 40
        self.joints_num = 22
        self.dim_pose = 263
        self.max_motion_length = 196
        if accelerator:
            accelerator.print('\n Loading %s mode HumanML3D %s dataset ...' % (mode,split))
        else:
            print('\n Loading %s mode HumanML3D dataset ...' % mode)
        super(HumanML3D, self).__init__(opt, split, mode, accelerator)
        

class KIT(Text2MotionDataset):
    def __init__(self, opt, split="train", mode='train', accelerator=None):
        self.data_root = '/data/yiheng_huang/data/KIT-ML'
        self.min_motion_len = 24
        self.joints_num = 21
        self.dim_pose = 251
        self.max_motion_length = 196
        if accelerator:
            accelerator.print('\n Loading %s mode KIT %s dataset ...' % (mode,split))
        else:
            print('\n Loading %s mode KIT dataset ...' % mode)
        super(KIT, self).__init__(opt, split, mode, accelerator)


        
            

