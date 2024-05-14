import os
from argparse import Namespace
import re
from os.path import join as pjoin



def is_float(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')
    try:
        reg = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')
        res = reg.match(str(numStr))
        if res:
            flag = True
    except Exception as ex:
        print("is_float() - error: " + str(ex))
    return flag


def is_number(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')
    if str(numStr).isdigit():
        flag = True
    return flag
  
def get_opt(opt, opt_path):
    opt_dict = vars(opt)

    skip = ('-------------- End ----------------',
            '------------ Options -------------',
            '\n')
    print('Reading', opt_path)
    with open(opt_path) as f:
        for line in f:
            if line.strip() not in skip:
                print(line.strip())
                key, value = line.strip().split(': ')
                if getattr(opt, key, None) is not None:
                    continue
                if value in ('True', 'False'):
                    opt_dict[key] = True if value == 'True' else False
                elif is_float(value):
                    opt_dict[key] = float(value)
                elif is_number(value):
                    opt_dict[key] = int(value)
                elif ',' in value:
                    value = value[1:-1].split(',')
                    opt_dict[key] = [int(i) for i in value]
                else:
                    opt_dict[key] = str(value)
    
    # opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.save_root =  os.path.dirname(opt_path)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')

    if opt.dataset_name == 't2m' or opt.dataset_name == 'humanml':
        opt.joints_num = 22
        opt.dim_pose = 263
        opt.max_motion_length = 196
        opt.radius = 4
        opt.fps = 20
    elif opt.dataset_name == 'kit':
        opt.joints_num = 21
        opt.dim_pose = 251
        opt.max_motion_length = 196
        opt.radius = 240 * 8
        opt.fps = 12.5
    else:
        raise KeyError('Dataset not recognized')

