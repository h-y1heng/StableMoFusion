import math
from os.path import join as pjoin

import torch
import torch.optim as optim
import utils.paramUtil as paramUtil
from models import vGRFmodel
from UnderPressure import models, util, anim
from UnderPressure.demo import retarget_to_underpressure
from UnderPressure.data import TOPOLOGY
from UnderPressure.metrics import MSLE
from smplx import SMPLH
from utils.kinematics import HybrIKJointsToRotmat
from motion_loader import get_dataset_loader
import argparse
from utils.plot_script import *
import os

# visualize contact label of foot joints
def get_footlabels_underpressure_mogen(model, position):
    contacts = model.contacts(position.unsqueeze(0))  # (batch, nframes, 2, 2)
    contacts = contacts[0]  # (nframes, 2, 2)

    labels = []
    for frame in contacts:
        label = []
        ltoes = frame[0, 0]
        rtoes = frame[0, 1]
        lankle = frame[1, 0]
        rankle = frame[1, 1]
        if lankle:
            label.append("lankle")
        if rankle:
            label.append("rankle")
        if ltoes:
            label.append("ltoes")
        if rtoes:
            label.append("rtoes")
        if len(label) > 0:
            label = ','.join(label) + " contact ground"
        else:
            label = "No joints contact ground"
        labels.append(label)
    return labels



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='kit',help='Opt path')
    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument("--gpu_id", type=int, default=0, help='GPU id')
    opt = parser.parse_args()
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
    device = torch.device('cuda:%d' % opt.gpu_id if torch.cuda.is_available() else 'cpu')
    JOINT_NAMES = paramUtil.HumanML3D_JOINT_NAMES if (opt.dataset_name == 't2m') else paramUtil.KIT_JOINT_NAMES
    # 1. load dataset
    train_loader = get_dataset_loader(opt, opt.batch_size, mode='xyz_gt', split="val")  # train
    val_loader = get_dataset_loader(opt, opt.batch_size,mode='xyz_gt' ,split="test")  # val


    smplh = SMPLH(
        "./data/smplh",
        use_pca=False,
        num_betas=16,
    )
    smplh.requires_grad_(False)
    smplh.to(device=device)
    joints2rotmat = HybrIKJointsToRotmat()

    # 2. load underpressure model
    checkpoints_dir = os.path.join(os.getcwd(), "checkpoints", "footskate")
    underpressure_model_path = os.path.join(checkpoints_dir, "underpressure_pretrained.tar")
    underpressure_model = models.DeepNetwork(state_dict=torch.load(underpressure_model_path)["model"]).to(device).eval()
    skeleton = skeleton = torch.load('./UnderPressure/dataset/S1_HoppingLeftFootRightFoot.pth')["skeleton"].view(23,3).to(device)
    # 3. load mogen model (transfer underpressure to other dataset)
    underpressure_mogen_model = vGRFmodel.DeepNetwork(joints_num=opt.joints_num).to(device)
    # 4. train underpressure_mogen
    framerate = opt.fps  
    FRAMERATE = 100
    num_epochs = 100
    optimizer = optim.Adam(underpressure_mogen_model.parameters(), lr=0.0002)
    best_loss = math.inf
    for epoch in range(num_epochs):
        underpressure_mogen_model.train()
        total_loss = 0
        for step, positions in enumerate(train_loader):  # (batch, nframes, 22, 3)
            optimizer.zero_grad()
            batch, nframe, njoints, _ = positions.shape
            positions = positions.to(device)
            # underpressure_positions = underpressure_positions.to(device)
            # 4.1 underpressure_mogen output
            mogen_pred = underpressure_mogen_model.vGRFs(positions)

            # 4.2.1 align with the direction of underpressure
            rotation_matrix = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float32).to(device)
            reshaped_positions = positions.view(-1, njoints, 3)  # (batch*nframes, 22, 3)
            expanded_rotation_matrix = rotation_matrix.unsqueeze(0).repeat(reshaped_positions.size(0), 1, 1)
            rot_positions = torch.einsum('nij,nkj->nki', expanded_rotation_matrix, reshaped_positions)

            # 4.2.2 retarget motion to Underpressure skeleton: 23 joints
            angles, trajectory = retarget_to_underpressure(rot_positions, JOINT_NAMES, niters=150,
                                                                          skeleton=skeleton)
            angles = angles.view(batch, nframe, 23, 4)
            trajectory = trajectory.view(batch, nframe, 1, 3)

            # 4.2.3 Framerate 100
            out_nframes = round(trajectory.shape[-3] / framerate * FRAMERATE)
            angles = util.resample(angles, out_nframes, dim=-3, interpolation_fn=util.SU2.slerp)
            trajectory = util.resample(trajectory, out_nframes, dim=-3)
            underpressure_positions = anim.FK(angles, skeleton, trajectory, TOPOLOGY)

            # 4.3 underpressure output
            underpressure_pred = underpressure_model.vGRFs(underpressure_positions)
            # 4.4 resample to original framerate
            underpressure_pred = util.resample(underpressure_pred, nframe, dim=-3)
            loss = MSLE(underpressure_pred, mogen_pred)
            total_loss = total_loss + loss.item()
            print(f'epoch: {epoch}  step: {step}  loss:{loss}')
            loss.backward()
            optimizer.step()


        # checkpointing
        if best_loss > total_loss:
            best_loss = total_loss
            checkpoint_save_path = os.path.join(checkpoints_dir, f'{opt.dataset_name}_pretrained.tar')
            torch.save(dict(model=underpressure_mogen_model.state_dict()), checkpoint_save_path)
            # visualization checking
            underpressure_mogen_model.eval()
            for step, positions in enumerate(val_loader):
                positions = positions[0].to(device)
                labels = get_footlabels_underpressure_mogen(underpressure_mogen_model, positions)
                positions = positions.cpu().numpy()
                fname = f'{step:02}.mp4'
                kinematic_tree = paramUtil.t2m_kinematic_chain if (
                            opt.dataset_name == 't2m') else paramUtil.kit_kinematic_chain
                plot_3d_motion(pjoin(checkpoints_dir, fname), kinematic_tree, positions, title="", fps=opt.fps,
                               radius=opt.radius, label=labels)
                break
