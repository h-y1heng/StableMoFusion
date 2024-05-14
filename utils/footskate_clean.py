from utils.motion_process import recover_from_ric
import torch
from utils.kinematics import HybrIKJointsToRotmat, HybrIKJointsToRotmat_Tensor
from utils.transforms import matrix_to_axis_angle
from UnderPressure.demo import retarget_to_underpressure, Skeletons, FRAMERATE
from UnderPressure.demo import TOPOLOGY
from UnderPressure import util
from UnderPressure import anim
from UnderPressure.data import Contacts
from UnderPressure.metrics import MSLE
import torch.optim as optim
from os.path import join as pjoin
from utils.utils import *
from tqdm import tqdm

def sigmoid_like(x, degree=2):
    m = (x > 0.5).float()
    s = 1 - 2 * m
    return m + 0.5 * s * (2 * (m + s * x))**degree

def weights(t, m):
    w = sigmoid_like(torch.arange(m) / (m-1), degree=2)
    if t >= 2 * m:
        return torch.cat([w, torch.ones(t - 2 * m), (1 - w)])
    else:
        return torch.cat([w[:t // 2 + t % 1], (1 - w)[-t // 2:]])

def footskate_clean(pred_motions, opt, epoch, underpressure_model, model, JOINT_NAMES):
    """
    transfer the footskate cleanup function of underpressure to our motion
    """
    device = pred_motions.device

    mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
    std = np.load(pjoin(opt.meta_dir, 'std.npy'))
    pred_motions = pred_motions.detach().cpu().numpy() * std + mean
    pred_motions = torch.from_numpy(pred_motions).float()
    pred_joints = recover_from_ric(pred_motions, opt.joints_num)

    floor_height = pred_joints.min(dim=0)[0].min(dim=0)[0][1]
    pred_joints[:, :, 1] -= floor_height  # Put on Floor Y axis

    # skeleton to underpressure
    framerate = opt.fps
    keypoints3d = pred_joints.clone()
    keypoints3d_gt = keypoints3d.clone()
    joints2rotmat = HybrIKJointsToRotmat_Tensor()
    print("keypoints3d_gt.shape:", keypoints3d_gt.shape)
    pose_gt = joints2rotmat(keypoints3d_gt)
    pose_gt = matrix_to_axis_angle(pose_gt)
    trajectory_gt = keypoints3d_gt[1:, 0, :] - keypoints3d_gt[:-1, 0, :]
    # Align with the orientation of the UnderPressure data
    rotation_matrix = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float32)
    nframes, njoints, _ = pred_joints.shape
    expanded_rotation_matrix = rotation_matrix.unsqueeze(0).repeat(nframes, 1, 1)
    pred_joints = torch.einsum('nij,nkj->nki', expanded_rotation_matrix, pred_joints)
    # Retargeting to UnderPressure skeleton
    skeleton = Skeletons.all()[0]  # (23, 3)
    angles, trajectory = retarget_to_underpressure(
        pred_joints,
        JOINT_NAMES,
        niters=150,
        skeleton=skeleton,
    )
    # resample angles and trajectory from input framerate 'framerate' to FRAMERATE 100
    out_nframes = round(trajectory.shape[-3] / framerate * FRAMERATE)
    angles = util.resample(angles, out_nframes, dim=-3, interpolation_fn=util.SU2.slerp)
    trajectory = util.resample(trajectory, out_nframes)
    skeleton = skeleton.unsqueeze(0)
    # predict vGRF
    positions = anim.FK(angles, skeleton, trajectory, TOPOLOGY)
    vGRFs_init = underpressure_model.vGRFs(positions.unsqueeze(0)).detach()
    vGRFs_init = util.resample(vGRFs_init, nframes, dim=-3)
    contacts = Contacts.from_forces(vGRFs_init)  # (batch, nframes, 2, 2)
    contact_ranges = util.nonzero_ranges(contacts, dim=-3)
    contact_weights = {}
    contact_locations = [[[[] for lr in [0, 1]] for fb in [0, 1]] for i in range(len(contact_ranges))]
    foot_joints = [
        ["left_foot", "right_foot"],
        ["left_ankle", "right_ankle"]
    ]
    foot_joints_jidxs = torch.as_tensor([[JOINT_NAMES.index(joint) for joint in joints] for joints in foot_joints])
    keypoints3d = keypoints3d.unsqueeze(0)
    for i in range(len(contact_ranges)):   #  len(contact_ranges) = batch
        for fb, lr in [[0, 0], [0, 1], [1, 0], [1, 1]]:  # fb=0:foot  fb=1:ankle  lr=0:left   lr=1:right
            for j in range(len(contact_ranges[i][fb][lr])):
                start, stop = contact_ranges[i][fb][lr][j].tolist()  # range of consecutive frames
                x = keypoints3d[i, (start + stop) // 2, foot_joints_jidxs[fb, lr], 0]
                z = keypoints3d[i, (start + stop) // 2, foot_joints_jidxs[fb, lr], 2]
                y = torch.zeros_like(x)  # (1,)  the y value of the ground is 0
                contact_locations[i][fb][lr].append(torch.tensor([x, y, z]))
                length = stop - start
                contact_weights[length] = contact_weights.get(length, weights(length, 5))

    pred_motions.requires_grad_()
    pred_motions = torch.nn.Parameter(pred_motions)
    optimizer = optim.Adam([pred_motions], lr=1e-3)

    vGRFs_init = model.vGRFs(keypoints3d).detach()  # (batch, nframes, 2, 16)
    mse = torch.nn.MSELoss()
    for i in tqdm(range(epoch),desc='optimizing footskate',leave=False):
        pred_joints = recover_from_ric(pred_motions, opt.joints_num)
        floor_height = pred_joints.min(dim=0)[0].min(dim=0)[0][1]
        pred_joints[:, :, 1] -= floor_height
        pose = joints2rotmat(pred_joints)
        pose = matrix_to_axis_angle(pose)
        trajectory = pred_joints[1:, 0, :] - pred_joints[:-1, 0, :]
        footloss = 0
        for j in range(len(contact_ranges)):  # batch
            # compute loss of contact_joint
            for fb, lr in [[0,0], [0,1], [1,0], [1,1]]:
                for r, target in zip(contact_ranges[j][fb][lr], contact_locations[j][fb][lr]):
                    dist2 = (pred_joints[r[0]:r[1], foot_joints_jidxs[fb][lr], :] - target).square().sum(dim=-1)
                    footloss += (contact_weights[(r[1]-r[0]).item()] * dist2).mean()

        footloss = float(1e-5) * footloss
        vGRFs_pred = model.vGRFs(pred_joints.unsqueeze(0))
        vGRFloss = float(5e-5) * MSLE(vGRFs_init, vGRFs_pred)
        trajectory_loss = float(1e2) * mse(trajectory, trajectory_gt)
        pose_loss = float(1e-3) * mse(pose, pose_gt)

        loss = footloss + vGRFloss + trajectory_loss + pose_loss
        # print(f'i: {i}, footloss: {footloss:.2f}, vGRFloss: {vGRFloss:.2f}, trajectory_loss: {trajectory_loss:.2f}, pose_loss: {pose_loss:.2f}, loss: {loss:.2f}')

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    pred_motions = (pred_motions.detach().cpu().numpy() - mean) / std
    pred_motions = torch.from_numpy(pred_motions)
    pred_motions = pred_motions.to(device)
    pred_motions = pred_motions.unsqueeze(0)
    return pred_motions
