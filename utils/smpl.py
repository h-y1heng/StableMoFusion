
import torch
from tqdm import tqdm

import numpy as np
from utils.transforms import *

import pickle
from typing import Optional
# import smplx
# from smplx.lbs import vertices2joints
import os
# from smplx import SMPL as _SMPL
# from smplx.body_models import ModelOutput

smpl_joints = [
    "root",  # 0
    "lhip",  # 1
    "rhip",  # 2
    "belly", # 3
    "lknee", # 4
    "rknee", # 5
    "spine", # 6
    "lankle",# 7
    "rankle",# 8
    "chest", # 9
    "ltoes", # 10
    "rtoes", # 11
    "neck",  # 12
    "linshoulder", # 13
    "rinshoulder", # 14
    "head", # 15
    "lshoulder", # 16
    "rshoulder",  # 17
    "lelbow", # 18
    "relbow",  # 19
    "lwrist", # 20
    "rwrist", # 21
    # "lhand", # 22
    # "rhand", # 23
]

smpl_parents = [
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    # 20,
    # 21,
]

smpl_offsets = [
    [0.0, 0.0, 0.0],
    [0.05858135, -0.08228004, -0.01766408],
    [-0.06030973, -0.09051332, -0.01354254],
    [0.00443945, 0.12440352, -0.03838522],
    [0.04345142, -0.38646945, 0.008037],
    [-0.04325663, -0.38368791, -0.00484304],
    [0.00448844, 0.1379564, 0.02682033],
    [-0.01479032, -0.42687458, -0.037428],
    [0.01905555, -0.4200455, -0.03456167],
    [-0.00226458, 0.05603239, 0.00285505],
    [0.04105436, -0.06028581, 0.12204243],
    [-0.03483987, -0.06210566, 0.13032329],
    [-0.0133902, 0.21163553, -0.03346758],
    [0.07170245, 0.11399969, -0.01889817],
    [-0.08295366, 0.11247234, -0.02370739],
    [0.01011321, 0.08893734, 0.05040987],
    [0.12292141, 0.04520509, -0.019046],
    [-0.11322832, 0.04685326, -0.00847207],
    [0.2553319, -0.01564902, -0.02294649],
    [-0.26012748, -0.01436928, -0.03126873],
    [0.26570925, 0.01269811, -0.00737473],
    [-0.26910836, 0.00679372, -0.00602676],
    # [0.08669055, -0.01063603, -0.01559429],
    # [-0.0887537, -0.00865157, -0.01010708],
]


def set_line_data_3d(line, x):
    line.set_data(x[:, :2].T)
    line.set_3d_properties(x[:, 2])


def set_scatter_data_3d(scat, x, c):
    scat.set_offsets(x[:, :2])
    scat.set_3d_properties(x[:, 2], "z")
    scat.set_facecolors([c])


def get_axrange(poses):
    pose = poses[0]
    x_min = pose[:, 0].min()
    x_max = pose[:, 0].max()

    y_min = pose[:, 1].min()
    y_max = pose[:, 1].max()

    z_min = pose[:, 2].min()
    z_max = pose[:, 2].max()

    xdiff = x_max - x_min
    ydiff = y_max - y_min
    zdiff = z_max - z_min

    biggestdiff = max([xdiff, ydiff, zdiff])
    return biggestdiff


def plot_single_pose(num, poses, lines, ax, axrange, scat, contact):
    pose = poses[num]
    static = contact[num]
    indices = [7, 8, 10, 11]

    for i, (point, idx) in enumerate(zip(scat, indices)):
        position = pose[idx : idx + 1]
        color = "r" if static[i] else "g"
        set_scatter_data_3d(point, position, color)

    for i, (p, line) in enumerate(zip(smpl_parents, lines)):
        # don't plot root
        if i == 0:
            continue
        # stack to create a line
        data = np.stack((pose[i], pose[p]), axis=0)
        set_line_data_3d(line, data)

    if num == 0:
        if isinstance(axrange, int):
            axrange = (axrange, axrange, axrange)
        xcenter, ycenter, zcenter = 0, 0, 2.5
        stepx, stepy, stepz = axrange[0] / 2, axrange[1] / 2, axrange[2] / 2

        x_min, x_max = xcenter - stepx, xcenter + stepx
        y_min, y_max = ycenter - stepy, ycenter + stepy
        z_min, z_max = zcenter - stepz, zcenter + stepz

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)


class SMPLSkeleton:
    def __init__(
        self, device=None,
    ):
        offsets = smpl_offsets
        parents = smpl_parents
        assert len(offsets) == len(parents)

        self._offsets = torch.Tensor(offsets).to(device)
        self._parents = np.array(parents)
        self._compute_metadata()

    def _compute_metadata(self):
        self._has_children = np.zeros(len(self._parents)).astype(bool)
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._has_children[parent] = True

        self._children = []
        for i, parent in enumerate(self._parents):
            self._children.append([])
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._children[parent].append(i)

    def forward(self, rotations, root_positions):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where N = batch size, L = sequence length, J = number of joints):
         -- rotations: (N, L, J, 3) tensor of axis-angle rotations describing the local rotations of each joint.
         -- root_positions: (N, L, 3) tensor describing the root joint positions.
        """
        assert len(rotations.shape) == 4
        assert len(root_positions.shape) == 3
        # transform from axis angle to quaternion
        rotations = axis_angle_to_quaternion(rotations)

        positions_world = []
        rotations_world = []

        expanded_offsets = self._offsets.expand(
            rotations.shape[0],
            rotations.shape[1],
            self._offsets.shape[0],
            self._offsets.shape[1],
        )

        # Parallelize along the batch and time dimensions
        for i in range(self._offsets.shape[0]):
            if self._parents[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(rotations[:, :, 0])
            else:
                positions_world.append(
                    quaternion_apply(
                        rotations_world[self._parents[i]], expanded_offsets[:, :, i]
                    )
                    + positions_world[self._parents[i]]
                )
                if self._has_children[i]:
                    rotations_world.append(
                        quaternion_multiply(
                            rotations_world[self._parents[i]], rotations[:, :, i]
                        )
                    )
                else:
                    # This joint is a terminal node -> it would be useless to compute the transformation
                    rotations_world.append(None)

        return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2)


# class SMPL_old(smplx.SMPLLayer):
#     def __init__(self, *args, joint_regressor_extra: Optional[str] = None, update_hips: bool = False, **kwargs):
#         """
#         Extension of the official SMPL implementation to support more joints.
#         Args:
#             Same as SMPLLayer.
#             joint_regressor_extra (str): Path to extra joint regressor.
#         """
#         super(SMPL, self).__init__(*args, **kwargs)
#         smpl_to_openpose = [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
#                             7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
            
#         if joint_regressor_extra is not None:
#             self.register_buffer('joint_regressor_extra', torch.tensor(pickle.load(open(joint_regressor_extra, 'rb'), encoding='latin1'), dtype=torch.float32))
#         self.register_buffer('joint_map', torch.tensor(smpl_to_openpose, dtype=torch.long))
#         self.update_hips = update_hips

#     def forward(self, *args, **kwargs):
#         """
#         Run forward pass. Same as SMPL and also append an extra set of joints if joint_regressor_extra is specified.
#         """
#         smpl_output = super(SMPL, self).forward(*args, **kwargs)
#         joints = smpl_output.joints[:, self.joint_map, :]
#         if self.update_hips:
#             joints[:,[9,12]] = joints[:,[9,12]] + \
#                 0.25*(joints[:,[9,12]]-joints[:,[12,9]]) + \
#                 0.5*(joints[:,[8]] - 0.5*(joints[:,[9,12]] + joints[:,[12,9]]))
#         if hasattr(self, 'joint_regressor_extra'):
#             extra_joints = vertices2joints(self.joint_regressor_extra, smpl_output.vertices)
#             joints = torch.cat([joints, extra_joints], dim=1)
#         smpl_output.joints = joints
#         return smpl_output

# Map joints to SMPL joints
JOINT_MAP = {
    'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17,
    'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16,
    'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0,
    'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8,
    'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7,
    'OP REye': 25, 'OP LEye': 26, 'OP REar': 27,
    'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30,
    'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34,
    'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45,
    'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7,
    'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17,
    'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20,
    'Neck (LSP)': 47, 'Top of Head (LSP)': 48,
    'Pelvis (MPII)': 49, 'Thorax (MPII)': 50,
    'Spine (H36M)': 51, 'Jaw (H36M)': 52,
    'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26,
    'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27
}
JOINT_NAMES = [
    'OP Nose', 'OP Neck', 'OP RShoulder',
    'OP RElbow', 'OP RWrist', 'OP LShoulder',
    'OP LElbow', 'OP LWrist', 'OP MidHip',
    'OP RHip', 'OP RKnee', 'OP RAnkle',
    'OP LHip', 'OP LKnee', 'OP LAnkle',
    'OP REye', 'OP LEye', 'OP REar',
    'OP LEar', 'OP LBigToe', 'OP LSmallToe',
    'OP LHeel', 'OP RBigToe', 'OP RSmallToe', 'OP RHeel',
    'Right Ankle', 'Right Knee', 'Right Hip',
    'Left Hip', 'Left Knee', 'Left Ankle',
    'Right Wrist', 'Right Elbow', 'Right Shoulder',
    'Left Shoulder', 'Left Elbow', 'Left Wrist',
    'Neck (LSP)', 'Top of Head (LSP)',
    'Pelvis (MPII)', 'Thorax (MPII)',
    'Spine (H36M)', 'Jaw (H36M)',
    'Head (H36M)', 'Nose', 'Left Eye',
    'Right Eye', 'Left Ear', 'Right Ear'
]
BASE_DATA_DIR = "/data2/TSMC_data/base_data"
JOINT_IDS = {JOINT_NAMES[i]: i for i in range(len(JOINT_NAMES))}
JOINT_REGRESSOR_TRAIN_EXTRA = os.path.join(BASE_DATA_DIR, 'J_regressor_extra.npy')
SMPL_MEAN_PARAMS = os.path.join(BASE_DATA_DIR, 'smpl_mean_params.npz')
SMPL_MODEL_DIR = BASE_DATA_DIR
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]


# class SMPL(_SMPL):
#     """ Extension of the official SMPL implementation to support more joints """

#     def __init__(self, *args, **kwargs):
#         super(SMPL, self).__init__(*args, **kwargs)
#         joints = [JOINT_MAP[i] for i in JOINT_NAMES]
#         J_regressor_extra = np.load(JOINT_REGRESSOR_TRAIN_EXTRA)
#         self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))
#         self.joint_map = torch.tensor(joints, dtype=torch.long)


#     def forward(self, *args, **kwargs):
#         kwargs['get_skin'] = True
#         smpl_output = super(SMPL, self).forward(*args, **kwargs)
#         extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)
#         joints = torch.cat([smpl_output.joints, extra_joints], dim=1)
#         joints = joints[:, self.joint_map, :]
#         output = ModelOutput(vertices=smpl_output.vertices,
#                              global_orient=smpl_output.global_orient,
#                              body_pose=smpl_output.body_pose,
#                              joints=joints,
#                              betas=smpl_output.betas,
#                              full_pose=smpl_output.full_pose)
#         return output


# def get_smpl_faces():
#     print("Get SMPL faces")
#     smpl = SMPL(SMPL_MODEL_DIR, batch_size=1, create_transl=False)
#     return smpl.faces