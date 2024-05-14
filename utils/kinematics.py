from typing import Tuple

import numpy as np

import utils.constants as constants
import torch

class HybrIKJointsToRotmat:
    def __init__(self):
        self.naive_hybrik = constants.SMPL_HYBRIK
        self.num_nodes = 22
        self.parents = constants.SMPL_BODY_PARENTS
        self.child = constants.SMPL_BODY_CHILDS
        self.bones = np.array(constants.SMPL_BODY_BONES).reshape(24, 3)[
            : self.num_nodes
        ]

    def multi_child_rot(
        self, t: np.ndarray, p: np.ndarray, pose_global_parent: np.ndarray
    ) -> Tuple[np.ndarray]:
        """
        t: B x 3 x child_num
        p: B x 3 x child_num
        pose_global_parent: B x 3 x 3
        """
        m = np.matmul(
            t, np.transpose(np.matmul(np.linalg.inv(pose_global_parent), p), [0, 2, 1])
        )
        u, s, vt = np.linalg.svd(m)
        r = np.matmul(np.transpose(vt, [0, 2, 1]), np.transpose(u, [0, 2, 1]))
        err_det_mask = (np.linalg.det(r) < 0.0).reshape(-1, 1, 1)
        id_fix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]).reshape(
            1, 3, 3
        )
        r_fix = np.matmul(
            np.transpose(vt, [0, 2, 1]), np.matmul(id_fix, np.transpose(u, [0, 2, 1]))
        )
        r = r * (1.0 - err_det_mask) + r_fix * err_det_mask
        return r, np.matmul(pose_global_parent, r)

    def single_child_rot(
        self,
        t: np.ndarray,
        p: np.ndarray,
        pose_global_parent: np.ndarray,
        twist: np.ndarray = None,
    ) -> Tuple[np.ndarray]:
        """
        t: B x 3 x 1
        p: B x 3 x 1
        pose_global_parent: B x 3 x 3
        twist: B x 2 if given, default to None
        """
        p_rot = np.matmul(np.linalg.inv(pose_global_parent), p)
        cross = np.cross(t, p_rot, axisa=1, axisb=1, axisc=1)
        sina = np.linalg.norm(cross, axis=1, keepdims=True) / (
            np.linalg.norm(t, axis=1, keepdims=True)
            * np.linalg.norm(p_rot, axis=1, keepdims=True)
        )
        cross = cross / np.linalg.norm(cross, axis=1, keepdims=True)
        cosa = np.sum(t * p_rot, axis=1, keepdims=True) / (
            np.linalg.norm(t, axis=1, keepdims=True)
            * np.linalg.norm(p_rot, axis=1, keepdims=True)
        )
        sina = sina.reshape(-1, 1, 1)
        cosa = cosa.reshape(-1, 1, 1)
        skew_sym_t = np.stack(
            [
                0.0 * cross[:, 0],
                -cross[:, 2],
                cross[:, 1],
                cross[:, 2],
                0.0 * cross[:, 0],
                -cross[:, 0],
                -cross[:, 1],
                cross[:, 0],
                0.0 * cross[:, 0],
            ],
            1,
        )
        skew_sym_t = skew_sym_t.reshape(-1, 3, 3)
        dsw_rotmat = (
            np.eye(3).reshape(1, 3, 3)
            + sina * skew_sym_t
            + (1.0 - cosa) * np.matmul(skew_sym_t, skew_sym_t)
        )
        if twist is not None:
            skew_sym_t = np.stack(
                [
                    0.0 * t[:, 0],
                    -t[:, 2],
                    t[:, 1],
                    t[:, 2],
                    0.0 * t[:, 0],
                    -t[:, 0],
                    -t[:, 1],
                    t[:, 0],
                    0.0 * t[:, 0],
                ],
                1,
            )
            skew_sym_t = skew_sym_t.reshape(-1, 3, 3)
            sina = twist[:, 1].reshape(-1, 1, 1)
            cosa = twist[:, 0].reshape(-1, 1, 1)
            dtw_rotmat = (
                np.eye(3).reshape([1, 3, 3])
                + sina * skew_sym_t
                + (1.0 - cosa) * np.matmul(skew_sym_t, skew_sym_t)
            )
            dsw_rotmat = np.matmul(dsw_rotmat, dtw_rotmat)
        return dsw_rotmat, np.matmul(pose_global_parent, dsw_rotmat)

    def __call__(self, joints: np.ndarray, twist: np.ndarray = None) -> np.ndarray:
        """
        joints: B x N x 3
        twist: B x N x 2 if given, default to None
        """
        expand_dim = False
        if len(joints.shape) == 2:
            expand_dim = True
            joints = np.expand_dims(joints, 0)
            if twist is not None:
                twist = np.expand_dims(twist, 0)
        assert len(joints.shape) == 3
        batch_size = np.shape(joints)[0]
        joints_rel = joints - joints[:, self.parents]
        joints_hybrik = 0.0 * joints_rel
        pose_global = np.zeros([batch_size, self.num_nodes, 3, 3])
        pose = np.zeros([batch_size, self.num_nodes, 3, 3])
        for i in range(self.num_nodes):
            if i == 0:
                joints_hybrik[:, 0] = joints[:, 0]
            else:
                joints_hybrik[:, i] = (
                    np.matmul(
                        pose_global[:, self.parents[i]],
                        self.bones[i].reshape(1, 3, 1),
                    ).reshape(-1, 3)
                    + joints_hybrik[:, self.parents[i]]
                )
            if self.child[i] == -2:
                pose[:, i] = pose[:, i] + np.eye(3).reshape(1, 3, 3)
                pose_global[:, i] = pose_global[:, self.parents[i]]
                continue
            if i == 0:
                r, rg = self.multi_child_rot(
                    np.transpose(self.bones[[1, 2, 3]].reshape(1, 3, 3), [0, 2, 1]),
                    np.transpose(joints_rel[:, [1, 2, 3]], [0, 2, 1]),
                    np.eye(3).reshape(1, 3, 3),
                )

            elif i == 9:
                r, rg = self.multi_child_rot(
                    np.transpose(self.bones[[12, 13, 14]].reshape(1, 3, 3), [0, 2, 1]),
                    np.transpose(joints_rel[:, [12, 13, 14]], [0, 2, 1]),
                    pose_global[:, self.parents[9]],
                )
            else:
                p = joints_rel[:, self.child[i]]
                if self.naive_hybrik[i] == 0:
                    p = joints[:, self.child[i]] - joints_hybrik[:, i]
                twi = None
                if twist is not None:
                    twi = twist[:, i]
                r, rg = self.single_child_rot(
                    self.bones[self.child[i]].reshape(1, 3, 1),
                    p.reshape(-1, 3, 1),
                    pose_global[:, self.parents[i]],
                    twi,
                )
            pose[:, i] = r
            pose_global[:, i] = rg
        if expand_dim:
            pose = pose[0]
        return pose

class HybrIKJointsToRotmat_Tensor:
    def __init__(self):
        self.naive_hybrik = constants.SMPL_HYBRIK
        self.num_nodes = 22
        self.parents = constants.SMPL_BODY_PARENTS
        self.child = constants.SMPL_BODY_CHILDS
        self.bones = torch.tensor(constants.SMPL_BODY_BONES).reshape(24, 3)[:self.num_nodes]

    def multi_child_rot(self, t, p, pose_global_parent):
        """
        t: B x 3 x child_num
        p: B x 3 x child_num
        pose_global_parent: B x 3 x 3
        """
        m = torch.matmul(
            t, torch.transpose(torch.matmul(torch.inverse(pose_global_parent), p), 1, 2)
        )
        u, s, vt = torch.linalg.svd(m)
        r = torch.matmul(torch.transpose(vt, 1, 2), torch.transpose(u, 1, 2))
        err_det_mask = (torch.det(r) < 0.0).reshape(-1, 1, 1)
        id_fix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]).reshape(1, 3, 3)
        r_fix = torch.matmul(
            torch.transpose(vt, 1, 2), torch.matmul(id_fix, torch.transpose(u, 1, 2))
        )
        r = r * (~err_det_mask) + r_fix * err_det_mask
        return r, torch.matmul(pose_global_parent, r)

    def single_child_rot(
            self,
            t,
            p,
            pose_global_parent,
            twist = None,
    ) -> Tuple[torch.Tensor]:
        """
        t: B x 3 x 1
        p: B x 3 x 1
        pose_global_parent: B x 3 x 3
        twist: B x 2 if given, default to None
        """
        t_tensor = t.clone().detach()#torch.tensor(t)
        p_tensor = p.clone().detach()#torch.tensor(p)
        pose_global_parent_tensor = pose_global_parent.clone().detach()#torch.tensor(pose_global_parent)

        p_rot = torch.matmul(torch.linalg.inv(pose_global_parent_tensor), p_tensor)
        cross = torch.cross(t_tensor, p_rot, dim=1)
        sina = torch.linalg.norm(cross, dim=1, keepdim=True) / (
                torch.linalg.norm(t_tensor, dim=1, keepdim=True)
                * torch.linalg.norm(p_rot, dim=1, keepdim=True)
        )
        cross = cross / torch.linalg.norm(cross, dim=1, keepdim=True)
        cosa = torch.sum(t_tensor * p_rot, dim=1, keepdim=True) / (
                torch.linalg.norm(t_tensor, dim=1, keepdim=True)
                * torch.linalg.norm(p_rot, dim=1, keepdim=True)
        )
        sina = sina.reshape(-1, 1, 1)
        cosa = cosa.reshape(-1, 1, 1)
        skew_sym_t = torch.stack(
            [
                0.0 * cross[:, 0],
                -cross[:, 2],
                cross[:, 1],
                cross[:, 2],
                0.0 * cross[:, 0],
                -cross[:, 0],
                -cross[:, 1],
                cross[:, 0],
                0.0 * cross[:, 0],
            ],
            1,
        )
        skew_sym_t = skew_sym_t.reshape(-1, 3, 3)
        dsw_rotmat = (
                torch.eye(3).reshape(1, 3, 3)
                + sina * skew_sym_t
                + (1.0 - cosa) * torch.matmul(skew_sym_t, skew_sym_t)
        )
        if twist is not None:
            twist_tensor = torch.tensor(twist)
            skew_sym_t = torch.stack(
                [
                    0.0 * t_tensor[:, 0],
                    -t_tensor[:, 2],
                    t_tensor[:, 1],
                    t_tensor[:, 2],
                    0.0 * t_tensor[:, 0],
                    -t_tensor[:, 0],
                    -t_tensor[:, 1],
                    t_tensor[:, 0],
                    0.0 * t_tensor[:, 0],
                ],
                1,
            )
            skew_sym_t = skew_sym_t.reshape(-1, 3, 3)
            sina = twist_tensor[:, 1].reshape(-1, 1, 1)
            cosa = twist_tensor[:, 0].reshape(-1, 1, 1)
            dtw_rotmat = (
                    torch.eye(3).reshape([1, 3, 3])
                    + sina * skew_sym_t
                    + (1.0 - cosa) * torch.matmul(skew_sym_t, skew_sym_t)
            )
            dsw_rotmat = torch.matmul(dsw_rotmat, dtw_rotmat)

        return dsw_rotmat, torch.matmul(pose_global_parent_tensor, dsw_rotmat)

    def __call__(self, joints, twist = None) -> torch.Tensor:
        """
        joints: B x N x 3
        twist: B x N x 2 if given, default to None
        """
        expand_dim = False
        if len(joints.shape) == 2:
            expand_dim = True
            joints = joints.unsqueeze(0)
            if twist is not None:
                twist = twist.unsqueeze(0)
        assert len(joints.shape) == 3
        batch_size = joints.shape[0]
        joints_rel = joints - joints[:, self.parents]
        joints_hybrik = torch.zeros_like(joints_rel)
        pose_global = torch.zeros([batch_size, self.num_nodes, 3, 3])
        pose = torch.zeros([batch_size, self.num_nodes, 3, 3])
        for i in range(self.num_nodes):
            if i == 0:
                joints_hybrik[:, 0] = joints[:, 0]
            else:
                joints_hybrik[:, i] = (
                        torch.matmul(
                            pose_global[:, self.parents[i]],
                            self.bones[i].reshape(1, 3, 1),
                        ).reshape(-1, 3)
                        + joints_hybrik[:, self.parents[i]]
                )
            if self.child[i] == -2:
                pose[:, i] = pose[:, i] + torch.eye(3).reshape(1, 3, 3)
                pose_global[:, i] = pose_global[:, self.parents[i]]
                continue
            if i == 0:
                t = self.bones[[1, 2, 3]].reshape(1, 3, 3).permute(0, 2, 1)
                p = joints_rel[:, [1, 2, 3]].permute(0, 2, 1)
                pose_global_parent = torch.eye(3).reshape(1, 3, 3)
                r, rg = self.multi_child_rot(t, p, pose_global_parent)
            elif i == 9:
                t = self.bones[[12, 13, 14]].reshape(1, 3, 3).permute(0, 2, 1)
                p = joints_rel[:, [12, 13, 14]].permute(0, 2, 1)
                r, rg = self.multi_child_rot(t, p, pose_global[:, self.parents[9]],)
            else:
                p = joints_rel[:, self.child[i]]
                if self.naive_hybrik[i] == 0:
                    p = joints[:, self.child[i]] - joints_hybrik[:, i]
                twi = None
                if twist is not None:
                    twi = twist[:, i]
                t = self.bones[self.child[i]].reshape(-1, 3, 1)
                p = p.reshape(-1, 3, 1)
                nframes, _, _ = p.shape
                t = t.repeat(nframes, 1, 1)
                r, rg = self.single_child_rot(t, p, pose_global[:, self.parents[i]], twi)
            pose[:, i] = r
            pose_global[:, i] = rg
        if expand_dim:
            pose = pose[0]
        return pose


if __name__ == "__main__":
    jts2rot_hybrik = HybrIKJointsToRotmat_Tensor()
    joints = torch.tensor(constants.SMPL_BODY_BONES).reshape(1, 24, 3)[:, :22]
    parents = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
    for i in range(1, 22):
        joints[:, i] = joints[:, i] + joints[:, parents[i]]
    print(joints.shape)
    pose = jts2rot_hybrik(joints)
    print(pose.shape)
