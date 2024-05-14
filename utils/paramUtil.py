import numpy as np

# Define a kinematic tree for the skeletal struture
kit_kinematic_chain = [[0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4], [3, 5, 6, 7], [3, 8, 9, 10]]

kit_raw_offsets = np.array(
    [
        [0, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [-1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, 1]
    ]
)

t2m_raw_offsets = np.array([[0,0,0],
                           [1,0,0],
                           [-1,0,0],
                           [0,1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,1,0],
                           [0,0,1],
                           [0,0,1],
                           [0,1,0],
                           [1,0,0],
                           [-1,0,0],
                           [0,0,1],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0]])

t2m_kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
t2m_left_hand_chain = [[20, 22, 23, 24], [20, 34, 35, 36], [20, 25, 26, 27], [20, 31, 32, 33], [20, 28, 29, 30]]
t2m_right_hand_chain = [[21, 43, 44, 45], [21, 46, 47, 48], [21, 40, 41, 42], [21, 37, 38, 39], [21, 49, 50, 51]]

kit_kinematic_tree = [
    [0, 1, 2, 3, 4],  # body
    [3, 5, 6, 7],  # right arm
    [3, 8, 9, 10],  # left arm
    [0, 11, 12, 13, 14, 15],  # right leg
    [0, 16, 17, 18, 19, 20], # left leg
]  

humanml3d_kinematic_tree = [
    [0, 3, 6, 9, 12, 15],  # body
    [9, 14, 17, 19, 21],  # right arm
    [9, 13, 16, 18, 20],  # left arm
    [0, 2, 5, 8, 11],  # right leg
    [0, 1, 4, 7, 10],  # left leg
] 


kit_tgt_skel_id = '03950'

t2m_tgt_skel_id = '000021'

KIT_JOINT_NAMES = [
    "pelvis", # 0
    "spine_1", # 1
    "spine_3", # 2
    "neck", # 3
    "head", # 4
    "left_shoulder", # 5
    "left_elbow", # 6
    "left_wrist", # 7
    "right_shoulder", # 8
    "right_elbow", # 9
    "right_wrist", # 10
    "left_hip", # 11
    "left_knee", # 12
    "left_ankle", # 13
    "left_heel", # 14
    "left_foot", # 15
    "right_hip", # 16
    "right_knee", # 17
    "right_ankle", # 18
    "right_heel", # 19
    "right_foot", # 20
]

HumanML3D_JOINT_NAMES = [
    "pelvis",  # 0: root
    "left_hip",  # 1: lhip
    "right_hip",  # 2: rhip
    "spine_1",  # 3: belly
    "left_knee",  # 4: lknee
    "right_knee",  # 5: rknee
    "spine_2",  # 6: spine
    "left_ankle",  # 7: lankle
    "right_ankle",  # 8: rankle
    "spine_3",  # 9: chest
    "left_foot",  # 10: ltoes
    "right_foot",  # 11: rtoes
    "neck",  # 12: neck
    "left_clavicle",  # 13: linshoulder
    "right_clavicle",  # 14: rinshoulder
    "head",  # 15: head
    "left_shoulder",  # 16: lshoulder
    "right_shoulder",  # 17: rshoulder
    "left_elbow",  # 18: lelbow
    "right_elbow",  # 19: relbow
    "left_wrist",  # 20: lwrist
    "right_wrist",  # 21: rwrist
]

HumanML3D2KIT = {
    0:0,  # pelvis--pelvis
    1:16, # left_hip--right_hip
    2:11, # right_hip--left_hip
    3:1, # spine_1--spine_1
    4:17, # left_knee--right_knee
    5:12, # right_knee--left_knee
    6:1, # spine_2--spine_1
    7:18, # left_ankle--right_ankle
    8:13, # right_ankle--left_ankle
    9:2, # spine_3--spine_3
    10:20, # left_foot--right_foot
    11:15, # right_foot--left_foot
    12:3, # neck--neck
    13:8, # left_clavicle--right_shoulder
    14:5, # right_clavicle--left_shoulder
    15:4, # head--head
    16:8, # left_shoulder--right_shoulder
    17:5, # right_shoulder--left_shoulder
    18:9, # left_elbow--right_elbow
    19:6, # right_elbow--left_elbow
    20:10, # left_wrist--right_wrist
    21:7, # right_wrist--left_wrist
}