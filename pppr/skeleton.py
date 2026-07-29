"""Human skeleton topology, bone lengths, and joint-angle limits (Sec. 4.3)."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

# COCO-17 joint names used throughout PPPR
JOINT_NAMES: List[str] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

NUM_JOINTS = len(JOINT_NAMES)

# Bone edges E (Sec. 4.3)
SKELETON_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

# Reference bone lengths ell_mn (metres), initialized from dataset averages
SKELETON_LENGTHS: Dict[Tuple[int, int], float] = {
    (0, 1): 0.08,
    (0, 2): 0.08,
    (1, 3): 0.08,
    (2, 4): 0.08,
    (0, 5): 0.25,
    (0, 6): 0.25,
    (5, 6): 0.35,
    (5, 7): 0.30,
    (7, 9): 0.25,
    (6, 8): 0.30,
    (8, 10): 0.25,
    (5, 11): 0.45,
    (6, 12): 0.45,
    (11, 12): 0.25,
    (11, 13): 0.45,
    (13, 15): 0.45,
    (12, 14): 0.45,
    (14, 16): 0.45,
}

# Joint-angle triplets A = (m, n, o) with vertex n (Sec. 4.3)
JOINT_ANGLE_TRIPLETS: List[Tuple[int, int, int]] = [
    (5, 7, 9),   # left arm
    (6, 8, 10),  # right arm
    (11, 13, 15),  # left leg
    (12, 14, 16),  # right leg
    (5, 6, 12),  # shoulder-hip chain
    (6, 5, 11),
    (5, 11, 13),
    (6, 12, 14),
]

# Per-joint isotropic scale priors for anisotropic Gaussian init
JOINT_SCALES: Dict[str, float] = {
    "nose": 0.08,
    "left_eye": 0.05,
    "right_eye": 0.05,
    "left_ear": 0.05,
    "right_ear": 0.05,
    "left_shoulder": 0.12,
    "right_shoulder": 0.12,
    "left_elbow": 0.08,
    "right_elbow": 0.08,
    "left_wrist": 0.05,
    "right_wrist": 0.05,
    "left_hip": 0.12,
    "right_hip": 0.12,
    "left_knee": 0.09,
    "right_knee": 0.09,
    "left_ankle": 0.06,
    "right_ankle": 0.06,
}


def t_pose_positions(distance: float = 1.5, height: float = 1.7) -> torch.Tensor:
    """Canonical T-pose joint positions in radar coordinates (x forward)."""
    shoulder_w = 0.35
    hip_w = 0.25
    head = 0.15
    ankle_h = 0.05
    knee_h = height * 0.285
    hip_h = height * 0.53
    shoulder_h = height * 0.82
    nose_h = height * 0.95
    d = distance
    return torch.tensor(
        [
            [d, 0.0, nose_h],
            [d, -head / 2, nose_h + head / 4],
            [d, head / 2, nose_h + head / 4],
            [d, -head, nose_h],
            [d, head, nose_h],
            [d, -shoulder_w / 2, shoulder_h],
            [d, shoulder_w / 2, shoulder_h],
            [d, -shoulder_w, shoulder_h - 0.15],
            [d, shoulder_w, shoulder_h - 0.15],
            [d, -shoulder_w * 1.5, shoulder_h - 0.3],
            [d, shoulder_w * 1.5, shoulder_h - 0.3],
            [d, -hip_w / 2, hip_h],
            [d, hip_w / 2, hip_h],
            [d, -hip_w / 2, knee_h],
            [d, hip_w / 2, knee_h],
            [d, -hip_w / 2, ankle_h],
            [d, hip_w / 2, ankle_h],
        ],
        dtype=torch.float32,
    )


def bone_length_tensor(device=None) -> torch.Tensor:
    lengths = torch.zeros(len(SKELETON_CONNECTIONS), dtype=torch.float32, device=device)
    for i, (a, b) in enumerate(SKELETON_CONNECTIONS):
        key = (min(a, b), max(a, b))
        lengths[i] = SKELETON_LENGTHS.get(key, SKELETON_LENGTHS.get((a, b), 0.3))
    return lengths
