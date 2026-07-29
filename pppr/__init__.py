"""PPPR: Person Parametric Physics-informed Representation for mmWave HPE."""

from .representation import JointGaussian, PPPRInstance, pack_pppr, unpack_pppr
from .mhp import MHPOptimizer, optimize_frame
from .reconstruction import pppr_to_heatmap, pppr_to_pointcloud

__version__ = "1.0.0"
__all__ = [
    "JointGaussian",
    "PPPRInstance",
    "pack_pppr",
    "unpack_pppr",
    "MHPOptimizer",
    "optimize_frame",
    "pppr_to_heatmap",
    "pppr_to_pointcloud",
]
