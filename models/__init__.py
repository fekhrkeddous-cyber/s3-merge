"""
S³-Merge: Spatially-Restricted Single-Shot Token Merging

Core modules for training-free token compression in Vision Transformers.
"""

from .sr_tome import (
    sr_tome_merge_exact,
    SRToMeBlock,
    PropAttnBlock,
    apply_patch,
    create_sr_tome_v3,
)

from .sr_tome_downstream import (
    sr_tome_merge_with_unmerge_cached,
    sr_tome_unmerge,
    SRToMeDownstream,
    create_sr_tome_downstream,
    GridMappingCache,
)

__version__ = "1.0.0"

__all__ = [
    # Classification
    "sr_tome_merge_exact",
    "SRToMeBlock",
    "PropAttnBlock",
    "apply_patch",
    "create_sr_tome_v3",
    # Dense prediction (detection/segmentation)
    "sr_tome_merge_with_unmerge_cached",
    "sr_tome_unmerge",
    "SRToMeDownstream",
    "create_sr_tome_downstream",
    "GridMappingCache",
]
