"""Dashboard settings to ComfyUI graph translation.

``graph`` is the node builder, ``image`` and ``video`` turn tab settings into
prompts, ``regional`` implements regional prompting, ``seedvr`` the optional
SeedVR2 upscaler stage.
"""
from __future__ import annotations

from .graph import Graph, Node, merge, ui_workflow_to_prompt, validate
from .image import BuildResult, build_image_graph
from .video import (DEFAULT_FPS, FAMILIES, GRIDS, MODES, MODE_LABELS, build_video_graph,
                    modes_for, snap_frames)

__all__ = ["Graph", "Node", "validate", "merge", "ui_workflow_to_prompt", "BuildResult",
           "build_image_graph", "build_video_graph", "modes_for", "snap_frames", "FAMILIES",
           "MODES", "MODE_LABELS", "GRIDS", "DEFAULT_FPS"]
