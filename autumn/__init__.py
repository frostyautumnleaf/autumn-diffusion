"""Autumn Diffusion - a local dashboard for image, video and music generation.

The package owns three things:

* the :mod:`autumn.installer` that puts ComfyUI and YuE inside the program dir,
* the :mod:`autumn.server` that serves the dashboard and proxies the engines,
* :mod:`autumn.workflow` that turns dashboard settings into ComfyUI graphs.

Nothing here assumes a GPU: every engine call goes through an interface that the
test-suite can mock.
"""
from __future__ import annotations

__app__ = "Autumn Diffusion"
__version__ = "0.1.0"
__all__ = ["__app__", "__version__"]
