"""Compatibility namespace for the LENS research workspace.

The canonical reusable implementation lives in ``lens_method/lens``.  This
module preserves the historical ``import lens`` API used by experiment runners
and immutable formal source snapshots.
"""

from lens_method.lens import *  # noqa: F401,F403
from lens_method.lens import __all__
