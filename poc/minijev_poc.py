"""Compatibility module: the POC code now lives in the minijev package (src/minijev). Old imports keep working:

    from minijev_poc import Engine, Settings, ask
"""

from minijev import *  # noqa: F401,F403
from minijev import __all__  # noqa: F401
from minijev.settings import _FLOAT_OR_FITTED  # noqa: F401
