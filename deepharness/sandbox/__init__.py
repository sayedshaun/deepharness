"""Execution environments for commands: what a command can reach while it runs.

Separate from tools because nothing here is a tool: a tool may call a sandbox,
and so may a graph node with no model involved. Each backend is its own module,
so adding one (a cloud VM, say) touches nothing else.
"""

from .base import Sandbox
from .bubblewrap import Bubblewrap

__all__ = ["Bubblewrap", "Sandbox"]
