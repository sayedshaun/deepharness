"""The one directory a harness tool is allowed to touch.

A tool that takes a path takes it from a model, and a model asked to read
"../../.ssh/id_rsa" will happily ask for it. Confinement therefore lives in one
object that every path-taking tool resolves through, rather than in a check each
tool is trusted to remember.
"""

from __future__ import annotations

from pathlib import Path

from ..errors import ConfigurationError, OutsideWorkspace


class Workspace:
    """A root directory, and the only paths that may be resolved against it.

    Resolution is symlink-aware on purpose: a link inside the root pointing out
    of it would otherwise be a way around the check, so the resolved target has
    to be inside the resolved root, not merely the path as written.
    """

    __slots__ = ("_root",)

    def __init__(self, root: str | Path = "."):
        resolved = Path(root).expanduser().resolve()
        if not resolved.is_dir():
            raise ConfigurationError(
                f"workspace root {resolved} is not an existing directory"
            )
        self._root = resolved

    @property
    def root(self) -> Path:
        return self._root

    def resolve(self, path: str) -> Path:
        """An absolute path inside the workspace, or OutsideWorkspace.

        Raising rather than clamping: a request for a path outside the root is
        a mistake worth telling the model about, and silently rewriting it to
        something inside would answer a question nobody asked.
        """
        candidate = (self._root / path).expanduser()
        resolved = candidate.resolve()
        if resolved != self._root and self._root not in resolved.parents:
            raise OutsideWorkspace(path, self._root)
        return resolved

    def relative(self, path: Path) -> str:
        """A resolved path as the model should see it: relative to the root."""
        return str(path.relative_to(self._root)) or "."
