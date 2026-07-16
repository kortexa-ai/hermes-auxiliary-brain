"""Standalone Hermes plugin entry point."""

if __package__:
    from .auxiliary_brain.plugin import register
else:  # pytest imports a repository-root __init__.py as a plain module
    from auxiliary_brain.plugin import register

__all__ = ["register"]
