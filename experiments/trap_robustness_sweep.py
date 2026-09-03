from importlib import import_module

_impl = import_module("experiments.05_trap_robustness_sweep")
main = _impl.main

__all__ = ["main"]
