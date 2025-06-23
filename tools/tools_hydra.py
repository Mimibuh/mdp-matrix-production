from types import ModuleType

from typeguard import typechecked

from typing import Callable, Any

"""
This module provides a utility to create a proxy function for Hydra configurations.
"""


@typechecked
def hydra_proxy(target_class, partial: bool = False) -> Callable[..., dict[str, Any]]:
    """Creates a proxy function for interacting with Hydra configurations."""

    target_class_name = _get_full_class_name(target_class)

    @typechecked
    def proxy_function(**params) -> dict[str, Any]:
        # Ensure no conflicting reserved keys in the arguments
        assert "_target_" not in params
        assert "_partial_" not in params
        assert "_convert_" not in params

        # Prepare Hydra-specific metadata
        if partial:
            params["_partial_"] = True

        # Combine all parameters and return
        return dict(_target_=target_class_name, **params)

    return proxy_function


def _get_full_class_name(target_class) -> str:
    """Returns the fully qualified name of the class or module."""
    if isinstance(target_class, ModuleType):
        return target_class.__name__

    module = target_class.__module__
    if module == "builtins":
        return target_class.__qualname__

    return f"{module}.{target_class.__qualname__}"
