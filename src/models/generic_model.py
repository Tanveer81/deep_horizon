"""
Generic class for models.
"""

from typing import Any, Mapping


class ModelFactory:
    """Regression Model Factory."""

    @staticmethod
    def search_space() -> Mapping[str, Any]:
        """The models search space."""
        raise NotImplementedError

    @staticmethod
    def from_config(config: Mapping[str, Any]) -> Any:
        """Factory method."""
        raise NotImplementedError
