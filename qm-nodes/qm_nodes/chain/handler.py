"""Base handler abstract class for the chain pattern."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class Handler(ABC):
    """Abstract base class for chain handlers.

    Each handler receives a data dictionary, processes it, and returns
    the modified dictionary. Handlers are composed into chains for
    sequential processing.
    """

    @abstractmethod
    def handle(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the data and return the modified result.

        Args:
            data: Dictionary containing the processing state.

        Returns:
            Modified data dictionary.
        """
        pass
