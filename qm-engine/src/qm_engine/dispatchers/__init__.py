"""Task dispatchers — pluggable strategies for executing successor nodes."""

from qm_engine.dispatchers.base import TaskDispatcher
from qm_engine.dispatchers.sync_dispatcher import SyncDispatcher
from qm_engine.dispatchers.thread_dispatcher import ThreadDispatcher

__all__ = ["TaskDispatcher", "SyncDispatcher", "ThreadDispatcher"]
