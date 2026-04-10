"""Tests for FlowMemory and PersistentMemory."""

from uuid import uuid4

from quartermaster_engine.memory.flow_memory import FlowMemory
from quartermaster_engine.memory.persistent_memory import InMemoryPersistence
from quartermaster_engine.stores.memory_store import InMemoryStore


class TestFlowMemory:
    def setup_method(self):
        self.store = InMemoryStore()
        self.flow_id = uuid4()
        self.memory = FlowMemory(self.flow_id, self.store)

    def test_set_and_get(self):
        self.memory.set("name", "Alice")
        assert self.memory.get("name") == "Alice"

    def test_get_default(self):
        assert self.memory.get("missing") is None
        assert self.memory.get("missing", "default") == "default"

    def test_delete(self):
        self.memory.set("key", "value")
        self.memory.delete("key")
        assert self.memory.get("key") is None

    def test_delete_nonexistent_key(self):
        # Should not raise
        self.memory.delete("nonexistent")

    def test_list_keys(self):
        self.memory.set("a", 1)
        self.memory.set("b", 2)
        self.memory.set("c", 3)
        keys = self.memory.list_keys()
        assert sorted(keys) == ["a", "b", "c"]

    def test_get_all(self):
        self.memory.set("x", 10)
        self.memory.set("y", 20)
        assert self.memory.get_all() == {"x": 10, "y": 20}

    def test_clear(self):
        self.memory.set("a", 1)
        self.memory.set("b", 2)
        self.memory.clear()
        assert self.memory.list_keys() == []

    def test_overwrite(self):
        self.memory.set("key", "v1")
        self.memory.set("key", "v2")
        assert self.memory.get("key") == "v2"

    def test_complex_values(self):
        self.memory.set("list", [1, 2, 3])
        self.memory.set("dict", {"nested": True})
        assert self.memory.get("list") == [1, 2, 3]
        assert self.memory.get("dict") == {"nested": True}

    def test_isolation_between_flows(self):
        other_flow_id = uuid4()
        other_memory = FlowMemory(other_flow_id, self.store)

        self.memory.set("key", "flow1")
        other_memory.set("key", "flow2")

        assert self.memory.get("key") == "flow1"
        assert other_memory.get("key") == "flow2"


class TestInMemoryPersistence:
    def setup_method(self):
        self.persistence = InMemoryPersistence()
        self.agent_id = uuid4()

    def test_write_and_read(self):
        self.persistence.write(self.agent_id, "fact", "The sky is blue")
        assert self.persistence.read(self.agent_id, "fact") == "The sky is blue"

    def test_read_nonexistent(self):
        assert self.persistence.read(self.agent_id, "missing") is None

    def test_update(self):
        self.persistence.write(self.agent_id, "mood", "happy")
        self.persistence.update(self.agent_id, "mood", "ecstatic")
        assert self.persistence.read(self.agent_id, "mood") == "ecstatic"

    def test_update_nonexistent_creates(self):
        self.persistence.update(self.agent_id, "new_key", "new_value")
        assert self.persistence.read(self.agent_id, "new_key") == "new_value"

    def test_delete(self):
        self.persistence.write(self.agent_id, "temp", "data")
        self.persistence.delete(self.agent_id, "temp")
        assert self.persistence.read(self.agent_id, "temp") is None

    def test_delete_nonexistent(self):
        # Should not raise
        self.persistence.delete(self.agent_id, "ghost")

    def test_search_by_key(self):
        self.persistence.write(self.agent_id, "user_name", "Alice")
        self.persistence.write(self.agent_id, "user_age", "30")
        self.persistence.write(self.agent_id, "mood", "happy")

        results = self.persistence.search(self.agent_id, "user")
        assert len(results) == 2
        assert {r.key for r in results} == {"user_name", "user_age"}

    def test_search_by_value(self):
        self.persistence.write(self.agent_id, "fact1", "Alice likes cats")
        self.persistence.write(self.agent_id, "fact2", "Bob likes dogs")

        results = self.persistence.search(self.agent_id, "cats")
        assert len(results) == 1
        assert results[0].key == "fact1"

    def test_search_case_insensitive(self):
        self.persistence.write(self.agent_id, "Name", "ALICE")
        results = self.persistence.search(self.agent_id, "alice")
        assert len(results) == 1

    def test_search_with_limit(self):
        for i in range(20):
            self.persistence.write(self.agent_id, f"key_{i}", f"value_{i}")
        results = self.persistence.search(self.agent_id, "value", limit=5)
        assert len(results) == 5

    def test_list_keys(self):
        self.persistence.write(self.agent_id, "a", "1")
        self.persistence.write(self.agent_id, "b", "2")
        keys = self.persistence.list_keys(self.agent_id)
        assert sorted(keys) == ["a", "b"]

    def test_isolation_between_agents(self):
        other_agent = uuid4()
        self.persistence.write(self.agent_id, "secret", "mine")
        self.persistence.write(other_agent, "secret", "yours")

        assert self.persistence.read(self.agent_id, "secret") == "mine"
        assert self.persistence.read(other_agent, "secret") == "yours"
