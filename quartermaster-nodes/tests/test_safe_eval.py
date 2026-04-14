"""Tests for the simpleeval-backed safe expression evaluator."""

from __future__ import annotations

import pytest

from quartermaster_nodes.safe_eval import SafeEvalError, safe_eval


# ── Literals ────────────────────────────────────────────────────────────


class TestLiterals:
    def test_int(self) -> None:
        assert safe_eval("42") == 42

    def test_float(self) -> None:
        assert safe_eval("3.14") == pytest.approx(3.14)

    def test_string(self) -> None:
        assert safe_eval("'hello'") == "hello"

    def test_bool_true(self) -> None:
        assert safe_eval("True") is True

    def test_bool_false(self) -> None:
        assert safe_eval("False") is False

    def test_none(self) -> None:
        assert safe_eval("None") is None

    def test_list(self) -> None:
        assert safe_eval("[1, 2, 3]") == [1, 2, 3]

    def test_dict(self) -> None:
        assert safe_eval("{'a': 1, 'b': 2}") == {"a": 1, "b": 2}

    def test_tuple(self) -> None:
        assert safe_eval("(1, 2)") == (1, 2)

    def test_set(self) -> None:
        assert safe_eval("{1, 2, 3}") == {1, 2, 3}


# ── Variables ───────────────────────────────────────────────────────────


class TestVariables:
    def test_simple_variable(self) -> None:
        assert safe_eval("x", {"x": 10}) == 10

    def test_variable_in_expression(self) -> None:
        assert safe_eval("x + y", {"x": 3, "y": 4}) == 7

    def test_undefined_variable_raises(self) -> None:
        with pytest.raises(SafeEvalError):
            safe_eval("z", {"x": 1})


# ── Arithmetic ──────────────────────────────────────────────────────────


class TestArithmetic:
    def test_add(self) -> None:
        assert safe_eval("2 + 3") == 5

    def test_subtract(self) -> None:
        assert safe_eval("10 - 4") == 6

    def test_multiply(self) -> None:
        assert safe_eval("3 * 5") == 15

    def test_divide(self) -> None:
        assert safe_eval("10 / 4") == 2.5

    def test_floor_divide(self) -> None:
        assert safe_eval("10 // 3") == 3

    def test_modulo(self) -> None:
        assert safe_eval("10 % 3") == 1

    def test_power(self) -> None:
        assert safe_eval("2 ** 10") == 1024

    def test_power_too_large_raises(self) -> None:
        with pytest.raises(SafeEvalError):
            safe_eval("2 ** 100000")

    def test_unary_neg(self) -> None:
        assert safe_eval("-5") == -5

    def test_unary_pos(self) -> None:
        assert safe_eval("+5") == 5

    def test_bitwise_and(self) -> None:
        assert safe_eval("0b1100 & 0b1010") == 0b1000

    def test_bitwise_or(self) -> None:
        assert safe_eval("0b1100 | 0b1010") == 0b1110


# ── Comparisons ─────────────────────────────────────────────────────────


class TestComparisons:
    def test_eq(self) -> None:
        assert safe_eval("1 == 1") is True

    def test_neq(self) -> None:
        assert safe_eval("1 != 2") is True

    def test_lt(self) -> None:
        assert safe_eval("1 < 2") is True

    def test_lte(self) -> None:
        assert safe_eval("2 <= 2") is True

    def test_gt(self) -> None:
        assert safe_eval("3 > 2") is True

    def test_gte(self) -> None:
        assert safe_eval("3 >= 3") is True

    def test_in(self) -> None:
        assert safe_eval("'a' in items", {"items": ["a", "b"]}) is True

    def test_not_in(self) -> None:
        assert safe_eval("'c' not in items", {"items": ["a", "b"]}) is True

    def test_is_none(self) -> None:
        assert safe_eval("x is None", {"x": None}) is True

    def test_is_not_none(self) -> None:
        assert safe_eval("x is not None", {"x": 5}) is True

    def test_chained_comparison(self) -> None:
        assert safe_eval("1 < 2 < 3") is True
        assert safe_eval("1 < 2 > 3") is False


# ── Boolean operators ───────────────────────────────────────────────────


class TestBooleanOps:
    def test_and_true(self) -> None:
        assert safe_eval("True and True") is True

    def test_and_false(self) -> None:
        assert safe_eval("True and False") is False

    def test_or_true(self) -> None:
        assert safe_eval("False or True") is True

    def test_or_false(self) -> None:
        assert safe_eval("False or False") is False

    def test_not(self) -> None:
        assert safe_eval("not False") is True

    def test_complex_boolean(self) -> None:
        assert safe_eval("x > 0 and y < 10", {"x": 5, "y": 3}) is True

    def test_short_circuit_and(self) -> None:
        # Should not raise even though y is undefined, because x is falsy
        assert safe_eval("x and y", {"x": 0, "y": "unused"}) == 0

    def test_short_circuit_or(self) -> None:
        assert safe_eval("x or y", {"x": 42, "y": "unused"}) == 42


# ── Ternary ─────────────────────────────────────────────────────────────


class TestTernary:
    def test_true_branch(self) -> None:
        assert safe_eval("'yes' if True else 'no'") == "yes"

    def test_false_branch(self) -> None:
        assert safe_eval("'yes' if False else 'no'") == "no"

    def test_with_variable(self) -> None:
        assert safe_eval("x * 2 if x > 0 else 0", {"x": 5}) == 10


# ── Subscript ───────────────────────────────────────────────────────────


class TestSubscript:
    def test_dict_access(self) -> None:
        assert safe_eval("row['name']", {"row": {"name": "Alice"}}) == "Alice"

    def test_list_index(self) -> None:
        assert safe_eval("items[0]", {"items": [10, 20, 30]}) == 10

    def test_negative_index(self) -> None:
        assert safe_eval("items[-1]", {"items": [10, 20, 30]}) == 30

    def test_slice(self) -> None:
        assert safe_eval("items[1:3]", {"items": [10, 20, 30, 40]}) == [20, 30]

    def test_nested_subscript(self) -> None:
        data = {"users": [{"name": "Alice"}, {"name": "Bob"}]}
        assert safe_eval("data['users'][1]['name']", {"data": data}) == "Bob"


# ── Attribute access ────────────────────────────────────────────────────


class TestAttributes:
    def test_str_upper(self) -> None:
        assert safe_eval("name.upper()", {"name": "alice"}) == "ALICE"

    def test_str_startswith(self) -> None:
        assert safe_eval("name.startswith('al')", {"name": "alice"}) is True

    def test_str_split(self) -> None:
        assert safe_eval("text.split(',')", {"text": "a,b,c"}) == ["a", "b", "c"]

    def test_dict_keys(self) -> None:
        result = safe_eval("list(d.keys())", {"d": {"a": 1, "b": 2}})
        assert sorted(result) == ["a", "b"]

    def test_dict_get(self) -> None:
        assert safe_eval("d.get('x', 'default')", {"d": {}}) == "default"

    def test_private_attr_blocked(self) -> None:
        with pytest.raises(SafeEvalError):
            safe_eval("x.__class__", {"x": 1})

    def test_dunder_blocked(self) -> None:
        with pytest.raises(SafeEvalError):
            safe_eval("x.__subclasses__()", {"x": int})


# ── Function calls ──────────────────────────────────────────────────────


class TestFunctions:
    def test_len(self) -> None:
        assert safe_eval("len(items)", {"items": [1, 2, 3]}) == 3

    def test_str_conversion(self) -> None:
        assert safe_eval("str(42)") == "42"

    def test_int_conversion(self) -> None:
        assert safe_eval("int('42')") == 42

    def test_abs(self) -> None:
        assert safe_eval("abs(-5)") == 5

    def test_min_max(self) -> None:
        assert safe_eval("min(1, 2, 3)") == 1
        assert safe_eval("max(1, 2, 3)") == 3

    def test_sorted(self) -> None:
        assert safe_eval("sorted([3, 1, 2])") == [1, 2, 3]

    def test_sum(self) -> None:
        assert safe_eval("sum([1, 2, 3])") == 6

    def test_isinstance(self) -> None:
        assert safe_eval("isinstance(x, int)", {"x": 42}) is True

    def test_any_all(self) -> None:
        assert safe_eval("any([False, True, False])") is True
        assert safe_eval("all([True, True, True])") is True
        assert safe_eval("all([True, False, True])") is False

    def test_range(self) -> None:
        assert safe_eval("list(range(5))") == [0, 1, 2, 3, 4]


# ── Comprehensions ──────────────────────────────────────────────────────


class TestComprehensions:
    def test_list_comp(self) -> None:
        assert safe_eval("[x * 2 for x in items]", {"items": [1, 2, 3]}) == [2, 4, 6]

    def test_list_comp_with_filter(self) -> None:
        assert safe_eval("[x for x in items if x > 2]", {"items": [1, 2, 3, 4]}) == [3, 4]

    def test_set_comp(self) -> None:
        assert safe_eval("{x % 3 for x in items}", {"items": [1, 2, 3, 4, 5]}) == {0, 1, 2}

    def test_dict_comp(self) -> None:
        assert safe_eval(
            "{k: v for k, v in pairs}",
            {"pairs": [("a", 1), ("b", 2)]},
        ) == {"a": 1, "b": 2}

    def test_nested_comp(self) -> None:
        assert safe_eval(
            "[x + y for x in [1, 2] for y in [10, 20]]"
        ) == [11, 21, 12, 22]


# ── Blocked constructs (security) ──────────────────────────────────────


class TestSecurity:
    def test_import_blocked(self) -> None:
        with pytest.raises(SafeEvalError):
            safe_eval("__import__('os')")

    def test_eval_blocked(self) -> None:
        with pytest.raises(SafeEvalError):
            safe_eval("eval('1+1')")

    def test_exec_blocked(self) -> None:
        # exec is a statement, not an expression — should fail to parse
        with pytest.raises(SafeEvalError):
            safe_eval("exec('x=1')")

    def test_open_blocked(self) -> None:
        with pytest.raises(SafeEvalError):
            safe_eval("open('/etc/passwd')")

    def test_globals_blocked(self) -> None:
        with pytest.raises(SafeEvalError):
            safe_eval("globals()")

    def test_builtins_blocked(self) -> None:
        with pytest.raises(SafeEvalError):
            safe_eval("__builtins__")

    def test_dunder_class(self) -> None:
        with pytest.raises(SafeEvalError):
            safe_eval("''.__class__")

    def test_dunder_subclasses(self) -> None:
        with pytest.raises(SafeEvalError):
            safe_eval("''.__class__.__subclasses__()")

    def test_type_blocked(self) -> None:
        with pytest.raises(SafeEvalError):
            safe_eval("type('X', (), {})")

    def test_empty_expression(self) -> None:
        with pytest.raises(SafeEvalError, match="empty"):
            safe_eval("")

    def test_whitespace_only(self) -> None:
        with pytest.raises(SafeEvalError, match="empty"):
            safe_eval("   ")

    def test_syntax_error(self) -> None:
        with pytest.raises(SafeEvalError):
            safe_eval("if True:")

    def test_expression_length_limit(self) -> None:
        with pytest.raises(SafeEvalError, match="too long"):
            safe_eval("x" * 10_001, {"x": 1})


# ── Real-world node expressions ─────────────────────────────────────────


class TestNodeExpressions:
    """Expressions that actual QM nodes would evaluate."""

    def test_if_node_score_threshold(self) -> None:
        assert safe_eval("score > 0.5", {"score": 0.8}) is True
        assert safe_eval("score > 0.5", {"score": 0.3}) is False

    def test_if_node_string_comparison(self) -> None:
        assert safe_eval("sentiment == 'positive'", {"sentiment": "positive"}) is True

    def test_switch_department_match(self) -> None:
        assert safe_eval("department == 'billing'", {"department": "billing"}) is True
        assert safe_eval("department == 'billing'", {"department": "tech"}) is False

    def test_var_node_string_concat(self) -> None:
        assert safe_eval("first + ' ' + last", {"first": "John", "last": "Doe"}) == "John Doe"

    def test_static_decision_length_check(self) -> None:
        assert safe_eval("len(input) > 100", {"input": "short"}) is False
        assert safe_eval("len(input) > 100", {"input": "x" * 200}) is True

    def test_memory_expression(self) -> None:
        assert safe_eval("count + 1", {"count": 5}) == 6

    def test_filter_row_expression(self) -> None:
        row = {"name": "Alice", "age": 25, "active": True}
        assert safe_eval("row['age'] > 18", {"row": row}) is True
        assert safe_eval("row['active'] and row['age'] >= 21", {"row": row}) is True

    def test_complex_expression(self) -> None:
        ctx = {
            "category": "tech",
            "priority": 3,
            "tags": ["urgent", "bug"],
        }
        assert safe_eval(
            "category == 'tech' and priority > 2 and 'urgent' in tags",
            ctx,
        ) is True
