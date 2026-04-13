"""AST-based safe expression evaluator.

Replaces all ``eval()`` calls across quartermaster with a sandboxed
evaluator that walks the AST tree directly.  Never calls Python's
built-in ``eval()`` or ``exec()``.

Supports
~~~~~~~~
* Literals: ``int``, ``float``, ``str``, ``bool``, ``None``, ``list``,
  ``dict``, ``tuple``, ``set``, ``f-strings``
* Arithmetic: ``+  -  *  /  //  %  **``
* Comparisons: ``==  !=  <  <=  >  >=  in  not in  is  is not``
* Boolean: ``and  or  not``
* Bitwise: ``&  |  ^  ~  <<  >>``
* Variables from a context dict (``score > 0.5``)
* Subscript: ``row['name']``, ``items[0]``
* Attribute access (whitelisted): ``name.upper()``, ``items.append``
* Whitelisted function calls: ``len``, ``str``, ``int``, ``float``,
  ``bool``, ``abs``, ``round``, ``min``, ``max``, ``sorted``,
  ``list``, ``dict``, ``tuple``, ``set``, ``enumerate``, ``zip``,
  ``range``, ``isinstance``, ``hasattr``, ``any``, ``all``, ``sum``,
  ``map``, ``filter``
* Ternary: ``x if cond else y``
* Comprehensions: ``[x for x in items]``, ``{k: v for k, v in ...}``

Blocks
~~~~~~
* ``import``, ``__import__``, ``exec``, ``eval``, ``compile``,
  ``open``, ``getattr``, ``setattr``, ``delattr``, ``globals``,
  ``locals``, ``vars``, ``dir``, ``type``, ``breakpoint``, ``exit``,
  ``quit``, ``input``, ``print``, ``__builtins__``
* Any dunder attribute access (``__class__``, ``__subclasses__``, etc.)
* ``ast.Call`` on non-whitelisted functions
* Expressions longer than 10 000 chars or deeper than 50 AST levels
"""

from __future__ import annotations

import ast
import operator
from typing import Any

# ── Limits ───────────────────────────────────────────────────────────

MAX_EXPRESSION_LENGTH = 10_000
MAX_AST_DEPTH = 50
MAX_ITERATIONS = 1_000  # for comprehensions / generators
MAX_RESULT_SIZE = 10_000  # max elements in list/dict result

# ── Operators ────────────────────────────────────────────────────────

_BINARY_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
}

_UNARY_OPS: dict[type, Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
    ast.Invert: operator.invert,
}

_CMP_OPS: dict[type, Any] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}

# ── Whitelisted callables ────────────────────────────────────────────

_SAFE_FUNCTIONS: dict[str, Any] = {
    # type conversions
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    # math / aggregation
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "sorted": sorted,
    "reversed": reversed,
    # iteration helpers
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    # predicates
    "any": any,
    "all": all,
    "isinstance": isinstance,
    "hasattr": hasattr,
    # constants
    "True": True,
    "False": False,
    "None": None,
}

# Methods that are safe to call on common types.
_SAFE_METHODS: frozenset[str] = frozenset({
    # str
    "upper", "lower", "strip", "lstrip", "rstrip", "title", "capitalize",
    "startswith", "endswith", "replace", "split", "rsplit", "join",
    "find", "rfind", "index", "rindex", "count", "isdigit", "isalpha",
    "isalnum", "isspace", "isupper", "islower", "format", "encode",
    "zfill", "center", "ljust", "rjust", "removeprefix", "removesuffix",
    # list / tuple
    "append", "extend", "insert", "pop", "remove", "clear", "copy",
    "sort", "reverse", "index", "count",
    # dict
    "keys", "values", "items", "get", "pop", "update", "setdefault",
    "copy", "clear",
    # set
    "add", "discard", "remove", "union", "intersection", "difference",
    "symmetric_difference", "issubset", "issuperset", "isdisjoint", "copy",
})

_BLOCKED_NAMES: frozenset[str] = frozenset({
    "__import__", "eval", "exec", "compile", "open",
    "getattr", "setattr", "delattr",
    "globals", "locals", "vars", "dir", "type",
    "breakpoint", "exit", "quit", "input", "print",
    "__builtins__", "__build_class__", "__loader__",
    "memoryview", "bytearray", "classmethod", "staticmethod",
    "property", "super", "object",
})


# ── AST depth check ─────────────────────────────────────────────────

def _ast_depth(node: ast.AST, current: int = 0) -> int:
    best = current
    for child in ast.iter_child_nodes(node):
        d = _ast_depth(child, current + 1)
        if d > best:
            best = d
    return best


# ── Evaluator ────────────────────────────────────────────────────────

class SafeEvalError(Exception):
    """Raised when a safe-eval expression is invalid or forbidden."""


class _Evaluator:
    """Walks an AST tree and evaluates it against a context dict."""

    def __init__(self, context: dict[str, Any]) -> None:
        self._ctx = context
        self._iterations = 0

    def _tick(self) -> None:
        """Guard against infinite loops in comprehensions."""
        self._iterations += 1
        if self._iterations > MAX_ITERATIONS:
            raise SafeEvalError(
                f"Expression exceeded {MAX_ITERATIONS} iterations"
            )

    # ── dispatch ─────────────────────────────────────────────────────

    def visit(self, node: ast.AST) -> Any:  # noqa: C901 — intentionally large switch
        if isinstance(node, ast.Expression):
            return self.visit(node.body)

        # --- Literals ---
        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.List):
            return [self.visit(e) for e in node.elts]

        if isinstance(node, ast.Tuple):
            return tuple(self.visit(e) for e in node.elts)

        if isinstance(node, ast.Set):
            return {self.visit(e) for e in node.elts}

        if isinstance(node, ast.Dict):
            return {
                self.visit(k) if k is not None else None: self.visit(v)
                for k, v in zip(node.keys, node.values)
            }

        # --- Names (variable lookup) ---
        if isinstance(node, ast.Name):
            name = node.id
            # Context variables take priority — users may have vars named
            # "input", "type", etc. that shadow blocked builtins.
            if name in self._ctx:
                return self._ctx[name]
            if name in _BLOCKED_NAMES:
                raise SafeEvalError(f"Access to '{name}' is not allowed")
            if name in _SAFE_FUNCTIONS:
                return _SAFE_FUNCTIONS[name]
            raise SafeEvalError(
                f"Unknown variable '{name}'. "
                f"Available: {', '.join(sorted(self._ctx.keys())[:20])}"
            )

        # --- Arithmetic / bitwise ---
        if isinstance(node, ast.BinOp):
            op_fn = _BINARY_OPS.get(type(node.op))
            if op_fn is None:
                raise SafeEvalError(f"Unsupported operator: {type(node.op).__name__}")
            left = self.visit(node.left)
            right = self.visit(node.right)
            if isinstance(node.op, ast.Pow) and isinstance(right, (int, float)):
                if abs(right) > 10_000:
                    raise SafeEvalError(f"Exponent too large: {right}")
            return op_fn(left, right)

        if isinstance(node, ast.UnaryOp):
            op_fn = _UNARY_OPS.get(type(node.op))
            if op_fn is None:
                raise SafeEvalError(f"Unsupported unary operator: {type(node.op).__name__}")
            return op_fn(self.visit(node.operand))

        # --- Comparisons ---
        if isinstance(node, ast.Compare):
            left = self.visit(node.left)
            for op, comp in zip(node.ops, node.comparators):
                op_fn = _CMP_OPS.get(type(op))
                if op_fn is None:
                    raise SafeEvalError(f"Unsupported comparison: {type(op).__name__}")
                right = self.visit(comp)
                if not op_fn(left, right):
                    return False
                left = right
            return True

        # --- Boolean ---
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                result: Any = True
                for val in node.values:
                    result = self.visit(val)
                    if not result:
                        return result
                return result
            if isinstance(node.op, ast.Or):
                result = False
                for val in node.values:
                    result = self.visit(val)
                    if result:
                        return result
                return result
            raise SafeEvalError(f"Unsupported bool op: {type(node.op).__name__}")

        # --- Ternary: x if cond else y ---
        if isinstance(node, ast.IfExp):
            return self.visit(node.body) if self.visit(node.test) else self.visit(node.orelse)

        # --- Subscript: obj[key] ---
        if isinstance(node, ast.Subscript):
            obj = self.visit(node.value)
            slc = node.slice
            if isinstance(slc, ast.Slice):
                lower = self.visit(slc.lower) if slc.lower else None
                upper = self.visit(slc.upper) if slc.upper else None
                step = self.visit(slc.step) if slc.step else None
                return obj[lower:upper:step]
            key = self.visit(slc)
            return obj[key]

        # --- Attribute access (whitelisted) ---
        if isinstance(node, ast.Attribute):
            attr = node.attr
            if attr.startswith("_"):
                raise SafeEvalError(
                    f"Access to private/dunder attribute '{attr}' is not allowed"
                )
            obj = self.visit(node.value)
            if not hasattr(obj, attr):
                raise SafeEvalError(
                    f"Object {type(obj).__name__!r} has no attribute '{attr}'"
                )
            val = getattr(obj, attr)
            # If it's a callable method, only allow whitelisted ones
            if callable(val) and attr not in _SAFE_METHODS:
                raise SafeEvalError(
                    f"Method '{attr}' on {type(obj).__name__!r} is not allowed"
                )
            return val

        # --- Function / method calls ---
        if isinstance(node, ast.Call):
            func = self.visit(node.func)
            if not callable(func):
                raise SafeEvalError(f"'{func!r}' is not callable")
            args = [self.visit(a) for a in node.args]
            kwargs = {kw.arg: self.visit(kw.value) for kw in node.keywords if kw.arg}
            # Block **kwargs (double-star)
            if any(kw.arg is None for kw in node.keywords):
                raise SafeEvalError("**kwargs unpacking is not allowed")
            return func(*args, **kwargs)

        # --- Starred (for *args in function calls) ---
        if isinstance(node, ast.Starred):
            return self.visit(node.value)

        # --- Formatted string (f-string) ---
        if isinstance(node, ast.JoinedStr):
            parts: list[str] = []
            for val in node.values:
                if isinstance(val, ast.Constant):
                    parts.append(str(val.value))
                elif isinstance(val, ast.FormattedValue):
                    parts.append(str(self.visit(val.value)))
                else:
                    parts.append(str(self.visit(val)))
            return "".join(parts)

        if isinstance(node, ast.FormattedValue):
            return self.visit(node.value)

        # --- List / dict / set comprehensions ---
        if isinstance(node, ast.ListComp):
            return self._eval_comprehension(node.elt, node.generators, list)

        if isinstance(node, ast.SetComp):
            return self._eval_comprehension(node.elt, node.generators, set)

        if isinstance(node, ast.DictComp):
            return self._eval_dict_comprehension(
                node.key, node.value, node.generators
            )

        if isinstance(node, ast.GeneratorExp):
            # Materialise generator into list (safe within limits)
            return self._eval_comprehension(node.elt, node.generators, list)

        raise SafeEvalError(f"Unsupported expression: {type(node).__name__}")

    # ── comprehension helpers ────────────────────────────────────────

    def _eval_comprehension(
        self,
        elt: ast.AST,
        generators: list[ast.comprehension],
        factory: type,
    ) -> Any:
        results: list[Any] = []
        self._eval_comp_loop(elt, generators, 0, results)
        if factory is set:
            return set(results)
        return results

    def _eval_comp_loop(
        self,
        elt: ast.AST,
        generators: list[ast.comprehension],
        idx: int,
        results: list[Any],
    ) -> None:
        if idx >= len(generators):
            self._tick()
            if len(results) >= MAX_RESULT_SIZE:
                raise SafeEvalError(
                    f"Comprehension produced more than {MAX_RESULT_SIZE} elements"
                )
            results.append(self.visit(elt))
            return

        gen = generators[idx]
        iterable = self.visit(gen.iter)
        target_name = self._extract_target(gen.target)

        for item in iterable:
            self._tick()
            self._assign_target(gen.target, item, target_name)
            # Check all if-conditions
            if all(self.visit(cond) for cond in gen.ifs):
                self._eval_comp_loop(elt, generators, idx + 1, results)

    def _eval_dict_comprehension(
        self,
        key_node: ast.AST,
        value_node: ast.AST,
        generators: list[ast.comprehension],
    ) -> dict:
        pairs: list[tuple[Any, Any]] = []

        def collect(idx: int) -> None:
            if idx >= len(generators):
                self._tick()
                if len(pairs) >= MAX_RESULT_SIZE:
                    raise SafeEvalError(
                        f"Dict comprehension produced more than {MAX_RESULT_SIZE} entries"
                    )
                pairs.append((self.visit(key_node), self.visit(value_node)))
                return
            gen = generators[idx]
            iterable = self.visit(gen.iter)
            target_name = self._extract_target(gen.target)
            for item in iterable:
                self._tick()
                self._assign_target(gen.target, item, target_name)
                if all(self.visit(cond) for cond in gen.ifs):
                    collect(idx + 1)

        collect(0)
        return dict(pairs)

    def _extract_target(self, target: ast.AST) -> str | None:
        if isinstance(target, ast.Name):
            return target.id
        return None

    def _assign_target(self, target: ast.AST, value: Any, name: str | None) -> None:
        if isinstance(target, ast.Name):
            self._ctx[target.id] = value
        elif isinstance(target, ast.Tuple) or isinstance(target, ast.List):
            for t, v in zip(target.elts, value):
                if isinstance(t, ast.Name):
                    self._ctx[t.id] = v
        else:
            raise SafeEvalError(f"Unsupported assignment target: {type(target).__name__}")


# ── Public API ───────────────────────────────────────────────────────

def safe_eval(expression: str, context: dict[str, Any] | None = None) -> Any:
    """Safely evaluate a Python expression against a context dict.

    Uses AST parsing — **never** calls Python's ``eval()`` or ``exec()``.

    Args:
        expression: A Python expression string.
        context: Variables available to the expression.  A shallow copy
            is made so comprehension variables don't leak.

    Returns:
        The result of evaluating the expression.

    Raises:
        SafeEvalError: If the expression uses forbidden constructs,
            references undefined variables, or exceeds safety limits.

    Examples::

        >>> safe_eval("score > 0.5", {"score": 0.8})
        True
        >>> safe_eval("name.upper()", {"name": "alice"})
        'ALICE'
        >>> safe_eval("[x * 2 for x in items]", {"items": [1, 2, 3]})
        [2, 4, 6]
        >>> safe_eval("row['age'] >= 18", {"row": {"age": 21}})
        True
    """
    if not expression or not expression.strip():
        raise SafeEvalError("Expression is empty")

    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise SafeEvalError(
            f"Expression too long ({len(expression)} chars, max {MAX_EXPRESSION_LENGTH})"
        )

    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise SafeEvalError(f"Invalid expression syntax: {exc}") from exc

    depth = _ast_depth(tree)
    if depth > MAX_AST_DEPTH:
        raise SafeEvalError(
            f"Expression too deeply nested (depth {depth}, max {MAX_AST_DEPTH})"
        )

    ctx = dict(context) if context else {}
    evaluator = _Evaluator(ctx)

    try:
        return evaluator.visit(tree)
    except SafeEvalError:
        raise
    except Exception as exc:
        raise SafeEvalError(f"Evaluation error: {exc}") from exc
