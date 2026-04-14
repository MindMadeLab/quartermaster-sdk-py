"""Safe expression evaluator backed by ``simpleeval``.

Provides a sandboxed ``safe_eval()`` function that never calls Python's
built-in ``eval()`` or ``exec()``.  Uses the ``simpleeval`` library
(``EvalWithCompoundTypes``) under the hood so that list/dict/set
comprehensions, subscripts, and ternary expressions all work out of the
box.

Supports
~~~~~~~~
* Literals: ``int``, ``float``, ``str``, ``bool``, ``None``, ``list``,
  ``dict``, ``tuple``, ``set``
* Arithmetic: ``+  -  *  /  //  %  **``
* Comparisons: ``==  !=  <  <=  >  >=  in  not in  is  is not``
* Boolean: ``and  or  not``
* Bitwise: ``&  |  ^  ~  <<  >>``
* Variables from a context dict (``score > 0.5``)
* Subscript: ``row['name']``, ``items[0]``
* Attribute access (safe subset — dunders blocked by simpleeval)
* Whitelisted function calls: ``len``, ``str``, ``int``, ``float``,
  ``bool``, ``abs``, ``round``, ``min``, ``max``, ``sorted``,
  ``list``, ``dict``, ``tuple``, ``set``, ``enumerate``, ``zip``,
  ``range``, ``isinstance``, ``any``, ``all``, ``sum``
* Ternary: ``x if cond else y``
* Comprehensions: ``[x for x in items]``, ``{k: v for k, v in ...}``

Blocks
~~~~~~
* ``import``, ``__import__``, ``exec``, ``eval``, ``compile``,
  ``open``, ``getattr``, ``setattr``, ``delattr``, ``globals``,
  ``locals``, ``vars``, ``dir``, ``type``, ``breakpoint``, ``exit``,
  ``quit``, ``input``, ``print``, ``__builtins__``
* Any dunder attribute access (``__class__``, ``__subclasses__``, etc.)
* Expressions longer than 10 000 chars
"""

from __future__ import annotations

import ast
from typing import Any

import simpleeval
from simpleeval import (
    EvalWithCompoundTypes,
    FeatureNotAvailable,
    IterableTooLong,
    MAX_COMPREHENSION_LENGTH,
    NameNotDefined,
    NumberTooHigh,
)

# ── Limits ───────────────────────────────────────────────────────────

MAX_EXPRESSION_LENGTH = 10_000

# Cap the exponent to a safe maximum (simpleeval's default is 4_000_000).
simpleeval.MAX_POWER = 10_000


# ── Exception ────────────────────────────────────────────────────────


class SafeEvalError(Exception):
    """Raised when a safe-eval expression is invalid or forbidden."""


# ── Extended evaluator ───────────────────────────────────────────────


class _SafeEvaluator(EvalWithCompoundTypes):
    """Extends ``EvalWithCompoundTypes`` with set-comprehension support."""

    def __init__(self, names: dict[str, Any] | None = None) -> None:
        super().__init__(names=names or {})
        # Register the SetComp handler (not included upstream).
        self.nodes[ast.SetComp] = self._eval_set_comprehension

    def _eval_set_comprehension(self, node: ast.SetComp) -> set:
        """Evaluate ``{expr for x in iterable [if cond]}``."""
        to_return: list[Any] = []
        extra_names: dict[str, Any] = {}

        previous_name_evaller = self.nodes[ast.Name]

        def eval_names_extra(n: ast.Name) -> Any:
            if n.id in extra_names:
                return extra_names[n.id]
            return previous_name_evaller(n)

        self.nodes[ast.Name] = eval_names_extra

        def recurse_targets(target: ast.AST, value: Any) -> None:
            if isinstance(target, ast.Name):
                extra_names[target.id] = value
            else:
                for t, v in zip(target.elts, value):  # type: ignore[attr-defined]
                    recurse_targets(t, v)

        def do_generator(gi: int = 0) -> None:
            g = node.generators[gi]
            for i in self._eval(g.iter):
                self._max_count += 1
                if self._max_count > MAX_COMPREHENSION_LENGTH:
                    raise IterableTooLong("Comprehension generates too many elements")
                recurse_targets(g.target, i)
                if all(self._eval(iff) for iff in g.ifs):
                    if len(node.generators) > gi + 1:
                        do_generator(gi + 1)
                    else:
                        to_return.append(self._eval(node.elt))

        try:
            do_generator()
        finally:
            self.nodes[ast.Name] = previous_name_evaller

        return set(to_return)


# ── Public API ───────────────────────────────────────────────────────


def safe_eval(expression: str, context: dict[str, Any] | None = None) -> Any:
    """Safely evaluate a Python expression against a context dict.

    Uses ``simpleeval.EvalWithCompoundTypes`` — **never** calls Python's
    ``eval()`` or ``exec()``.

    Args:
        expression: A Python expression string.
        context: Variables available to the expression.

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

    evaluator = _SafeEvaluator(names=context or {})

    # Add safe built-in functions
    evaluator.functions.update(
        {
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "len": len,
            "sorted": sorted,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "any": any,
            "all": all,
            "isinstance": isinstance,
        }
    )

    # Allow isinstance() — simpleeval blocks it by default via
    # DISALLOW_FUNCTIONS.  We remove it from the module-level set so
    # the _eval_call check passes.  This is safe because the function
    # itself is harmless (read-only type check).
    simpleeval.DISALLOW_FUNCTIONS.discard(isinstance)

    try:
        return evaluator.eval(expression)
    except (
        FeatureNotAvailable,
        NameNotDefined,
        NumberTooHigh,
        IterableTooLong,
        TypeError,
        ValueError,
        AttributeError,
        KeyError,
        IndexError,
        ZeroDivisionError,
    ) as e:
        raise SafeEvalError(str(e)) from e
    except Exception as e:
        raise SafeEvalError(f"Evaluation failed: {e}") from e
