"""Safe AST-based numeric calculator.

Allows: +, -, *, /, //, %, **, unary +/-, parentheses, int/float literals.
Rejects: any name, attribute, call, comprehension, subscript, or other node.
"""
from __future__ import annotations

import ast
import operator as op


_BIN_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}

_UNARY_OPS = {
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def calculate(expr: str) -> float:
    tree = ast.parse(expr, mode="eval")
    return float(_eval(tree.body))


def _eval(node: ast.AST):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError(f"Disallowed constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise ValueError(f"Disallowed operator: {op_type.__name__}")
        return _BIN_OPS[op_type](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ValueError(f"Disallowed unary operator: {op_type.__name__}")
        return _UNARY_OPS[op_type](_eval(node.operand))
    raise ValueError(f"Disallowed expression node: {type(node).__name__}")


if __name__ == "__main__":
    cases = [
        ("2023年毛利率近似", "(1500.5 - 800.2) / 1500.5 * 100"),
        ("同比增长", "(120 - 100) / 100"),
        ("复利", "(1 + 0.08) ** 5"),
    ]
    for label, expr in cases:
        print(f"{label}: {expr} = {calculate(expr)}")
