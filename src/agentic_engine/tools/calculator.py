import ast
import operator
import math
from typing import Dict, Any

class SafeCalculatorTool:
    """
    AST-based secure mathematical evaluator for agent arithmetic without eval() security vulnerabilities.
    """
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.Mod: operator.mod
    }

    FUNCTIONS = {
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "sin": math.sin,
        "cos": math.cos,
        "floor": math.floor,
        "ceil": math.ceil
    }

    def evaluate(self, expression: str) -> Dict[str, Any]:
        clean_expr = expression.replace(" ", "")
        try:
            tree = ast.parse(clean_expr, mode="eval")
            result = self._eval_node(tree.body)
            return {"success": True, "expression": expression, "result": float(result)}
        except Exception as e:
            return {"success": False, "error": f"Evaluation error: {str(e)}"}

    def _eval_node(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type in self.OPERATORS:
                return self.OPERATORS[op_type](self._eval_node(node.left), self._eval_node(node.right))
            raise ValueError(f"Unsupported binary operator: {op_type}")
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type in self.OPERATORS:
                return self.OPERATORS[op_type](self._eval_node(node.operand))
            raise ValueError(f"Unsupported unary operator: {op_type}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in self.FUNCTIONS:
                args = [self._eval_node(arg) for arg in node.args]
                return self.FUNCTIONS[func_name](*args)
            raise ValueError(f"Unsupported function: {func_name}")
        raise TypeError(f"Unsupported AST node: {type(node)}")
