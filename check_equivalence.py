from z3 import *
import sympy
from sympy.parsing.sympy_parser import parse_expr
from sympy.abc import x, y, z, a, b, c  # Common variables
import json
import random
import numpy as np
from typing import Dict, List, Tuple, Union

class ExpressionEvaluator:
    def __init__(self, num_random_tests: int = 100):
        self.num_random_tests = num_random_tests
    
    def evaluate_random(self, expr_str: str, variables: List[str], num_points: int = None) -> List[float]:
        """Evaluate an expression at random points."""
        if num_points is None:
            num_points = self.num_random_tests
            
        try:
            # Replace {-n} style exponents with regular negative exponents
            expr_str = expr_str.replace('^{-', '^(-')
            # Parse the expression using SymPy
            expr = parse_expr(expr_str.replace('^', '**'))
            
            results = []
            for _ in range(num_points):
                # Create random values for variables
                var_values = {var: random.uniform(0.1, 10.0) for var in variables}
                result = float(expr.evalf(subs=var_values))
                results.append(result)
                
            return results
        except Exception as e:
            raise ValueError(f"Error evaluating expression '{expr_str}': {str(e)}")
    
    def check_equivalence_random(self, equation1: str, equation2: str, 
                               tolerance: float = 1e-10) -> Tuple[bool, Dict]:
        """
        Check if two expressions are equivalent using random sampling.
        Returns (is_equivalent, details).
        """
        try:
            # Extract variables from both expressions
            expr1 = parse_expr(equation1.replace('^', '**'))
            expr2 = parse_expr(equation2.replace('^', '**'))
            variables = set(map(str, expr1.free_symbols | expr2.free_symbols))
            
            # Evaluate both expressions at random points
            results1 = self.evaluate_random(equation1, variables)
            results2 = self.evaluate_random(equation2, variables)
            
            # Compare results
            differences = [abs(r1 - r2) for r1, r2 in zip(results1, results2)]
            max_diff = max(differences)
            avg_diff = sum(differences) / len(differences)
            
            is_equivalent = max_diff <= tolerance
            
            details = {
                'max_difference': max_diff,
                'avg_difference': avg_diff,
                'num_points': self.num_random_tests,
                'tolerance': tolerance,
                'variables': list(variables)
            }
            
            return is_equivalent, details
            
        except Exception as e:
            raise ValueError(f"Error checking equivalence: {str(e)}")

def sympy_to_z3(expr, variables=None):
    """Convert a SymPy expression to a Z3 expression."""
    if variables is None:
        variables = {}
    
    if expr.is_number:
        return float(expr)
    
    if expr.is_Symbol:
        var_name = str(expr)
        if var_name not in variables:
            variables[var_name] = Real(var_name)
        return variables[var_name]
    
    if expr.is_Add:
        return sum(sympy_to_z3(arg, variables) for arg in expr.args)
    
    if expr.is_Mul:
        result = sympy_to_z3(expr.args[0], variables)
        for arg in expr.args[1:]:
            result *= sympy_to_z3(arg, variables)
        return result
    
    if expr.is_Pow:
        base = sympy_to_z3(expr.base, variables)
        if expr.exp.is_number:
            if expr.exp < 0:
                return 1 / (base ** float(abs(expr.exp)))
            return base ** float(expr.exp)
    
    if isinstance(expr, sympy.core.mul.Mul) and any(arg.is_Pow and arg.exp.is_negative for arg in expr.args):
        num = sympy.core.mul.Mul(*[arg for arg in expr.args if not (arg.is_Pow and arg.exp.is_negative)])
        den = sympy.core.mul.Mul(*[arg.base ** abs(arg.exp) for arg in expr.args if arg.is_Pow and arg.exp.is_negative])
        return sympy_to_z3(num, variables) / sympy_to_z3(den, variables)
    
    raise ValueError(f"Unsupported expression type: {type(expr)}")

def check_equivalence_z3(equation1: str, equation2: str) -> Tuple[bool, Dict]:
    """
    Check if two expressions are equivalent using Z3.
    Returns (is_equivalent, details).
    """
    try:
        # Replace {-n} style exponents with regular negative exponents
        equation1 = equation1.replace('^{-', '^(-')
        equation2 = equation2.replace('^{-', '^(-')
        
        # Parse the expressions using SymPy
        expr1 = parse_expr(equation1.replace('^', '**'))
        expr2 = parse_expr(equation2.replace('^', '**'))
        
        # Convert to Z3 expressions
        z3_expr1 = sympy_to_z3(expr1)
        z3_expr2 = sympy_to_z3(expr2)
        
        # Create Z3 solver
        s = Solver()
        s.add(z3_expr1 != z3_expr2)
        
        # Check if there exists a solution where expressions are different
        result = s.check()
        is_equivalent = result == unsat
        
        details = {
            'solver_result': str(result),
            'variables': list(map(str, expr1.free_symbols | expr2.free_symbols))
        }
        
        if result == sat:
            model = s.model()
            details['counterexample'] = {str(d): str(model[d]) for d in model.decls()}
        
        return is_equivalent, details
        
    except Exception as e:
        raise ValueError(f"Error checking equivalence: {str(e)}")

def test_expressions(json_file: str = None, test_cases: List[Tuple[str, str]] = None):
    """Test expressions using both Z3 and random sampling."""
    evaluator = ExpressionEvaluator(num_random_tests=1000)
    
    def run_tests(expr1: str, expr2: str, context: str = ""):
        print(f"\n{context}")
        print(f"Testing: {expr1} ≟ {expr2}")
        
        # Test with Z3
        try:
            z3_equiv, z3_details = check_equivalence_z3(expr1, expr2)
            print(f"Z3 Result: {'Equivalent' if z3_equiv else 'Not equivalent'}")
            if not z3_equiv and 'counterexample' in z3_details:
                print(f"Z3 Counterexample: {z3_details['counterexample']}")
        except Exception as e:
            print(f"Z3 Error: {str(e)}")
            z3_equiv = None
        
        # Test with random sampling
        try:
            random_equiv, random_details = evaluator.check_equivalence_random(expr1, expr2)
            print(f"Random Sampling Result: {'Equivalent' if random_equiv else 'Not equivalent'}")
            print(f"Max difference: {random_details['max_difference']:.2e}")
            print(f"Avg difference: {random_details['avg_difference']:.2e}")
        except Exception as e:
            print(f"Random Sampling Error: {str(e)}")
            random_equiv = None
        
        # Compare results
        if z3_equiv is not None and random_equiv is not None:
            if z3_equiv != random_equiv:
                print("Warning: Z3 and random sampling disagree!")
        
        return z3_equiv, random_equiv

    # Test manual test cases
    if test_cases:
        print("\nTesting manual test cases...")
        for eq1, eq2 in test_cases:
            run_tests(eq1, eq2, "Manual test case")
    
    # Test JSON file
    if json_file:
        print(f"\nTesting examples from {json_file}...")
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            for item in data:
                decision1 = item.get('decision1', item.get('decision', ''))
                decision2 = item.get('decision2', '')
                examples = item.get('examples', [])
                
                print(f"\nTesting examples for decisions: {decision1} & {decision2}")
                print("-" * 60)
                
                for example in examples:
                    before = example.get('before', '')
                    after = example.get('after', '')
                    if before and after:
                        run_tests(before, after, "JSON test case")
                        
        except Exception as e:
            print(f"Error processing JSON file: {str(e)}")

if __name__ == "__main__":
    # Define some test cases
    test_cases = [
        ("x + y", "y + x"),
        ("2*x + 3*x", "5*x"),
        ("x^2 * x^3", "x^5"),
        ("x^{-3}", "1/x^3"),
        ("1/(1/x + 1/y)", "(x*y)/(x + y)"),
        ("(x + 1)^2/(x + 1)", "x + 1"),
        ("((1/x)/(1/y))^2", "(y/x)^2"),
        ("(x^2)^3 * (y^3)^2", "x^6 * y^6"),
    ]
    
    # Test both manual cases and JSON files
    test_expressions(
        json_file='dataset.json',
        test_cases=test_cases
    )