from z3 import *
import sympy
from sympy.parsing.sympy_parser import parse_expr
from sympy.abc import x, y, z, a, b, c  # Common variables
import json

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
    
    # Handle operations
    if expr.is_Add:
        return sum(sympy_to_z3(arg, variables) for arg in expr.args)
    
    if expr.is_Mul:
        result = sympy_to_z3(expr.args[0], variables)
        for arg in expr.args[1:]:
            result *= sympy_to_z3(arg, variables)
        return result
    
    if expr.is_Pow:
        base = sympy_to_z3(expr.base, variables)
        # Handle negative and fractional exponents
        if expr.exp.is_number:
            if expr.exp < 0:
                return 1 / (base ** float(abs(expr.exp)))
            return base ** float(expr.exp)
    
    # Handle division
    if isinstance(expr, sympy.core.mul.Mul) and any(arg.is_Pow and arg.exp.is_negative for arg in expr.args):
        num = sympy.core.mul.Mul(*[arg for arg in expr.args if not (arg.is_Pow and arg.exp.is_negative)])
        den = sympy.core.mul.Mul(*[arg.base ** abs(arg.exp) for arg in expr.args if arg.is_Pow and arg.exp.is_negative])
        return sympy_to_z3(num, variables) / sympy_to_z3(den, variables)
    
    raise ValueError(f"Unsupported expression type: {type(expr)}")

def convert_to_z3(expr_str):
    """Convert a string mathematical expression to Z3 format using SymPy for parsing."""
    try:
        # Replace {-n} style exponents with regular negative exponents
        expr_str = expr_str.replace('^{-', '^(-')
        # Parse the expression using SymPy
        sympy_expr = parse_expr(expr_str.replace('^', '**'))
        # Convert to Z3 expression
        return sympy_to_z3(sympy_expr)
    except Exception as e:
        raise ValueError(f"Error converting expression '{expr_str}': {str(e)}")

def check_equivalence(equation1, equation2):
    """
    Check if two mathematical expressions are equivalent using Z3.
    Returns True if the expressions are equivalent, False otherwise.
    """
    try:
        # Convert both expressions to Z3 format
        expr1 = convert_to_z3(equation1)
        expr2 = convert_to_z3(equation2)
        
        # Create Z3 solver
        s = Solver()
        
        # Add constraint that expressions are not equal
        s.add(expr1 != expr2)
        
        # Check if there exists a solution where expressions are different
        result = s.check()
        if result == unsat:
            return True  # Expressions are equivalent
        elif result == sat:
            return False
        else:
            return False  # unknown
        
    except Exception as e:
        print(f"Error checking equivalence: {str(e)}")
        return False

def test_json_file(filepath):
    """Test equivalence for all examples in a JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        results = []
        for item in data:
            decision1 = item.get('decision1', item.get('decision', ''))
            decision2 = item.get('decision2', '')
            examples = item.get('examples', [])
            
            print(f"\nTesting examples for decisions: {decision1} & {decision2}")
            print("-" * 60)
            
            for idx, example in enumerate(examples, 1):
                before = example.get('before', '')
                after = example.get('after', '')
                
                if before and after:
                    is_equivalent = check_equivalence(before, after)
                    results.append({
                        'decisions': [decision1, decision2],
                        'before': before,
                        'after': after,
                        'equivalent': is_equivalent
                    })
                    
                    print(f"Example {idx}:")
                    print(f"Before: {before}")
                    print(f"After:  {after}")
                    print(f"Equivalent: {is_equivalent}")
                    print()
        
        # Print summary
        total = len(results)
        equivalent = sum(1 for r in results if r['equivalent'])
        print("\nSummary:")
        print(f"Total examples tested: {total}")
        print(f"Equivalent: {equivalent}")
        print(f"Not equivalent: {total - equivalent}")
        print(f"Success rate: {(equivalent/total)*100:.2f}%")
        
        return results
        
    except Exception as e:
        print(f"Error processing JSON file: {str(e)}")
        return []

# Example usage and tests
if __name__ == "__main__":
    # First run the manual test cases
    print("Running manual test cases...")
    test_cases = [
        # Basic arithmetic
        ("x + y", "y + x"),
        ("2*x + 3*x", "5*x"),
        
        # Powers and negative exponents
        ("x^2 * x^3", "x^5"),
        ("x^{-3}", "1/x^3"),
        
        # Complex fractions
        ("1/(1/x + 1/y)", "(x*y)/(x + y)"),
        
        # Nested expressions
        ("(x + 1)^2/(x + 1)", "x + 1"),
        
        # From your dataset
        ("((1/x)/(1/y))^2", "(y/x)^2"),
        ("(x^2)^3 * (y^3)^2", "x^6 * y^6"),
        ("1/(1/x + 1/y + 1/z)", "(x*y*z)/(x*y + y*z + x*z)")
    ]
    
    for eq1, eq2 in test_cases:
        result = check_equivalence(eq1, eq2)
        print(f"\nTesting: {eq1} ≟ {eq2}")
        print(f"Result: {'Equivalent' if result else 'Not equivalent'}")
    
    # Then test the JSON files
    print("\n\nTesting JSON examples...")
    json_files = [
        'data/dataset.json',
        'data/complex_dataset.json'
    ]
    
    for json_file in json_files:
        print(f"\nProcessing {json_file}...")
        results = test_json_file(json_file)