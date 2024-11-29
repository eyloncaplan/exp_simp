'''
Note: There are two equivalence functions: z3 and random sampling. Having both functions helps for stress testing (z3 equivalence function can be used to find bugs in random sampling equivalence function, and vice versa).

Another justification for having both is that one may be faster than the other. We can default to using the faster method to checking if the two expressions are equivalent. If the faster method returns false, then we use the slower method (ideally most of the examples are positive examples, so we don't have to use the slower method often).
'''

from z3 import *
import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from sympy.abc import x, y, z, a, b, c  # Common variables
import json
import random
import numpy as np
from typing import Dict, List, Tuple, Union
import datetime
import re

standard_transformations += (implicit_multiplication_application,)

def process_fractions(expr_str: str) -> str:
    """Remove spaces in fractions."""
    def clean_fraction(match):
        return ''.join(match.group(0).split())
    expr_str = re.sub(r'\d*\s*/\s*[a-zA-Z0-9^()\{\}]+', clean_fraction, expr_str)
    expr_str = re.sub(r'[a-zA-Z0-9^()\{\}]+\s*/\s*[a-zA-Z0-9^()\{\}]+', clean_fraction, expr_str)
    return expr_str

def process_multiplication(expr_str: str) -> str:
    """Make implicit multiplication explicit."""
    # First normalize spaces
    expr_str = ' '.join(expr_str.split())
    
    # Handle terms with exponents first
    def add_mult_between_exp_terms(match):
        terms = match.group(1).split()
        return '*'.join(terms)
    
    # Handle expressions like x^(...) followed by other terms
    expr_str = re.sub(r'([a-zA-Z0-9^()\{\}]+(?:\s+[a-zA-Z0-9^()\{\}]+)+)', 
                      add_mult_between_exp_terms, 
                      expr_str)
    
    # Handle remaining pairs of variables
    while True:
        new_expr = re.sub(r'([a-zA-Z])(?:\s*)([a-zA-Z])', r'\1*\2', expr_str)
        if new_expr == expr_str:
            break
        expr_str = new_expr
    
    # Handle remaining cases of implicit multiplication
    expr_str = re.sub(r'(\d+)([a-zA-Z])', r'\1*\2', expr_str)  # 2x -> 2*x
    expr_str = re.sub(r'(\d+|\w+|\))(?=\()', r'\1*', expr_str)  # 2(x) -> 2*(x)
    
    return expr_str

def process_expression(expr_str: str) -> str:
    """Process mathematical expressions for parsing."""
    # First handle mixed numbers
    expr_str = process_mixed_numbers(expr_str)
    
    # Handle exponents before other operations
    expr_str = process_exponents(expr_str)
    
    # Remove spaces in fractions
    expr_str = process_fractions(expr_str)
    
    # Make implicit multiplication explicit
    expr_str = process_multiplication(expr_str)
    
    # Handle spacing around operators (except / which was handled in fractions)
    for op in ['+', '-', '*']:
        expr_str = re.sub(f'\\{op}', f' {op} ', expr_str)
    
    # Clean up spaces around parentheses
    expr_str = re.sub(r'\s*\(\s*', '(', expr_str)
    expr_str = re.sub(r'\s*\)\s*', ')', expr_str)
    
    # Final cleanup of multiple spaces
    expr_str = ' '.join(expr_str.split())
    
    return expr_str

def process_exponents(expr_str: str) -> str:
    """Handle various exponent notations."""
    # Remove spaces in fractions in exponents first
    def clean_exponent_fraction(match):
        exp = match.group(1)
        return '^(' + ''.join(exp.split()) + ')'
    
    # Handle curly brace exponents and convert to parentheses
    expr_str = re.sub(r'\^\{([^}]+)\}', clean_exponent_fraction, expr_str)
    
    # Handle existing parenthetical exponents
    expr_str = re.sub(r'\^\(([^)]+)\)', clean_exponent_fraction, expr_str)
    
    # Handle bare fraction exponents
    expr_str = re.sub(r'\^(\d+/\d+)', r'^(\1)', expr_str)
    
    return expr_str

def process_mixed_numbers(expr_str: str) -> str:
    """Convert mixed numbers to improper fractions."""
    parts = expr_str.split()
    result = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts) and re.match(r'^\d+$', parts[i]) and re.match(r'^\d+/\d+$', parts[i + 1]):
            whole = int(parts[i])
            num, den = map(int, parts[i + 1].split('/'))
            improper_num = whole * den + num
            result.append(f"{improper_num}/{den}")
            i += 2
        else:
            result.append(parts[i])
            i += 1
    return ' '.join(result)


class ExpressionEvaluator:
    def __init__(self, num_random_tests: int = 10):
        self.num_random_tests = num_random_tests
    
    def check_equivalence_random(self, equation1: str, equation2: str, tolerance: float = 1e-10) -> Tuple[bool, Dict]:
        """
        Check if two expressions are equivalent using random sampling.
        Returns (is_equivalent, details).
        """
        try:
            # Process exponents first
            equation1 = process_expression(equation1)
            equation2 = process_expression(equation2)

            # import ipdb; ipdb.set_trace()

            # Extract variables from both expressions
            expr1 = parse_expr(equation1.replace('^', '**'), transformations=standard_transformations)
            expr2 = parse_expr(equation2.replace('^', '**'), transformations=standard_transformations)
            variables = set(map(str, expr1.free_symbols | expr2.free_symbols))

            
            # Generate random points first
            random_points = []
            for _ in range(self.num_random_tests):
                var_values = {var: random.uniform(0.1, 10.0) for var in variables}
                random_points.append(var_values)
            
            # Evaluate both expressions using the same random points
            results1 = []
            results2 = []
            for var_values in random_points:
                r1 = float(expr1.evalf(subs=var_values))
                r2 = float(expr2.evalf(subs=var_values))
                results1.append(r1)
                results2.append(r2)
            
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
        # Ensure proper multiplication of terms
        result = 1.0
        for arg in expr.args:
            z3_arg = sympy_to_z3(arg, variables)
            if isinstance(result, float) and result == 1.0:
                result = z3_arg
            else:
                result *= z3_arg
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


def has_square_root(expr):
    """Check if expression contains square roots."""
    if isinstance(expr, sympy.Pow) and expr.exp.is_number and abs(expr.exp - 0.5) < 1e-10:
        return True
    for arg in expr.args if hasattr(expr, 'args') else []:
        if has_square_root(arg):
            return True
    return False

def get_variables_under_root(expr):
    """Get all variables that appear under a square root."""
    vars_under_root = set()
    if isinstance(expr, sympy.Pow) and expr.exp.is_number and abs(expr.exp - 0.5) < 1e-10:
        for symbol in expr.base.free_symbols:
            vars_under_root.add(str(symbol))
    for arg in expr.args if hasattr(expr, 'args') else []:
        vars_under_root.update(get_variables_under_root(arg))
    return vars_under_root

def check_equivalence_z3(equation1: str, equation2: str) -> Tuple[bool, Dict]:
    """
    Check if two expressions are equivalent using Z3.
    Returns (is_equivalent, details).
    """
    try:
        # Process expressions first
        equation1 = process_expression(equation1)
        equation2 = process_expression(equation2)
        
        # Parse the expressions using SymPy
        expr1 = parse_expr(equation1.replace('^', '**'), transformations=standard_transformations)
        expr2 = parse_expr(equation2.replace('^', '**'), transformations=standard_transformations)
        
        # Create Z3 solver
        s = Solver()
        
        # Create variables dictionary
        variables = {str(var): Real(str(var)) for var in expr1.free_symbols | expr2.free_symbols}
        
        # Convert to Z3 expressions
        z3_expr1 = sympy_to_z3(expr1, variables)
        z3_expr2 = sympy_to_z3(expr2, variables)
        
        # Add constraints to prevent division by zero
        for var in variables.values():
            s.add(var != 0)
            
        # Add constraints for denominators
        denominators1 = get_denominators(expr1)
        for denom in denominators1:
            z3_denom = sympy_to_z3(denom, variables)
            s.add(z3_denom != 0)
            
        denominators2 = get_denominators(expr2)
        for denom in denominators2:
            z3_denom = sympy_to_z3(denom, variables)
            s.add(z3_denom != 0)
        
        # Check for variables under square roots
        root_vars1 = get_variables_under_root(expr1)
        root_vars2 = get_variables_under_root(expr2)
        root_vars = root_vars1.union(root_vars2)
        
        # Add positivity constraints for variables under square roots
        for var_name in root_vars:
            if var_name in variables:
                s.add(variables[var_name] > 0)
        
        # Check equivalence
        s.add(z3_expr1 != z3_expr2)
        
        result = s.check()
        is_equivalent = result == unsat
        
        details = {
            'solver_result': str(result),
            'variables': list(variables.keys()),
            'root_variables': list(root_vars)
        }
        
        if result == sat:
            model = s.model()
            details['counterexample'] = {str(d): str(model[d]) for d in model.decls()}
        
        return is_equivalent, details
        
    except Exception as e:
        raise ValueError(f"Error checking equivalence: {str(e)}")

def get_denominators(expr):
    """Extract all denominators from a SymPy expression."""
    denominators = set()
    
    if isinstance(expr, sympy.core.power.Pow) and expr.exp.is_negative:
        denominators.add(expr.base ** abs(expr.exp))
    elif isinstance(expr, sympy.core.mul.Mul):
        for arg in expr.args:
            if isinstance(arg, sympy.core.power.Pow) and arg.exp.is_negative:
                denominators.add(arg.base ** abs(arg.exp))
    
    for arg in expr.args if hasattr(expr, 'args') else []:
        denominators.update(get_denominators(arg))
    
    return denominators


def test_expressions(json_file: str = None, test_cases: List[Tuple[str, str]] = None, 
                    stats_file: str = "decision_pair_stats.json", 
                    failures_file: str = "failed_cases.json"):
    """
    Test expressions using both Z3 and random sampling with detailed statistics.
    Logs results to separate files for decision pair statistics and failed cases.
    
    Args:
        json_file: Input JSON file with test cases
        test_cases: List of (expr1, expr2) tuples to test
        stats_file: Output file for decision pair statistics
        failures_file: Output file for failed test cases
    """
    evaluator = ExpressionEvaluator(num_random_tests=10)
    
    # Statistics tracking
    stats = {
        'total_tests': 0,
        'total_passed': 0,
        'total_failed': 0,
        'decision_pairs': {},
        'failed_cases': []
    }
    
    def run_tests(expr1: str, expr2: str, context: str = "", decision1: str = "", decision2: str = ""):
        print(f"\n{context}")
        print(f"Testing: {expr1} ≟ {expr2}")
        
        test_result = {
            'expression1': expr1,
            'expression2': expr2,
            'context': context,
            'decision1': decision1,
            'decision2': decision2,
            'z3_result': None,
            'random_result': None,
            'counterexample': None,
            'max_difference': None,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Test with Z3
        try:
            z3_equiv, z3_details = check_equivalence_z3(expr1, expr2)
            print(f"Z3 Result: {'Equivalent' if z3_equiv else 'Not equivalent'}")
            test_result['z3_result'] = z3_equiv
            if not z3_equiv and 'counterexample' in z3_details:
                print(f"Z3 Counterexample: {z3_details['counterexample']}")
                test_result['counterexample'] = z3_details['counterexample']
        except Exception as e:
            print(f"Z3 Error: {str(e)}")
            test_result['z3_error'] = str(e)
            test_result['z3_result'] = None
        
        # Test with random sampling
        try:
            random_equiv, random_details = evaluator.check_equivalence_random(expr1, expr2)
            print(f"Random Sampling Result: {'Equivalent' if random_equiv else 'Not equivalent'}")
            print(f"Max difference: {random_details['max_difference']:.2e}")
            test_result['random_result'] = random_equiv
            test_result['max_difference'] = random_details['max_difference']
        except Exception as e:
            print(f"Random Sampling Error: {str(e)}")
            test_result['random_error'] = str(e)
            test_result['random_result'] = None
        
        # Update statistics
        stats['total_tests'] += 1
        
        # Consider test passed if either method confirms equivalence
        is_equivalent = (test_result['z3_result'] or test_result['random_result'])
        
        if is_equivalent:
            stats['total_passed'] += 1
            if decision1 and decision2:
                key = (decision1, decision2)
                if key not in stats['decision_pairs']:
                    stats['decision_pairs'][key] = {
                        'decision1': decision1,
                        'decision2': decision2,
                        'passed': 0,
                        'failed': 0,
                        'total': 0,
                        'examples': []
                    }
                stats['decision_pairs'][key]['passed'] += 1
                stats['decision_pairs'][key]['total'] += 1
                stats['decision_pairs'][key]['examples'].append({
                    'expression1': expr1,
                    'expression2': expr2,
                    'result': 'passed'
                })
        else:
            stats['total_failed'] += 1
            if decision1 and decision2:
                key = (decision1, decision2)
                if key not in stats['decision_pairs']:
                    stats['decision_pairs'][key] = {
                        'decision1': decision1,
                        'decision2': decision2,
                        'passed': 0,
                        'failed': 0,
                        'total': 0,
                        'examples': []
                    }
                stats['decision_pairs'][key]['failed'] += 1
                stats['decision_pairs'][key]['total'] += 1
                stats['decision_pairs'][key]['examples'].append({
                    'expression1': expr1,
                    'expression2': expr2,
                    'result': 'failed'
                })
            stats['failed_cases'].append(test_result)
        
        return is_equivalent

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
                        run_tests(before, after, "JSON test case", decision1, decision2)
        
        except Exception as e:
            print(f"Error processing JSON file: {str(e)}")
    
    # Prepare decision pair statistics for logging
    decision_stats = {
        'summary': {
            'total_tests': stats['total_tests'],
            'total_passed': stats['total_passed'],
            'total_failed': stats['total_failed'],
            'pass_rate': f"{(stats['total_passed']/stats['total_tests']*100):.1f}%"
        },
        'decision_pairs': {}
    }
    
    # Convert decision pairs to a format suitable for JSON
    for (decision1, decision2), results in stats['decision_pairs'].items():
        key = f"{decision1} -> {decision2}"
        decision_stats['decision_pairs'][key] = {
            'decision1': decision1,
            'decision2': decision2,
            'passed': results['passed'],
            'failed': results['failed'],
            'total': results['total'],
            'pass_rate': f"{(results['passed']/results['total']*100):.1f}%",
            'examples': results['examples']
        }
    
    # Save decision pair statistics to file
    with open(stats_file, 'w') as f:
        json.dump(decision_stats, f, indent=2)
    print(f"\nDecision pair statistics saved to {stats_file}")
    
    # Save failed cases to file
    if stats['failed_cases']:
        failed_cases_data = {
            'total_failed': len(stats['failed_cases']),
            'cases': stats['failed_cases']
        }
        with open(failures_file, 'w') as f:
            json.dump(failed_cases_data, f, indent=2)
        print(f"Failed cases saved to {failures_file}")
    
    # Print summary to console
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    print(f"Total tests run: {stats['total_tests']}")
    print(f"Total passed: {stats['total_passed']} ({(stats['total_passed']/stats['total_tests']*100):.1f}%)")
    print(f"Total failed: {stats['total_failed']} ({(stats['total_failed']/stats['total_tests']*100):.1f}%)")
    print(f"\nDetailed statistics saved to {stats_file}")
    if stats['failed_cases']:
        print(f"Failed cases saved to {failures_file}")
    
    return stats

if __name__ == "__main__":
    import os
    import sys
    
    # Create data_stats directory if it doesn't exist
    output_dir = "data_stats"
    os.makedirs(output_dir, exist_ok=True)
    
    # List of datasets to process
    datasets = [
        # 'data/dataset.json',
        'data/complex_dataset.json',
        # Add more dataset paths here
    ]
    
    # Process command line arguments if provided
    if len(sys.argv) > 1:
        datasets = sys.argv[1:]

    test_expressions(
        test_cases = [
            ('((x + 1)^2)/(x + 1)', 'x + 1'), # this one is important, keep this here!
            ('2/(x^(1/2) + y^(1/2)) + z + 3a + z^3', '(2(x^(1/2) - y^(1/2)))/(x - y) + z + 3a + z^3'),
            ('1 / (1 / x + 1 / y + 1 / z) + a + b^2 + z^2','(x y z) / (y z + x z + x y) + a + b^2 + z^2'),
            ('1/(1/x^{1/2} + 1/y^{1/2})','(x^{1/2}y^{1/2})/(x^{1/2} + y^{1/2})')
        ]
    )
    
    # Process each dataset
    for dataset_path in datasets:
        try:
            # Extract dataset name without path and extension
            dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
            
            # Create output file paths
            stats_file = os.path.join(output_dir, f"{dataset_name}_decision_stats.json")
            failures_file = os.path.join(output_dir, f"{dataset_name}_failed_cases.json")
            
            print(f"\nProcessing dataset: {dataset_path}")
            print(f"Stats will be saved to: {stats_file}")
            print(f"Failed cases will be saved to: {failures_file}")
            
            # Run tests for this dataset
            stats = test_expressions(
                json_file=dataset_path,
                test_cases=None,
                stats_file=stats_file,
                failures_file=failures_file
            )
            
            print(f"Finished processing {dataset_path}\n")
            print("-" * 80)
            
        except Exception as e:
            print(f"Error processing dataset {dataset_path}: {str(e)}")
            continue