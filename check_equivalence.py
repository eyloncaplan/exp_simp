from z3 import *
# Things that this file should do:

# - A function to check if two equations are equivalent
# - A parser to parse text and convert it into an equation (could use some python smt parsing library, combined with some llm assistance)
# - Parse the json file, and see how many of the things are valid

def convert_to_z3(parsed):

    pass


def check_equivalence(equation1, equation2):
    equation1 = convert_to_z3(equation1)
    equation2 = convert_to_z3(equation2)
    # use z3 to check if the two equations are equivalent below
    s = Solver()
    s.add(equation1 != equation2)
    if s.check() == unsat:
        return True
    else:
        return False


