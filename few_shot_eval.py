# load from a pickle file
import pickle
from check_equivalence import check_equivalence_z3

with open('/homes/ecaplan/exp_simp/few_shot_results/few_shot_results.pkl', 'rb') as f:
    fs_results = pickle.load(f)

(pis, xs, final_xs, pi_scores, h_scores) = fs_results

equivs = []

for i in range(len(pis)):
    x = xs[i]
    final_x = final_xs[i]
    try:
        is_equivalent, details = check_equivalence_z3(x, final_x)
    except Exception as e:
        print(e)
        is_equivalent = False
    if is_equivalent:
        equivs.append(i)

print(f"Number of equivalent results: {len(equivs)}")
print(f"Number of non-equivalent results: {len(pis) - len(equivs)}")
print(f"Percentage of equivalent results: {len(equivs)/len(pis)*100:.2f}%")
