Code related to intermediate beta has been cleaned in the main code base to ease
new development, since it looks old and not in use. It may be revived in the
future if it is still needed.

Commit before the cleaning: 1de2320eb3b10f45664e2283529630b2eaf00be0

Involved files include:

- auto_LiRPA/beta_crown.py
- auto_LiRPA/operators/relu.py
- auto_LiRPA/optimized_bounds.py
- complete_verifier/alpha.py
- complete_verifier/attack/bab_attack.py
- complete_verifier/beta_CROWN_solver.py
- complete_verifier/batch_branch_and_bound.py
