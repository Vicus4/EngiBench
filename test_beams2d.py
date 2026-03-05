from engibench.problems.beams2d.v0 import Beams2D

# Create a problem
problem = Beams2D(seed=0)

# Inspect problem
problem.design_space  # Box(0.0, 1.0, (50, 100), float64)
problem.objectives  # (("compliance", "MINIMIZE"),)
problem.conditions  # (("volfrac", 0.35), ("forcedist", 0.0),...)
problem.dataset # A HuggingFace Dataset object

# Train your models, e.g., inverse design
# inverse_model = train_inverse(problem.dataset)
desired_conds = {"volfrac": 0.7, "forcedist": 0.3}
# generated_design = inverse_model.predict(desired_conds)

random_design, _ = problem.random_design()
# check constraints on the design, config pair
violated_constraints = problem.check_constraints(design=random_design, config=desired_conds)

if not violated_constraints:
   # Only simulate to get objective values
   objs = problem.simulate(design=random_design, config=desired_conds)
   problem.reset(seed=42)
   # Or run a gradient-based optimizer to polish the design
   opt_design, history = problem.optimize(starting_point=random_design, config=desired_conds)