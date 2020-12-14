import src.rl_mtd as mtd

learn = True
path = "parameters/tests/"

# ------------------------- info ------------------------- #
if learn:
    print(f"Start learning and simulating in {path}.")

else:
    print(f"Start simulating in {path}.")

# ------------------------ runner ------------------------ #
for i in range(3, 8):
    mtd.main(path,
             learn=learn,
             learn_steps=10 ** i,
             graph="simple_webservice",
             attack="professional",
             simulations_count=20,
             only_nodes=False,
             only_prevention_systems=False,
             nodes_pause=1,
             prevention_systems_pause=1)
