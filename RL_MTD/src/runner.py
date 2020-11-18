import src.rl_mtd as mtd

learn = True
path = "parameters/v3_normal/"

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
             simulations_count=200,
             attack="professional",
             only_nodes=False,
             only_detection_systems=False,
             nodes_pause=1,
             detection_systems_pause=1)
