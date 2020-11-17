import src.rl_mtd as mtd

learn = True
path = "parameters/v3_max_security/"

# ------------------------- info ------------------------- #
if learn:
    print(f"Start learning and simulating in {path}.")

else:
    print(f"Start simulating in {path}.")

# ------------------------ runner ------------------------ #
for i in range(3, 8):
    mtd.main(path,
             learn=learn,
             timesteps=10**i,
             simulations_count=20,
             only_nodes=False,
             only_detection_systems=False,
             nodes_pause=1,
             detection_systems_pause=1)
