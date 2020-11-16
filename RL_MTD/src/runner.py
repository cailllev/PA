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
    mtd.main(path, learn=learn, timesteps=10**i, only_nodes=True)
