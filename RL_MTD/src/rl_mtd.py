from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2, ACKTR, A2C, SAC, TD3

from src.env.mtd_env import MTDEnv

# RL Algos
#   Model Free
#       Policy Optimization
#           ACKTR
#           A2C
#           PP02
#       Q-Learning
#           * not usefull as only partially observable model (agent does not "know" the state)
#       Hybrid
#           SAC
#           TD3
#   Model Based
#       * no support from stable_baselines


# ------------------------- config ------------------------- #
learn = False
simulate_only_best = True
results_file = "results.txt"


# ------------------------ learning ------------------------ #
def get_best_algorithm():
    # type: () -> str
    try:
        file = open(results_file)
        last_line = None
        for line in file:
            last_line = line
        file.close()

        return last_line.split(" ")[0]

    except FileNotFoundError:
        return "Random"


env = MTDEnv()
model = PPO2(MlpPolicy, env, verbose=0)
algorithms = [PPO2, ACKTR, A2C, SAC, TD3]

if learn:
    for algorithm in algorithms:
        import time

        start = time.time()
        timesteps = 10 ** 6

        print(f"Learing {algorithm}...")
        print(f"{timesteps} steps to simulate.")
        print(f"Estimated time: {timesteps / 10 ** 5} min.")

        # crazy magic here
        model.learn(total_timesteps=timesteps)
        model.save(algorithm)

        print(f"{time.time() - start} sec to simulate {timesteps} steps")

# ------------------------ evaluating ------------------------ #
best_avg_steps = [0, "Random"]
show_model_after_each_step = False
show_results_after_each_sim = False

simulations_count = 100

if simulate_only_best:
    sim_types = [get_best_algorithm(), "Random", "Static"]
else:
    sim_types = [*algorithms, "Random", "Static"]

results = dict.fromkeys(sim_types)
# {
#   "PP02": {
#       "steps": List[int],
#       "total_reward": List[int],
#       "avg_steps": float,
#       "avg_rewards": float
#   },
#   "Random": {...},
#   ...
# }

f = open(results_file, "w")
f.write("*********************************************************************\n")
f.write(f"Starting Simulation Types: {', '.join(sim_types)}\n")
f.write(f"Simulations per Type:      {simulations_count}\n")

for sim_type in sim_types:
    f.write("*********************************************************************\n")
    f.write(f"Simulation Type: {sim_type}\n")

    min_steps = env.get_simulation_steps()
    max_steps = 0

    results[sim_type] = {}
    results[sim_type]["steps"] = []
    results[sim_type]["total_reward"] = []
    results[sim_type]["defender_wins"] = 0

    null_action = [0, 0]

    for _ in range(simulations_count):
        obs = env.reset()
        done = False

        # run 1 simulation (until attacker or defender wins)
        while not done:

            # select action according to sim_type
            if sim_type is "Random":
                action = env.action_space.sample()
            elif sim_type is "Static":
                action = null_action
            else:
                action, _ = model.predict(obs)

            # do action on model
            obs, rewards, done, info = env.step(action)
            if show_model_after_each_step:
                env.render()

        results[sim_type]["steps"].append(env.get_counter())
        results[sim_type]["total_reward"].append(env.get_total_reward() / env.get_counter())

        if env.defender_wins():
            results[sim_type]["defender_wins"] += 1

        if env.get_counter() < min_steps:
            min_steps = env.get_counter()

        if env.get_counter() > max_steps:
            max_steps = env.get_counter()

        if show_results_after_each_sim:
            print("\n" + str(env))

    sum_steps = 0
    sum_rewards = 0

    for steps in results[sim_type]["steps"]:
        sum_steps += steps

    for total_rewards in results[sim_type]["total_reward"]:
        sum_rewards += total_rewards

    avg_steps = round(sum_steps / simulations_count, 3)
    avg_reward = round(sum_rewards / simulations_count, 3)

    if avg_steps > best_avg_steps[0]:
        best_avg_steps = [avg_steps, sim_type]

    results[sim_type]["avg_steps"] = avg_steps
    results[sim_type]["avg_reward"] = avg_reward

    defender_wins = results[sim_type]['defender_wins']
    attacker_wins = simulations_count - defender_wins

    f.write(f"*****************{'*' * len(sim_type)}\n")
    f.write(f"Avg steps:        {avg_steps}\n")
    f.write(f"Avg reward/step: {avg_reward}\n")
    f.write(f"Min steps:        {min_steps}\n")
    f.write(f"Max steps:        {max_steps}\n\n")
    f.write(f"Defender wins:    {defender_wins}\n")
    f.write(f"Attacker wins:    {attacker_wins}\n")

f.write("*********************************************************************\n")
f.write(f"{best_avg_steps[1]} is the best algorithm with {best_avg_steps[0]} avg steps.\n")
f.close()

f = open(results_file, "r")
s = "".join(f.readlines())
print(s)
f.close()
