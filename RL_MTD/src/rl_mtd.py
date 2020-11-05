from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2, ACKTR, A2C

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


random = "Random"
static = "Static"

# ------------------------- config ------------------------- #
learn = True
timesteps = 10 ** 7

simulate_only_best = False
simulations_count = 1000

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
        return random


env = MTDEnv()
algorithms = {
    PPO2: {"name": "PPO2", "kwargs": {}},
    ACKTR: {"name": "ACKTR", "kwargs": {}},
    A2C: {"name": "A2C", "kwargs": {}},
}
default_algorithm = PPO2

if learn:
    print("*********************************************")
    print(f"Learning {len(algorithms)} algorithms.")
    for algorithm in algorithms:
        algorithm_name = algorithms[algorithm]["name"]
        import time

        start = time.time()

        print("*********************************************")
        print(f"Learning {algorithm_name}.")
        print(f"{timesteps} steps to simulate.")
        print(f"Estimated time: {timesteps / 10 ** 5} min.")
        print("...")

        model = algorithm(MlpPolicy, env, verbose=0, **algorithms[algorithm]["kwargs"])
        model.load(algorithm_name)
        model.learn(total_timesteps=timesteps)
        model.save(algorithm_name)

        print(f"{round(time.time() - start)} seconds to simulate {timesteps} steps.")

# ------------------------ evaluating ------------------------ #
best_avg_steps = [0, random]
show_model_after_each_step = False
show_results_after_each_sim = False

if simulate_only_best:
    # delete all algorihms except the best from dict
    best_algo = get_best_algorithm()
    for algo in algorithms:
        if algorithms[algo]["name"] == best_algo:
            val = algorithms[algo]
            algorithms.clear()
            algorithms[algo] = val
            break

# add random and static for evaluation
algorithms[random] = {"name": random}
algorithms[static] = {"name": static}

algorithm_names = []
for algorithm_name in algorithms:
    algorithm_names.append(algorithms[algorithm_name]["name"])

results = dict.fromkeys(algorithm_names)
# restults = {
#   "PP02": {
#       "steps": List[int],
#       "total_reward": List[int],
#       "min_steps": int,
#       "max_steps": int,
#       "avg_steps": float,
#       "avg_rewards": float
#   },
#   ...,
#   "Random": {...},
#   "Static": {...}
# }

print("*********************************************")
print(f"Prepared to simulate {', '.join(algorithm_names)} with {simulations_count} simulations")
print("...")


def update_results(algo_name):
    global results
    results[algo_name]["steps"].append(env.get_counter())
    results[algo_name]["total_reward"].append(env.get_total_reward() / env.get_counter())

    if env.defender_wins():
        results[algo_name]["defender_wins"] += 1

    if env.get_counter() < results[algo_name]["min_steps"]:
        results[algo_name]["min_steps"] = env.get_counter()

    if env.get_counter() > results[algo_name]["max_steps"]:
        results[algo_name]["max_steps"] = env.get_counter()

    if show_results_after_each_sim:
        print("\n" + str(env))


def evaluate_and_save_results(algo_name):
    global results
    global best_avg_steps

    sum_steps = 0
    for steps in results[algo_name]["steps"]:
        sum_steps += steps

    sum_rewards = 0
    for total_rewards in results[algo_name]["total_reward"]:
        sum_rewards += total_rewards

    avg_steps = round(sum_steps / simulations_count, 3)
    avg_reward = round(sum_rewards / simulations_count, 3)

    if avg_steps > best_avg_steps[0]:
        best_avg_steps = [avg_steps, algo_name]

    min_steps = results[algo_name]["min_steps"]
    max_steps = results[algo_name]["max_steps"]

    results[algo_name]["avg_steps"] = avg_steps
    results[algo_name]["avg_reward"] = avg_reward

    defender_wins = results[algo_name]['defender_wins']
    attacker_wins = simulations_count - defender_wins

    f.write("*********************************************************************\n")
    f.write(f"Simulation Type: {algo_name}\n")
    f.write(f"*****************{'*' * len(algo_name)}\n")
    f.write(f"Avg steps:        {avg_steps}\n")
    f.write(f"Avg reward/step: {avg_reward}\n")
    f.write(f"Min steps:        {min_steps}\n")
    f.write(f"Max steps:        {max_steps}\n\n")
    f.write(f"Defender wins:    {defender_wins}\n")
    f.write(f"Attacker wins:    {attacker_wins}\n")


def run_random_simulation():
    env.reset()
    done = False

    while not done:
        action = env.action_space.sample()

        # do action on model
        obs, rewards, done, info = env.step(action)
        if show_model_after_each_step:
            env.render()


def run_static_simulation():
    env.reset()
    done = False
    null_action = [0, 0]

    while not done:
        obs, rewards, done, info = env.step(null_action)
        if show_model_after_each_step:
            env.render()


def run_rl_simulation(mod):
    obs = env.reset()
    done = False

    while not done:
        action, _ = mod.predict(obs)

        # do action on model
        obs, rewards, done, info = env.step(action)
        if show_model_after_each_step:
            env.render()


f = open(results_file, "w")
f.write("*********************************************************************\n")
f.write(f"Starting Simulation Types: {', '.join(algorithm_names)}\n")
f.write(f"Simulations per Type:      {simulations_count}\n")

for algorithm in algorithms:
    algorithm_name = algorithms[algorithm]["name"]

    if algorithm_name is not random and algorithm_name is not static:
        try:
            model = algorithm(MlpPolicy, env, verbose=0, **algorithms[algorithm]["kwargs"])
            model.load(algorithm_name)
        except ValueError:
            f.write(f"{algorithm_name} not trained yet, skipping to next algo\n")
            continue
    else:
        # only used to omit warning
        model = default_algorithm(MlpPolicy, env, verbose=0)

    results[algorithm_name] = {}
    results[algorithm_name]["steps"] = []
    results[algorithm_name]["min_steps"] = env.get_simulation_steps()
    results[algorithm_name]["max_steps"] = 0
    results[algorithm_name]["total_reward"] = []
    results[algorithm_name]["defender_wins"] = 0

    for _ in range(simulations_count):
        # run 1 simulation (until attacker or defender wins)
        if algorithm_name is random:
            run_random_simulation()

        elif algorithm_name is static:
            run_static_simulation()

        else:
            run_rl_simulation(model)

        update_results(algorithm_name)

    evaluate_and_save_results(algorithm_name)

f.write("*********************************************************************\n")
f.write(f"{best_avg_steps[1]} is the best algorithm with {best_avg_steps[0]} avg steps.\n")
f.close()

f = open(results_file, "r")
s = "".join(f.readlines())
print(s)
f.close()
