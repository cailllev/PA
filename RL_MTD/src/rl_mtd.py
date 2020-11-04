from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2

from src.env.mtd_env import MTDEnv

# ------------------------ learning ------------------------ #
learn = False

env = MTDEnv()
model = PPO2(MlpPolicy, env, verbose=0)

if learn:
    import time
    start = time.time()
    timesteps = 10**6

    print("Learing...")
    print(f"{timesteps} steps to simulate.")
    print(f"Estimated time: {timesteps / 10**5} min.")

    # crazy magic here
    model.learn(total_timesteps=timesteps)
    model.save("PPO2_Agent")

    print(f"{time.time() - start} sec to simulate {timesteps} steps")

else:
    model.load("PPO2_Agent")

# ------------------------ evaluating ------------------------ #
show_model_after_each_step = False
show_results_after_each_sim = False

simulations_count = 200

sim_types = ["RL", "Random", "Static"]
results = dict.fromkeys(sim_types)
# {
#   "RL": {
#       "steps": List[int],
#       "total_reward": List[int],
#       "avg_steps": float,
#       "avg_rewards": float
#   },
#   "Random": {...},
#   ...
# }

print("*********************************************************************")
print(f"Starting Simulation Types: {', '.join(sim_types)}")
print(f"Simulations per Type:      {simulations_count}")

for sim_type in sim_types:
    print("*********************************************************************")
    print("Simulation Type: " + sim_type)

    min_steps = env.get_simulation_steps()
    max_steps = 0

    results[sim_type] = {}
    results[sim_type]["steps"] = []
    results[sim_type]["total_reward"] = []

    null_action = [0, 0]

    for _ in range(simulations_count):
        obs = env.reset()
        done = False

        # run 1 simulation (until attacker or defender wins)
        while not done:

            # select action according to sim_type
            if sim_type is "RL":
                action, _ = model.predict(obs)
            elif sim_type is "Random":
                action = env.action_space.sample()
            else:
                action = null_action

            # do action on model
            obs, rewards, done, info = env.step(action)
            if show_model_after_each_step:
                env.render()

        results[sim_type]["steps"].append(env.get_counter())
        results[sim_type]["total_reward"].append(env.get_total_reward() / env.get_counter())

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
    results[sim_type]["avg_steps"] = avg_steps
    results[sim_type]["avg_reward"] = avg_reward

    print("")
    print(f"Avg steps:        {avg_steps}")
    print(f"Avg reward/step: {avg_reward}")
    print(f"Min steps:        {min_steps}")
    print(f"Max steps:        {max_steps}")
