from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2

from src.env.mtd_env import MTDEnv

learn = False

env = MTDEnv()
model = PPO2(MlpPolicy, env, verbose=0)

if learn:
    import time
    start = time.time()
    timesteps = 10**6

    print("Learing...")
    print(f"{timesteps} steps to simulate.")
    print(f"Estimated time: {timesteps /10**5} min.")

    # crazy magic here
    model.learn(total_timesteps=timesteps)
    model.save("PPO2_Agent")

    print(f"{time.time() - start} sec to simulate {timesteps} steps")

else:
    model.load("PPO2_Agent")


simulations = 100
verbose = 0
results = []
min_steps = 1000
max_steps = 0

for _ in range(simulations):
    obs = env.reset()
    done = False

    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if verbose:
            env.render()

    results.append([env.get_steps(), env.get_total_reward() / env.get_steps()])

    if env.get_steps() < min_steps:
        min_steps = env.get_steps()

    if env.get_steps() > max_steps:
        max_steps = env.get_steps()

    print("\n" + str(env))

sum_steps = 0
sum_rewards = 0

for steps, total_reward in results:
    sum_steps += steps
    sum_rewards += total_reward

avg_steps = sum_steps / len(results)
avg_reward = sum_rewards / len(results)

print("*********************************************************************")
print(f"\nAverage steps: {avg_steps}\nAverage reward per step: {avg_reward}")
print(f"\nMin steps: {min_steps}\nMax steps: {max_steps}")
