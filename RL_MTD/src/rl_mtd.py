from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import A2C, PPO

from src.env.mtd_env import MTDEnv
from src.defender2000 import Defender2000

from datetime import datetime
import time
import os

# stable baselines3 agents
# Name      Box         Discrete    MultiD.     MultiBinary Multi Processing
# A2C       1           1           1           1           1   ️
# DDPG      1           -           -           -           -
# DQN       -           1           -           -           -
# HER       1           1           -           -           -
# PPO       1           1           1           1           1
# SAC       1           -           -           -           -
# TD3       1           -           -           -           -

# https://medium.com/@SmartLabAI/reinforcement-learning-algorithms-an-intuitive-overview-904e2dff5bbc#2bfa
#
# RL Algos
#   Model Free
#       Policy Optimization (ActorCriticRLModel)
#           A2C     Advantage Actor Critic
#                       - Advantage: Similarly to PG where the update rule used the dicounted returns from a set of
#                       experiences in order to tell the agnet which acttions were “good” or “bad”.
#                       - Actor-critic: combines the benefits of both approaches from policy-iteration method as PG and
#                       value-iteration method as Q-learning (See below). The network will estimate both a value
#                       function V(s) (how good a certain state is to be in) and a policy π(s).
#
#           PPO    Proximal Policy Optimization algorithm
#                       Also an on-policy algorithm which similarly to TRPO can perform on discrete or continuous action
#                       spaces. PPO shares motivation with TRPO in the task of answering the question: how to increase
#                       policy improvement without the risk of performance collapse? The idea is that PPO improves the
#                       stability of the Actor training by limiting the policy update at each training step.
#
#       Q-Learning
#           DQN     Deep Q Neural Network (OffPolicyRLModel)
#                       ------------------------------------------------------------------------------------------------
#                       Too much overhead to implement DQN, plus DQN is even worse than Defender2000 after initial tests
#                       ------------------------------------------------------------------------------------------------
#                       DQN is Q-learning with Neural Networks . The motivation behind is simply related to big state
#                       space environments where defining a Q-table would be a very complex, challenging and time-
#                       consuming task. Instead of a Q-table Neural Networks approximate Q-values for each action based
#                       on the state.
#
#           HER     Hindsight Experience Replay (BaseRLModel)
#                       ------------------------------------------------------------------------------------------------
#                       Too many challanges to implement HER, unfixable with current setup. Initial list of requirements
#                           - cannot use normal env, needs VecEnv (but not DummyVecEnv, doesnt work)
#                           - cannot create env in SubProcEnv, multiprocessing exception
#                           - cannot use MultiDiscrete
#                       ------------------------------------------------------------------------------------------------
#                       In Hindsight Experience Replay method, basically a DQN is suplied with a state and a desired
#                       end-state, or in other words goal. It allow to quickly learn when the rewards are sparse. In
#                       other words when the rewards are uniform for most of the time, with only a few rare reward-
#                       values that really stand out.
#
#       Hybrid (OffPolicyRLModel)
#           DDPG    Deep Deterministic Policy Gradient
#                       ------------------------------------------------------------------------------------------------
#                       only Box space supported, not fixable (and not reasonable to use)!
#                       ------------------------------------------------------------------------------------------------
#                       https://arxiv.org/pdf/1802.09477.pdf
#                       [...] Our algorithm builds on Double Q-learning, by taking the minimum value between a pair of
#                       critics to limit overestimation. We draw the connection between target networks and
#                       overestimation bias, and suggest delaying policy updates to reduce per-update error and further
#                       improve performance. [...]
#
#           SAC     Soft Actor Critic
#                       ------------------------------------------------------------------------------------------------
#                       only Box space supported, not fixable (and not reasonable to use)!
#                       ------------------------------------------------------------------------------------------------
#                       https://arxiv.org/abs/1801.01290
#                       [...]. By combining off-policy updates with a stable stochastic actor-critic formulation, our
#                       method achieves state-of-the-art performance on a range of continuous control benchmark tasks,
#                       outperforming prior on-policy and off-policy methods. [...]
#
#           TD3     Twin Delayed DDPG
#                       ------------------------------------------------------------------------------------------------
#                       only Box space supported, not fixable (and not reasonable to use)!
#                       ------------------------------------------------------------------------------------------------
#                       https://arxiv.org/pdf/1509.02971.pdf
#                       We adapt the ideas underlying the success of Deep Q-Learning to the continuous
#                       action domain. We present an actor-critic, model-free algorithm based on the deterministic
#                       policy gradient that can operate over continuous action spaces. [...]
#
#   Model Based
#       * no support from stable_baselines plus not usefull because agent cannot predict the next state most of the time


print("Initializing...")
# TODO add /v3_only_restart_node, /v3_only_switch_detection_system
# TODO redo all test and learnings, bug in env where graph did not reset

# ------------------------- const ------------------------- #
rl = "RL"
random = "Random"
defender2000 = "Defender2000"
static = "Static"

non_rl = [random, defender2000, static]

# ------------------------- config ------------------------- #
learn = False
timesteps = 10 ** 6

simulations_count = 100

simulate_only_one = False  # False -> simulate all, A2C -> simulate only A2C

show_model_after_each_step = False
show_results_after_each_sim = False

# ------------------------- init ------------------------- #
env = MTDEnv()

# only to learn, Defender2000, Random and Static are added later
# after 50 steps the agent updates it policy (it learns each 50 steps)
algorithms = {
    "A2C": A2C(ActorCriticPolicy, env, n_steps=50, verbose=0),
    "PPO": PPO(ActorCriticPolicy, env, n_steps=50, verbose=0),
}

# times estimations
base_learn_time_estimate = (timesteps / 10 ** 5) * 60
alorithm_learn_times = {
    "A2C": base_learn_time_estimate * 1.4,
    "PPO": base_learn_time_estimate * 2.5
}
steps_per_sim = env.get_steps_per_simulation()
sim_time_estimate = simulations_count * steps_per_sim / 1500

parameters_folder = "parameters/v3/"
if simulate_only_one:
    results_file = "temp.txt"
    if simulate_only_one in non_rl:
        sim_time_estimate //= 200
    else:
        sim_time_estimate /= 2
else:
    results_file = f"{parameters_folder}results_{simulations_count}_simulations.txt"

print("Initialization complete.")


def timestamp_to_datetime(stamp):
    # type: (float) -> str
    return datetime.fromtimestamp(stamp).strftime('%I:%M:%S')


# ------------------------ learning ------------------------ #
if learn:
    print("*********************************************")
    print(f"Learning {', '.join(algorithms.keys())} algorithms.")
    for algorithm in algorithms:

        model = algorithms[algorithm]
        env.reset()

        print("*********************************************")
        print(f"Learning {algorithm}.")

        try:
            model.load(parameters_folder + algorithm)
            print(f"Found parameters.")

        except FileNotFoundError:
            print(f"No learned parameters found, start from scratch.")

        start = time.time()

        print(f"{timesteps} steps to learn, with updates to policy each {model.n_steps} steps.")
        print(f"Start:              {timestamp_to_datetime(start)}.")
        print(f"Estimated finish:   {timestamp_to_datetime(start+alorithm_learn_times[algorithm])}.")
        print("...")

        model.learn(total_timesteps=timesteps)
        model.save(parameters_folder + algorithm)

        print(f"Actual finish:      {timestamp_to_datetime(time.time())}.")

# ------------------------ simulation helpers ------------------------ #
best_avg_steps = [0, random]
best_avg_reward = [0, random]

# add non rl algorithms (random, defender2000 and static) for evaluation
algorithms[random] = random
algorithms[defender2000] = Defender2000()
algorithms[static] = static

if simulate_only_one:
    val = algorithms[simulate_only_one]
    algorithms.clear()
    algorithms[simulate_only_one] = val

start = time.time()
print("*********************************************")
print(f"Prepared to simulate {', '.join(algorithms.keys())} with {simulations_count} simulations.")
print(f"Start:              {timestamp_to_datetime(start)}.")
print(f"Estimated finish:   {timestamp_to_datetime(start+sim_time_estimate)}.")
print("...")


def run_simulation(algorithm_type, m):
    obs = env.reset()
    done = False
    null_action = [0, 0]

    while not done:
        if algorithm_type is rl:
            action, _ = m.predict(obs)

        elif algorithm_type is defender2000:
            action = m.predict(obs)

        elif algorithm_type is random:
            action = env.action_space.sample()

        elif algorithm_type is static:
            action = null_action

        else:
            print(f"Unknown algorithm type: {algorithm_type}")
            action = null_action

        # do action on model
        obs, rewards, done, info = env.step(action)
        if show_model_after_each_step:
            env.render()


def update_results():
    global results
    results["steps"].append(env.get_counter())
    results["total_reward"].append(env.get_total_reward())
    results["null_action_ratio"] += env.get_null_action_ratio()

    if env.defender_wins():
        results["defender_wins"] += 1

    if env.get_counter() < results["min_steps"]:
        results["min_steps"] = env.get_counter()

    if env.get_counter() > results["max_steps"]:
        results["max_steps"] = env.get_counter()

    if show_results_after_each_sim:
        print("\n" + str(env))


def evaluate_and_save_results(algo_name):
    global results
    global best_avg_steps
    global best_avg_reward

    steps = results["steps"]
    rewards = results["total_reward"]

    sum_steps = sum(steps)
    sum_rewards = sum(rewards)

    avg_steps = round(sum_steps / simulations_count, 3)
    avg_reward = round(sum_rewards / sum_steps, 3)
    avg_null_action_ratio = round(results["null_action_ratio"] / simulations_count, 3)

    if avg_steps > best_avg_steps[0]:
        best_avg_steps = [avg_steps, algo_name]

    if avg_reward > best_avg_reward[0]:
        best_avg_reward = [avg_reward, algo_name]

    min_steps = results["min_steps"]
    max_steps = results["max_steps"]

    defender_wins = results['defender_wins']
    attacker_wins = simulations_count - defender_wins

    f.write("*********************************************************************\n")
    f.write(f"Simulation Type:          {algo_name}\n")
    f.write(f"--------------------------------------------------------------------\n")
    f.write(f"Avg steps:                {avg_steps}\n")
    f.write(f"Avg reward/step:          {avg_reward}\n")
    f.write(f"Avg null action ratio:    {avg_null_action_ratio}\n\n")
    f.write(f"Min steps:                {min_steps}\n")
    f.write(f"Max steps:                {max_steps}\n\n")
    f.write(f"Defender wins:            {defender_wins}\n")
    f.write(f"Attacker wins:            {attacker_wins}\n\n")
    f.write(f"Steps each sim:           {steps}\n")
    f.write(f"Reward each sim:          {rewards}\n")


# ------------------------ actual simulation ------------------------ #
f = open(results_file, "w")
f.write("*********************************************************************\n")
f.write(f"Starting Simulation Types: {', '.join(algorithms.keys())}\n")
f.write(f"Simulations per Type:      {simulations_count}\n")
f.write(f"Steps per Simulation:      {steps_per_sim}\n")

for algorithm in algorithms:
    if algorithm not in non_rl:
        algo_type = rl
        try:
            model = algorithms[algorithm]
            model = model.load(parameters_folder + algorithm)
        except FileNotFoundError:
            f.write(f"Algorithm not yet trained: {algorithm}\n")
            continue
    else:
        algo_type = algorithm
        model = algorithms[algorithm]

    # init or clear results
    results = {
        "null_action_ratio": 0,
        "min_steps": steps_per_sim,
        "max_steps": 0,
        "defender_wins": 0,
        "steps": [],
        "total_reward": []
    }

    print(f"Start simulating {algorithm}.")
    for i in range(simulations_count):
        if i % 20 == 0 and algorithm not in non_rl:
            print(f"  {i} simulations done.")

        # run 1 simulation (until attacker or defender wins)
        run_simulation(algo_type, model)

        update_results()

    evaluate_and_save_results(algorithm)

print(f"Actual finish:      {timestamp_to_datetime(time.time())}.")

f.write("*********************************************************************\n")
f.write(f"{best_avg_steps[1]} is the most effective algorithm with {best_avg_steps[0]} avg steps.\n")
f.write(f"{best_avg_reward[1]} is the most efficient algorithm with {best_avg_reward[0]} avg reward/step.\n")
f.close()


f = open(results_file, "r")
s = "".join(f.readlines())
print(s)
f.close()

if simulate_only_one:
    os.remove(results_file)
