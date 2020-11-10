from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.deepq.policies import DQNPolicy
from stable_baselines import A2C, ACKTR, PPO2  # , ACER, DQN

from src.env.mtd_env import MTDEnv, get_restartable_nodes_count, get_detection_systems_count
from src.defender2000 import Defender2000

# https://medium.com/@SmartLabAI/reinforcement-learning-algorithms-an-intuitive-overview-904e2dff5bbc#2bfa
#
# RL Algos
#   Model Free
#       Policy Optimization (ActorCriticRLModel)
#           A2C     Advantage Actor Critic
#           ***         - Advantage: Similarly to PG where the update rule used the dicounted returns from a set of
#                       experiences in order to tell the agnet which acttions were “good” or “bad”.
#                       - Actor-critic: combines the benefits of both approaches from policy-iteration method as PG and
#                       value-iteration method as Q-learning (See below). The network will estimate both a value
#                       function V(s) (how good a certain state is to be in) and a policy π(s).
#
#           ACER    Actor-Critic with Experience Replay
#                       ------------------------------------------------------------------------------------------------
#                       -> ValueError: ACER does not work with MultiDiscrete([7 2]) actions space
#                       maybe fixable with just one Discrete action space, just append all actions in one list?
#                       ------------------------------------------------------------------------------------------------
#                       <no ducomentation for ACER>
#
#           ACKTR   Actor Critic using Kronecker-Factored Trust Region
#           *****       <no ducomentation for ACTKR>
#
#           PP01    Proximal Policy Optimization algorithm (MPI version)
#                       ------------------------------------------------------------------------------------------------
#                       -> ImportError: cannot import name 'PPO1' from 'stable_baselines'
#                       unfixable?
#                       ------------------------------------------------------------------------------------------------
#                       Also an on-policy algorithm which similarly to TRPO can perform on discrete or continuous action
#                       spaces. PPO shares motivation with TRPO in the task of answering the question: how to increase
#                       policy improvement without the risk of performance collapse? The idea is that PPO improves the
#                       stability of the Actor training by limiting the policy update at each training step.
#
#           PP02    Proximal Policy Optimization algorithm (GPU version)
#           ****        <see above>
#
#           TRPO    Trust Region Policy Optimization
#                       ------------------------------------------------------------------------------------------------
#                       -> ImportError: cannot import name 'TRPO' from 'stable_baselines'
#                       unfixable?
#                       ------------------------------------------------------------------------------------------------
#                       A on-policy algorithm that can be used or environments with either discrete or continuous action
#                       spaces. TRPO updates policies by taking the largest step possible to improve performance, while
#                       satisfying a special constraint on how close the new and old policies are allowed to be.
#
#       Q-Learning
#           DQN     Deep Q Neural Network (OffPolicyRLModel)
#                       ------------------------------------------------------------------------------------------------
#                       -> AssertionError: Error: the action space for DQN must be of type gym.spaces.Discrete
#                       maybe fixable with just one Discrete action space, just append all actions in one list?
#                       ------------------------------------------------------------------------------------------------
#                       DQN is Q-learning with Neural Networks . The motivation behind is simply related to big state
#                       space environments where defining a Q-table would be a very complex, challenging and time-
#                       consuming task. Instead of a Q-table Neural Networks approximate Q-values for each action based
#                       on the state.
#
#           HER     Hindsight Experience Replay (BaseRLModel)
#                       ------------------------------------------------------------------------------------------------
#                       -> AttributeError: 'Discrete' object has no attribute 'spaces'
#                       HER works with continous action space, presumably very hard or even impossible to create a
#                       useful translation from discrete to continous?
#                       ------------------------------------------------------------------------------------------------
#                       In Hindsight Experience Replay method, basically a DQN is suplied with a state and a desired
#                       end-state, or in other words goal. It allow to quickly learn when the rewards are sparse. In
#                       other words when the rewards are uniform for most of the time, with only a few rare reward-
#                       values that really stand out.
#
#       Hybrid (OffPolicyRLModel)
#           DDPG    Deep Deterministic Policy Gradient
#                       ------------------------------------------------------------------------------------------------
#                       -> ImportError: cannot import name 'DDPG' from 'stable_baselines'
#                       unfixable?
#                       ------------------------------------------------------------------------------------------------
#                       https://arxiv.org/pdf/1802.09477.pdf
#                       [...] Our algorithm builds on Double Q-learning, by taking the minimum value between a pair of
#                       critics to limit overestimation. We draw the connection between target networks and
#                       overestimation bias, and suggest delaying policy updates to reduce per-update error and further
#                       improve performance. [...]
#
#           SAC     Soft Actor Critic
#                       ------------------------------------------------------------------------------------------------
#                       SAC only supports Box as action space (binary)
#                       https://stable-baselines.readthedocs.io/en/master/modules/sac.html#can-i-use
#
#                       plus the SAC object cannot be instantiated, probably unfixable (see below)
#                       1)    File "../RL_MTD/src/rl_mtd.py", line 133, in <module>
#                               "SAC": SAC(MlpPolicy, env, verbose=0, n_env=1, n_steps=simulations_count, n_batch=1),
#                           TypeError: __init__() got an unexpected keyword argument 'n_env'
#
#                       2)    File "../PA_MTD/RL_MTD/src/rl_mtd.py", line 134, in <module>
#                               "SAC": SAC(MlpPolicy, env, verbose=0),
#                             File "..\Python37\lib\site-packages\stable_baselines\sac\sac.py", line 124, in __init__
#                               self.setup_model()
#                             File "..\Python37\lib\site-packages\stable_baselines\sac\sac.py", line 144, in setup_model
#                               **self.policy_kwargs)
#                           TypeError: __init__() missing 3 required positional arguments: 'n_env', 'n_steps', and
#                           'n_batch'
#                       ------------------------------------------------------------------------------------------------
#                       https://arxiv.org/abs/1801.01290
#                       [...]. By combining off-policy updates with a stable stochastic actor-critic formulation, our
#                       method achieves state-of-the-art performance on a range of continuous control benchmark tasks,
#                       outperforming prior on-policy and off-policy methods. [...]
#
#           TD3     Twin Delayed DDPG
#                       ------------------------------------------------------------------------------------------------
#                       exactly the same error as in SAC, also probably unfixable
#                       ------------------------------------------------------------------------------------------------
#                       https://arxiv.org/pdf/1509.02971.pdf
#                       We adapt the ideas underlying the success of Deep Q-Learning to the continuous
#                       action domain. We present an actor-critic, model-free algorithm based on the deterministic
#                       policy gradient that can operate over continuous action spaces. [...]
#
#   Model Based
#       * no support from stable_baselines plus not usefull because agent cannot predict the next state most of the time


rl = "RL"
defender2000 = "Defender2000"
random = "Random"
static = "Static"

non_rl = [defender2000, random, static]

# ------------------------- config ------------------------- #
learn = False
timesteps = 10 ** 6

simulate_only_best = False
simulations_count = 100

env = MTDEnv()

# only to learn, Random and Static are added later
algorithms = {
    "A2C": A2C(MlpPolicy, env, verbose=0, n_steps=simulations_count),
    "ACKTR": ACKTR(MlpPolicy, env, verbose=0, n_steps=simulations_count),
    "PPO2": PPO2(MlpPolicy, env, verbose=0, n_steps=simulations_count),
    # "DQN": DQN(DQNPolicy, env, verbose=0),
    # "HER": HER(MlpPolicy, env, verbose=0, n_steps=simulations_count, model_class=DQN, n_sampled_goal=0),
    # "SAC": SAC(MlpPolicy, env, verbose=0),
    # "TD3": TD3(MlpPolicy, env, verbose=0)
}

parameters_folder = "parameters/v2/"
results_file = f"{parameters_folder}results_{simulations_count}_simulations.txt"
extensive_results_file = f"{parameters_folder}extensive_results_{simulations_count}_simulations.txt"
# TODO, write all steps (per simulation), write null actions percentage for better evaluation


# ------------------------ learning ------------------------ #
if learn:
    print("*********************************************")
    print(f"Learning {len(algorithms)} algorithms.")
    for algorithm in algorithms:

        model = algorithms[algorithm]

        print("*********************************************")
        print(f"Learning {algorithm}.")

        try:
            model.load(parameters_folder + algorithm)
            print(f"Found parameters.")

        except ValueError:
            print(f"No learned parameters found, start from scratch.")

        print(f"{timesteps} steps to simulate.")
        print(f"Estimated time: {timesteps / 10 ** 5} min.")
        print("...")

        import time

        start = time.time()

        model.learn(total_timesteps=timesteps)
        model.save(parameters_folder + algorithm)

        print(f"{round(time.time() - start)} seconds to simulate {timesteps} steps.")

# ------------------------ simulation helpers ------------------------ #
best_avg_steps = [0, random]
best_avg_reward = [0, random]
show_model_after_each_step = False
show_results_after_each_sim = False


def get_best_algorithm():
    # type: () -> str
    """
    get the best rl algorithm
    :return:
    """
    try:
        file = open(results_file)
        last_line = None
        for line in file:
            last_line = line
        file.close()

        return last_line.split(" ")[0]

    except FileNotFoundError:
        return random


if simulate_only_best:
    # delete all algorihms except the best from dict
    best_algo = get_best_algorithm()
    for algo in algorithms:
        if algorithms[algo] == best_algo:
            val = algorithms[algo]
            algorithms.clear()
            algorithms[algo] = val
            break

# add non rl algorithms (own, random and static) for evaluation
algorithms[defender2000] = Defender2000(get_restartable_nodes_count(), get_detection_systems_count())
algorithms[random] = random
algorithms[static] = static

results = dict.fromkeys(algorithms)
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
#   "Defender2000": {...},
#   "Random": {...},
#   "Static": {...}
# }

print("*********************************************")
print(f"Prepared to simulate {', '.join(algorithms.keys())} with {simulations_count} simulations.")
print(f"Estimated time to simulate all algoritmns: {simulations_count * (len(algorithms.keys()) - 3) / 300} min")
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
    global best_avg_reward

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

    if avg_reward > best_avg_reward[0]:
        best_avg_reward = [avg_reward, algo_name]

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
    f.write(f"Avg reward/step:  {avg_reward}\n")
    f.write(f"Min steps:        {min_steps}\n")
    f.write(f"Max steps:        {max_steps}\n\n")
    f.write(f"Defender wins:    {defender_wins}\n")
    f.write(f"Attacker wins:    {attacker_wins}\n")


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


# ------------------------ actual simulation ------------------------ #
f = open(results_file, "w")
f.write("*********************************************************************\n")
f.write(f"Starting Simulation Types: {', '.join(algorithms.keys())}\n")
f.write(f"Simulations per Type:      {simulations_count}\n")
f.write(f"Steps per Simulation:      {env.get_steps_per_simulation()}\n")

for algorithm in algorithms:
    if algorithm not in non_rl:
        algo_type = rl
        try:
            model = algorithms[algorithm]
            model.load(parameters_folder + algorithm)
        except ValueError:
            f.write(f"{algorithm} not trained yet, skipping to next algo\n")
            continue
    else:
        algo_type = algorithm
        model = algorithms[algorithm]

    results[algorithm] = {}
    results[algorithm]["steps"] = []
    results[algorithm]["min_steps"] = env.get_steps_per_simulation()
    results[algorithm]["max_steps"] = 0
    results[algorithm]["total_reward"] = []
    results[algorithm]["defender_wins"] = 0

    for _ in range(simulations_count):
        # run 1 simulation (until attacker or defender wins)
        run_simulation(algo_type, model)

        update_results(algorithm)

    evaluate_and_save_results(algorithm)

f.write("*********************************************************************\n")
f.write(f"{best_avg_steps[1]} is the most effective algorithm with {best_avg_steps[0]} avg steps.\n")
f.write(f"{best_avg_reward[1]} is the most efficient algorithm with {best_avg_reward[0]} avg reward/step.\n")
f.close()


f = open(results_file, "r")
s = "".join(f.readlines())
print(s)
f.close()
