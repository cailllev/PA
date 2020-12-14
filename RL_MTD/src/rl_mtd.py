from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import A2C, PPO

from src.env.mtd_env import MTDEnv, subtract_one, create_locked_lists, choose_random_from_list, set_config
from src.defender2000 import Defender2000

from datetime import datetime
from sys import stdout
import time
import os
import math

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
#                       - Advantage: Similarly to PG where the update rule used the discounted returns from a set of
#                       experiences in order to tell the agent which actions were “good” or “bad”.
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
#                       Too much overhead to correctly and reasonably implement DQN.
#                       Plus DQN is even worse than Defender2000 after initial tests -> not worth investigating
#                       ------------------------------------------------------------------------------------------------
#                       DQN is Q-learning with Neural Networks . The motivation behind is simply related to big state
#                       space environments where defining a Q-table would be a very complex, challenging and time-
#                       consuming task. Instead of a Q-table Neural Networks approximate Q-values for each action based
#                       on the state.
#
#           HER     Hindsight Experience Replay (BaseRLModel)
#                       ------------------------------------------------------------------------------------------------
#                       Too many challenges to implement HER, unfixable with current setup. Initial list of requirements
#                           - cannot use normal env, needs VecEnv (but not DummyVecEnv, doesnt work), so we are left
#                               with SubProcEnv or own VecEnv implementation
#                           - cannot create env in SubProcEnv, multiprocessing exception
#                           - own VecEnv implementation not reasonable
#                           - cannot use MultiDiscrete
#                       ------------------------------------------------------------------------------------------------
#                       In Hindsight Experience Replay method, basically a DQN is supplied with a state and a desired
#                       end-state, or in other words goal. It allow to quickly learn when the rewards are sparse. In
#                       other words when the rewards are uniform for most of the time, with only a few rare reward-
#                       values that really stand out.
#
#       Hybrid (OffPolicyRLModel)
#           DDPG    Deep Deterministic Policy Gradient
#                       ------------------------------------------------------------------------------------------------
#                       only Box space supported, not fixable (and probably not reasonable to use)!
#                       ------------------------------------------------------------------------------------------------
#                       https://arxiv.org/pdf/1802.09477.pdf
#                       [...] Our algorithm builds on Double Q-learning, by taking the minimum value between a pair of
#                       critics to limit overestimation. We draw the connection between target networks and
#                       overestimation bias, and suggest delaying policy updates to reduce per-update error and further
#                       improve performance. [...]
#
#           SAC     Soft Actor Critic
#                       ------------------------------------------------------------------------------------------------
#                       only Box space supported, not fixable (and probably not reasonable to use)!
#                       ------------------------------------------------------------------------------------------------
#                       https://arxiv.org/abs/1801.01290
#                       [...]. By combining off-policy updates with a stable stochastic actor-critic formulation, our
#                       method achieves state-of-the-art performance on a range of continuous control benchmark tasks,
#                       outperforming prior on-policy and off-policy methods. [...]
#
#           TD3     Twin Delayed DDPG
#                       ------------------------------------------------------------------------------------------------
#                       only Box space supported, not fixable (and probably not reasonable to use)!
#                       ------------------------------------------------------------------------------------------------
#                       https://arxiv.org/pdf/1509.02971.pdf
#                       We adapt the ideas underlying the success of Deep Q-Learning to the continuous
#                       action domain. We present an actor-critic, model-free algorithm based on the deterministic
#                       policy gradient that can operate over continuous action spaces. [...]
#
#   Model Based
#       * no support from stable_baselines plus not useful because agent cannot predict the next state most of the time


def main(parameters_folder="parameters/tests", learn=True, learn_steps=10**4, simulations_count=20,
         simulate_only_one=False, graph="simple_webservice", attack="professional",
         only_nodes=False, only_prevention_systems=False, nodes_pause=1, prevention_systems_pause=1):
    # type: (str, bool, int, int, bool, str, str, bool, bool, int, int) -> None
    """
    simulates (and trains) rl agents and own agents (defender2000, random and static) on MTDEnv and writes the results 
    and trained parameters to {{parameters_folder}}
    :param parameters_folder: where the parameters are saved to and loaded from
    :param learn: skip learning?
    :param learn_steps: steps for learning
    :param simulations_count: how many individual simulations per algorithm
    :param simulate_only_one: False or None -> simulate all; A2C -> simulate only A2C and write results to temp file
    :param graph: the name of the graph as in config/attack_graphs.json
    :param attack: the name of the attack as in config/attack_graphs.json[graph][attacks]
    :param only_nodes: only able to restart nodes, prevention systems are fixed
    :param only_prevention_systems: only able to switch prevention systems, nodes are fixed
    :param nodes_pause: pause between same node restarts (1=every step possible)
    :param prevention_systems_pause: pause between same prevention system switches (1=every step possible)
    """

    print("\nConfig:")
    print(f"Graph Name:                {graph}")
    print(f"Attack Name:               {attack}")
    print(f"Only Nodes:                {only_nodes}")
    print(f"Only Prevention Systems:   {only_prevention_systems}")
    print(f"Nodes Pause:               {nodes_pause}")
    print(f"Prevention Systems Pause:  {prevention_systems_pause}")

    # ------------------------- const ------------------------- #
    learn_steps_exponent = round(math.log10(learn_steps))
    parameters_folder += f"1e{learn_steps_exponent}_training/"

    rl = "RL"
    defender2000 = "Defender2000"
    random = "Random"
    static = "Static"

    non_rl = [defender2000, random, static]

    # ------------------------- debug ------------------------- #
    show_model_after_each_step = False
    show_results_after_each_sim = False

    # ------------------------- init ------------------------- #
    if not os.path.isdir(parameters_folder):
        os.mkdir(parameters_folder)

    set_config(graph, attack)
    env = MTDEnv(only_nodes, only_prevention_systems, nodes_pause, prevention_systems_pause)

    # only to learn; Defender2000, Random and Static are added later
    # after 50 steps the agent updates it policy (it learns each 50 steps)
    algorithms = {
        "A2C": A2C(ActorCriticPolicy, env, n_steps=50, verbose=0),
        "PPO": PPO(ActorCriticPolicy, env, n_steps=50, verbose=0),
    }

    learning_time = {
        "A2C": "unknown",
        "PPO": "unknown",
        defender2000: "0 sec",
        random: "0 sec",
        static: "0 sec"
    }

    # times estimations
    base_learn_time_estimate = (learn_steps / 10 ** 3)
    algorithm_learn_times = {
        "A2C": base_learn_time_estimate * 0.9,
        "PPO": base_learn_time_estimate * 1.8
    }
    steps_per_sim = env.get_steps_per_simulation()
    sim_time_estimate = simulations_count * steps_per_sim / 900
    if simulate_only_one:
        results_file = "temp.txt"
        if simulate_only_one in non_rl:
            sim_time_estimate /= 200
        else:
            sim_time_estimate /= 2
    else:
        results_file = f"{parameters_folder}results_{simulations_count}_simulations.txt"

    def timestamp_to_datetime(stamp):
        # type: (float) -> str
        return datetime.fromtimestamp(stamp).strftime('%A %H:%M:%S')

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

            print(f"{learn_steps} steps to learn, with updates to policy each {model.n_steps} steps.")
            print(f"Start:              {timestamp_to_datetime(start)}.")
            print(f"Estimated finish:   {timestamp_to_datetime(start+algorithm_learn_times[algorithm])}.")
            print("...")

            model.learn(total_timesteps=learn_steps)
            end = time.time()

            model.save(parameters_folder + algorithm)

            learning_time[algorithm] = f"{round(end - start, 1)} sec"

            print(f"Actual finish:      {timestamp_to_datetime(end)}.")

    # ------------------------ simulation helpers ------------------------ #
    best_avg_steps = [0, random]
    best_avg_reward = [-1000, random]

    # add non rl algorithms (random, defender2000 and static) for evaluation
    algorithms[defender2000] = Defender2000(only_nodes, only_prevention_systems, nodes_pause, prevention_systems_pause)
    algorithms[random] = random
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

    def run_simulation(m):
        obs = env.reset()
        done = False

        while not done:
            action, _ = m.predict(obs)

            # do action on model
            obs, _, done, _ = env.step(action)
            if show_model_after_each_step:
                env.render()

    def run_random_simulation():
        env.reset()
        done = False
        locked_nodes, locked_prevention_systems = create_locked_lists(*env.action_space.nvec)

        while not done:
            subtract_one(locked_nodes)
            subtract_one(locked_prevention_systems)

            # create new action until value in locked list is 0 and 0, i.e. no pause
            action = [0, 0]

            if not only_prevention_systems:
                action[0] = choose_random_from_list(locked_nodes, nodes_pause)
            if not only_nodes:
                action[1] = choose_random_from_list(locked_prevention_systems, prevention_systems_pause)

            # do action on model
            obs, _, done, _ = env.step(action)
            if show_model_after_each_step:
                env.render()

    def run_static_simulation():
        env.reset()
        done = False
        null_action = [0, 0]

        while not done:
            # do action on model
            obs, _, done, _ = env.step(null_action)
            if show_model_after_each_step:
                env.render()

    def update_results():
        """
        retrieves all information from the underlying env, parses it and saves it to "global" results dict
        """
        nonlocal results
        results["steps"].append(env.get_counter())

        reward = env.get_total_reward()
        penalty = env.get_invalid_actions_penalty()
        results["total_reward"].append(reward - penalty)

        null_actions_count = env.get_null_actions_count()
        results["null_actions_count"][0] += null_actions_count[0]
        results["null_actions_count"][1] += null_actions_count[1]

        invalid_actions_count = env.get_invalid_actions_count()
        results["invalid_actions_count"][0] += invalid_actions_count[0]
        results["invalid_actions_count"][1] += invalid_actions_count[1]

        if env.defender_wins():
            results["defender_wins"] += 1

        if env.get_counter() < results["min_steps"]:
            results["min_steps"] = env.get_counter()

        if env.get_counter() > results["max_steps"]:
            results["max_steps"] = env.get_counter()

        if show_results_after_each_sim:
            print("\n" + str(env))

    def evaluate_and_save_results(algo_name):
        """
        reads the "global" results dict, parses it's data and writes relevant info for later evaluation to the results
        file
        :param algo_name: the name of the algorithm (A2C, PPO, defender2000, ...)
        """
        nonlocal best_avg_steps
        nonlocal best_avg_reward

        steps = results["steps"]
        rewards = results["total_reward"]

        sum_steps = sum(steps)
        sum_rewards = sum(rewards)

        avg_steps = round(sum_steps / simulations_count, 3)
        avg_reward = round(sum_rewards / sum_steps, 3)

        avg_null_action_ratio = [0, 0]
        avg_null_action_ratio[0] = round(results["null_actions_count"][0] / sum_steps, 3)
        avg_null_action_ratio[1] = round(results["null_actions_count"][1] / sum_steps, 3)

        avg_invalid_actions = [0, 0]
        avg_invalid_actions[0] = round(results["invalid_actions_count"][0] / sum_steps, 3)
        avg_invalid_actions[1] = round(results["invalid_actions_count"][1] / sum_steps, 3)

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
        f.write(f"Learning Time:            {learning_time[algo_name]}\n")
        f.write("---------------------------------------------------------------------\n")
        f.write(f"Avg steps:                {avg_steps}\n")
        f.write(f"Avg reward/step:          {avg_reward}\n")
        f.write(f"Avg null action ratio:    {avg_null_action_ratio}\n")
        f.write(f"Avg invalid actions:      {avg_invalid_actions}\n\n")
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
    f.write("---------------------------------------------------------------------\n")
    f.write(f"Graph Name:                {graph}\n")
    f.write(f"Attack Name:               {attack}\n")
    f.write(f"Only Nodes:                {only_nodes}\n")
    f.write(f"Only Prevention Systems:   {only_prevention_systems}\n")
    f.write(f"Nodes Pause:               {nodes_pause}\n")
    f.write(f"Prevention Systems Pause:  {prevention_systems_pause}\n")

    for algorithm, model in algorithms.items():
        if algorithm not in non_rl:
            algo_type = rl
            try:
                model = model.load(parameters_folder + algorithm)
            except FileNotFoundError:
                f.write("---------------------------------------------------------------------\n")
                f.write(f"Algorithm not yet trained: {algorithm}\n")
                continue
        else:
            algo_type = algorithm

        # init or clear results
        results = {
            "null_actions_count": [0, 0],
            "invalid_actions_count": [0, 0],
            "min_steps": steps_per_sim,
            "max_steps": 0,
            "defender_wins": 0,
            "steps": [],
            "total_reward": []
        }

        print(f"Start simulating {algorithm}.")
        for i in range(simulations_count):
            # run 1 complete simulation (until attacker or defender wins)
            if algo_type in [rl, defender2000]:
                run_simulation(model)

            elif algo_type is random:
                run_random_simulation()

            elif algo_type is static:
                run_static_simulation()

            else:
                raise Exception(f"Unknown algorithm type: {algo_type}.")

            update_results()
            stdout.write("\r%d simulations done." % (i+1))
            stdout.flush()

        evaluate_and_save_results(algorithm)
        stdout.write("\n")
        stdout.flush()

    print(f"Actual finish:      {timestamp_to_datetime(time.time())}.")

    f.write("*********************************************************************\n")
    f.write(f"{best_avg_steps[1]} is the most effective algorithm with {best_avg_steps[0]} avg steps.\n")
    f.write(f"{best_avg_reward[1]} is the most efficient algorithm with {best_avg_reward[0]} avg reward/step.\n")
    f.close()

    # ------------------------- finalize ------------------------- #
    print("\nPrinting file content:\n")
    with open(results_file, "r") as f:
        s = "".join(f.readlines())
        print(s)

    if simulate_only_one:
        os.remove(results_file)


if __name__ == "__main__":
    main("parameters/tests/",
         learn=False,
         learn_steps=10 ** 5,
         graph="simple_webservice",
         attack="professional",
         simulations_count=5,
         simulate_only_one="Defender2000",
         only_nodes=False,
         only_prevention_systems=False,
         nodes_pause=1,
         prevention_systems_pause=1)
