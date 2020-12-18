# Setup Project
## requirements
- python version 3.8
  - everything tested with 3.8.5 and all python libraries in venv
- pip installs
```
pip install torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install stable_baselines3
pip install gym
```
## running given code 
download this repo (https://github.zhaw.ch/cailllev/PA_MTD/) and go to folder RL_MTD
  - run just once (default config includes ~5min of training and then simulates all algos
    - src.rl_mtd  => run can be configured at main call at line 447
  - run multiple times with different lengths of training sessions (1sec ... 4h):
    - src.runner => runs can be configured at main call at line 15 


default output of `` src.rl_mtd ``
```
Config:
Graph Name:                simple_webservice
Attack Name:               professional
Only Nodes:                False
Only Prevention Systems:   False
Nodes Pause:               1
Prevention Systems Pause:  1
*********************************************
Learning A2C, PPO algorithms.
*********************************************
Learning A2C.
No learned parameters found, start from scratch.
100000 steps to learn, with updates to policy each 50 steps.
Start:              Monday 13:31:11.
Estimated finish:   Monday 13:32:41.
...
Actual finish:      Monday 13:32:28.
*********************************************
Learning PPO.
No learned parameters found, start from scratch.
100000 steps to learn, with updates to policy each 50 steps.
Start:              Monday 13:32:28.
Estimated finish:   Monday 13:35:28.
...
Actual finish:      Monday 13:34:56.
*********************************************
Prepared to simulate A2C, PPO, Defender2000, Random, Static with 5 simulations.
Start:              Monday 13:34:56.
Estimated finish:   Monday 13:35:24.
...
Start simulating A2C.
5 simulations done.
Start simulating PPO.
5 simulations done.
Start simulating Defender2000.
5 simulations done.
Start simulating Random.
5 simulations done.
Start simulating Static.
5 simulations done.
Actual finish:      Monday 13:35:22.

Printing file content:

*********************************************************************
Starting Simulation Types: A2C, PPO, Defender2000, Random, Static
Simulations per Type:      5
Steps per Simulation:      5000
---------------------------------------------------------------------
Graph Name:                simple_webservice
Attack Name:               professional
Only Nodes:                False
Only Prevention Systems:   False
Nodes Pause:               1
Prevention Systems Pause:  1
*********************************************************************
Simulation Type:          A2C
Learning Time:            77.4 sec
---------------------------------------------------------------------
Avg steps:                5000.0
Avg reward/step:          52.602
Avg null action ratio:    [0.009, 0.0]
Avg invalid actions:      [0.0, 0.0]

Min steps:                5000
Max steps:                5000

Defender wins:            5
Attacker wins:            0

Steps each sim:           [5000, 5000, 5000, 5000, 5000]
Reward each sim:          [262704, 263129, 262702, 263206, 263299]
*********************************************************************
Simulation Type:          PPO
Learning Time:            147.5 sec
---------------------------------------------------------------------
Avg steps:                5000.0
Avg reward/step:          53.308
Avg null action ratio:    [0.175, 0.004]
Avg invalid actions:      [0.0, 0.0]

Min steps:                5000
Max steps:                5000

Defender wins:            5
Attacker wins:            0

Steps each sim:           [5000, 5000, 5000, 5000, 5000]
Reward each sim:          [267011, 266219, 266464, 266399, 266605]
*********************************************************************
Simulation Type:          Defender2000
Learning Time:            0 sec
---------------------------------------------------------------------
Avg steps:                150.2
Avg reward/step:          34.441
Avg null action ratio:    [0.083, 0.268]
Avg invalid actions:      [0.0, 0.0]

Min steps:                71
Max steps:                316

Defender wins:            0
Attacker wins:            5

Steps each sim:           [82, 71, 316, 126, 156]
Reward each sim:          [2415, 2035, 11524, 4549, 5342]
*********************************************************************
Simulation Type:          Random
Learning Time:            0 sec
---------------------------------------------------------------------
Avg steps:                80.4
Avg reward/step:          27.557
Avg null action ratio:    [0.129, 0.326]
Avg invalid actions:      [0.0, 0.0]

Min steps:                39
Max steps:                172

Defender wins:            0
Attacker wins:            5

Steps each sim:           [90, 59, 42, 39, 172]
Reward each sim:          [2585, 1292, 733, 1010, 5458]
*********************************************************************
Simulation Type:          Static
Learning Time:            0 sec
---------------------------------------------------------------------
Avg steps:                14.0
Avg reward/step:          -7.143
Avg null action ratio:    [1.0, 1.0]
Avg invalid actions:      [0.0, 0.0]

Min steps:                10
Max steps:                20

Defender wins:            0
Attacker wins:            5

Steps each sim:           [10, 14, 20, 13, 13]
Reward each sim:          [-300, -210, 250, -70, -170]
*********************************************************************
A2C is the most effective algorithm with 5000.0 avg steps.
PPO is the most efficient algorithm with 53.308 avg reward/step.


Process finished with exit code 0
```
## creating own network / attacks to train agents
1. open config/attack_graphs.json
2. under simple webservice one can configure:
    - nodes: what services run in their network (and how they are connected)
    - prevention systems: what prevention systems are in place
    - attacks: how likely an attacker gains control of such a service
    - hints
        - under "nodes" and "prevention systems" are the names of probabilities, in "attacks" are the names and the 
        corresponding values of probabilities
        - all names of probabilities in "nodes" and "prevention systems" have to be listed in "attacks", all probabilities in 
        "attacks" have to be used in "nodes" or "prevention systems"
3. to check if a configuration can be parsed, do:
    - run ``` src.model.graph ```
    - if no errors, the network can be parsed; but that does not guarantee the network makes sense
    - if there are errors (assertations fail), the way to fix the problem is described and it should be self self-explanatory
4. now run:
    - ``` src.rl_mtd.main(..., graph=<own_graph_name>, attack=<own_attack_name>, ...)```
