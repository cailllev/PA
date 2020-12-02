# Setup Project
## requirements
- python version 3.8
  - everything tested with 3.8.5
- pip installs
```
pip install torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install stable_baselines3
pip install gym
```
## running given code 
1. download this repo (https://github.zhaw.ch/cailllev/PA_MTD/) and go to folder RL_MTD
2. possible runs...
    - run just once (default config includes ~5min of training and then simulates all algos: ```A2C PPO Defender2000 Random Static```:
        - `` src.rl_mtd `` => run can be configured at main call at line 447 
    - run multiple times with different lengths of training sessions (1sec ... 4h):
        - `` src.runner `` => runs can be configured at main call at line 15
    - default output of `` src.rl_mtd ``
```
Config:
Graph Name:                simple_webservice
Attack Name:               professional
Only Nodes:                False
Only Detection Systems:    False
Nodes Pause:               1
Detection Systems Pause:   1
*********************************************
Learning A2C, PPO algorithms.
*********************************************
Learning A2C.
No learned parameters found, start from scratch.
10000 steps to learn, with updates to policy each 50 steps.
Start:              05:06:25.
Estimated finish:   05:06:34.
...
Actual finish:      05:06:40.
*********************************************
Learning PPO.
No learned parameters found, start from scratch.
10000 steps to learn, with updates to policy each 50 steps.
Start:              05:06:40.
Estimated finish:   05:06:58.
...
Actual finish:      05:07:06.
*********************************************
Prepared to simulate A2C, PPO, Defender2000, Random, Static with 20 simulations.
Start:              05:07:06.
Estimated finish:   05:08:58.
...
Start simulating A2C.
20 simulations done.
Start simulating PPO.
20 simulations done.
Start simulating Defender2000.
20 simulations done.
Start simulating Random.
20 simulations done.
Start simulating Static.
20 simulations done.
Actual finish:      05:08:32.

Printing file content:

*********************************************************************
Starting Simulation Types: A2C, PPO, Defender2000, Random, Static
Simulations per Type:      20
Steps per Simulation:      5000
---------------------------------------------------------------------
Graph Name:                simple_webservice
Attack Name:               professional
Only Nodes:                False
Only Detection Systems:    False
Nodes Pause:               1
Detection Systems Pause:   1
*********************************************************************
Simulation Type:          A2C
Learning Time:            15.4 sec
---------------------------------------------------------------------
Avg steps:                1094.1
Avg reward/step:          42.824
Avg null action ratio:    [0.162, 0.179]
Avg invalid actions:      [0.0, 0.0]

Min steps:                146
Max steps:                3282

Defender wins:            0
Attacker wins:            20

Steps each sim:           [3282, 579, 1066, 198, 3120, 1226, 2575, 154, 434, 593, 1031, 695, 873, 706, 1260, 198, 973, 328, 146, 2445]
Reward each sim:          [143687, 24906, 45810, 8050, 135131, 53874, 110028, 5901, 17087, 24581, 43360, 28869, 37063, 30323, 52747, 8508, 42120, 12729, 5700, 106604]
*********************************************************************
Simulation Type:          PPO
Learning Time:            26.2 sec
---------------------------------------------------------------------
Avg steps:                3394.65
Avg reward/step:          48.27
Avg null action ratio:    [0.133, 0.096]
Avg invalid actions:      [0.0, 0.0]

Min steps:                24
Max steps:                5000

Defender wins:            8
Attacker wins:            12

Steps each sim:           [1811, 5000, 137, 5000, 5000, 4691, 771, 4254, 3128, 4482, 24, 1562, 5000, 5000, 3266, 5000, 5000, 5000, 3432, 335]
Reward each sim:          [86887, 242894, 5944, 241324, 240777, 224464, 36983, 204026, 150812, 214845, 593, 74743, 241922, 244354, 157064, 242941, 243078, 241231, 166343, 15946]
*********************************************************************
Simulation Type:          Defender2000
Learning Time:            0 sec
---------------------------------------------------------------------
Avg steps:                169.05
Avg reward/step:          34.85
Avg null action ratio:    [0.103, 0.288]
Avg invalid actions:      [0.0, 0.0]

Min steps:                12
Max steps:                818

Defender wins:            0
Attacker wins:            20

Steps each sim:           [256, 14, 80, 89, 15, 141, 345, 360, 182, 66, 818, 140, 30, 161, 16, 114, 155, 250, 12, 137]
Reward each sim:          [8913, 4, 2540, 2821, 3, 4567, 12388, 13402, 6275, 1694, 30738, 4980, 759, 5690, 170, 3731, 5481, 8967, 3, 4703]
*********************************************************************
Simulation Type:          Random
Learning Time:            0 sec
---------------------------------------------------------------------
Avg steps:                104.55
Avg reward/step:          28.431
Avg null action ratio:    [0.131, 0.336]
Avg invalid actions:      [0.0, 0.0]

Min steps:                7
Max steps:                264

Defender wins:            0
Attacker wins:            20

Steps each sim:           [128, 96, 9, 163, 74, 114, 185, 41, 93, 53, 197, 30, 92, 216, 71, 7, 136, 264, 54, 68]
Reward each sim:          [4059, 2518, -210, 5184, 1853, 3436, 6238, 736, 2719, 1181, 5845, 579, 1901, 7341, 1726, -271, 3741, 7692, 1047, 2134]
*********************************************************************
Simulation Type:          Static
Learning Time:            0 sec
---------------------------------------------------------------------
Avg steps:                21.5
Avg reward/step:          6.395
Avg null action ratio:    [1.0, 1.0]
Avg invalid actions:      [0.0, 0.0]

Min steps:                5
Max steps:                46

Defender wins:            0
Attacker wins:            20

Steps each sim:           [25, 8, 13, 18, 19, 25, 8, 29, 43, 10, 46, 19, 26, 19, 29, 9, 12, 5, 38, 29]
Reward each sim:          [250, -220, -220, -170, 40, 300, -170, 240, 480, -200, 810, 240, 160, 290, 340, -210, 20, -300, 780, 290]
*********************************************************************
PPO is the most effective algorithm with 3394.65 avg steps.
PPO is the most efficient algorithm with 48.27 avg reward/step.


Process finished with exit code 0
```
## creating own network / attacks to train agents
1. open config/attack_graphs.json
2. under simple webservice one can configure:
    - nodes: what services run in their network (and how they are connected)
    - detection systems: what detection systems are in place
    - attacks: how likely an attacker gains control of such a service
    - hints
        - under "nodes" and "detection systems" are the names of probabilities, in "attacks" are the names and the 
        corresponding values of probabilities
        - all names of probabilities in "nodes" and "detection systems" have to be listed in "attacks", all probabilities in 
        "attacks" have to be used in "nodes" or "detection systems"
3. to check if a configuration can be parsed, do:
    - run ``` src.model.graph ```
    - if no errors, the network can be parsed; but that does not guarantee the network makes sense
    - if there are errors (assertations fail), the way to fix the problem is described and it should be self self-explanatory
4. now run:
    - ``` src.rl_mtd.main(..., graph=<own_graph_name>, attack=<own_attack_name>, ...)```
