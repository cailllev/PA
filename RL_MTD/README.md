# Setup Project
## requirements
- python version
```
3.8
```
- pip installs
```
pip install torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install stable_baselines3
pip install gym
```
## running given code => download repo
- run just once (includes ~5min of training and then simulate all algos: ```A2C PPO Defender2000 Random Static```):
    - ``` src.rl_mtd ``` => run is configurable at main call at line 427 
- run multiple times with different lengths of training sessions (1sec ... 4h):
    - ``` src.runner ``` => runs are configurable at main call at line 15
## creating own network / attacks to train agents
1. open config/attack_graphs.json
2. under simple webservice one can configure:
    - nodes: what services run in their network (and how they are connected)
    - detection systems: what detection systems are in place
    - attacks: how likely an attacker gains control of such a service
    - hints
        - under nodes and detection systems are the name of probabilities, in attacks are the names and the 
        corresponding values of probabilities
        - all names of probabilities in nodes and detection systems have to be listed in attacks, all probabilities in 
        attacks have to be used in nodes or detection systems
3. to check if a configuration can be parsed, do:
    - run ``` src.model.graph ```
    - if no errors, the network can be parsed; but that does not guarenteed the network makes sense
    - if there are errors (assertations fail), the way to fix the problem should be self self-explanatory
4. now run:
    - ``` src.rl_mtd.main(..., graph=<own_graph_name>, attack=<own_attack_name>, ...)```
