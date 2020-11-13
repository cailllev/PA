In this folder are the parameters for the RL agents. <br>
The different versions (v1, v2, ...) describe the progress and major refactorings of the mtd_env or RL methodology. <br>
Below are the summaries for each version: <p>

# v1
- first experiences with RL agents
- simulated RL methods: A2C, ACKTR, PP02
- 10^7 simulated steps to learn, still all RL agents are only marginally better than random
- although without switching of IDS (BUG, at the time undetected), random is way better than static
  - random: avg steps 53, avg reward -23; static: avg steps 4.1, avg reward: -50
```
rewards
  "switch_detection_system": -2
  "restart_node": -5
  "progression": -10
  "attacker_wins": -100

  "in_honeypot": 1
  "caught_attacker": 5
  "defender_wins": 100
```

# v2
- major bug fix, switch IDS was not possible
- add bias per step to encurrage long simulations (+30)
- new defense method added (defender2000), trying to find a good policy / rules by hand 
  - => avg steps: 135 and avg reward: 16.0 ==> quite bad algorithm
- now all agents (A2C, ACKTR, PP02 and even random) achive vicory over attacker most of the time
- Random (27.2) is slightly less effective than A2C (27.4) and ACKTR (28.9), PP02 is very effective with avg reward of 
    32.8. Just lucky? 10 and 100 are more like the others
```
rewards
  "switch_detection_system": -2
  "restart_node": -5
  "progression": -10
  "attacker_wins": -100

  "in_honeypot": 1
  "caught_attacker": 5
  "defender_wins": 100

  "bias_per_step": 30
```

# v3
- rework to implement stable_baselines3 for RL agents (PyTorch 1.4+)
  - openai gym has no special requirements
- refactored action space from multidiscrete to discrete (still same functionality) to add RL method DQN
  - refactored back after seeing DQN performs very bad
- remove bug in mtd_env, reset env did not reset graph -> fixed
```
rewards
  "switch_detection_system": -2,
  "restart_node": -5,
  "progression": -20,
  "attacker_wins": -40,

  "in_honeypot": 10  # to compensate for removing "progression: -1" of honeypot
  "caught_attacker": 5
  "defender_wins": 100

  "bias_per_step": 5
```