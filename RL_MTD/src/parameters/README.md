In this folder are the parameters for the RL agents. <br>
The different versions (v1, v2, ...) describe the progress and major refactorings of the mtd_env or RL methodology. <br>
Below are the summaries for each version: <p>

# v1
- first experiences with RL agents
- simulated RL methods: A2C, ACKTR, PP02
- 10^7 simulated steps to learn, stil all RL agents are only marginally better than random
- although without switching of IDS (BUG), random is way better than static
  - random: avg steps 53, avg reward -23; static: avg steps 4.1, avg reward: -50

# v2
- major bug fix, switch IDS was not possible
- add bias per step to encurrage long simulations (+30)
- new defense method added (defender2000), trying to find a good policy / rules by hand 
  - => avg steps: 135 and avg reward: 16.0 ==> quite bad algorithm
- now all agents (A2C, ACKTR, PP02 and even random) achive vicory over attacker most of the time
- Random (27.2) is slightly less effective than A2C (27.4) and ACKTR (28.9), PP02 is very effective with avg reward of 
    32.8. Just lucky? 10 and 100 are more like the others

# v3 TODO
- refactored action space from multidiscrete to discrete (still same functionality) to add RL methods ACER and DQN