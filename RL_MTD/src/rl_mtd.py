from src.env.mtd_env import MTDEnv

env = MTDEnv()

obs = env.reset()
env.render()
done = False
while env.get_counter() < 1000 and not done:
    obs, rewards, done, info = env.step(env.action_space.sample())
    env.render()
print(env.progress)
