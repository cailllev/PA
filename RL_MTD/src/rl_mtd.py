from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from .env.mtd_env import MTDEnv

env = DummyVecEnv([lambda: MTDEnv()])

# model = PPO2(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=20000)

obs = env.reset()
for i in range(2000):
    # action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(env.action_space.sample())
    env.render()
