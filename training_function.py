import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# from IPython.display import clear_output
import os

class VecPendulumRewardWrapper(VecEnvWrapper):
    def __init__(self, venv: VecEnv):
        super().__init__(venv=venv)

    def reset(self) -> np.ndarray:
        obs = self.venv.reset()
        return obs
    
    def step_async(self, actions: np.ndarray) -> None:
        self.venv.step_async(actions)

    def step_wait(self) -> VecEnvStepReturn:
        obs, reward, done, info = self.venv.step_wait()
        reward = 1- obs[:, 0]**2 - obs[:, 1]**2  ######## MODIFY FOR REWARD FUNCTION
        return obs, reward, done, info
    
def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        return env
    return _init

def TrainSAC(model_name: str, learning_rate=0.0003, buffer_size=100000, learning_starts=100, batch_size=256, tau=0.005, gamma=0.99, total_timesteps=1e5):
    env_id = 'InvertedPendulum-v4'  # Replace with your MuJoCo environment
    num_envs = 16  # Number of parallel environments

    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_envs)])
    print(env.seed())
    # Add the reward wrapper
    env = VecPendulumRewardWrapper(env)
    env = VecMonitor(env)  # Optional: for monitoring and logging

    # Check if the folder exists, if not, create it
    save_path = "./trainings"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = SAC('MlpPolicy', env, learning_rate=learning_rate, buffer_size=buffer_size, learning_starts=learning_starts, batch_size=batch_size, tau=tau, gamma=gamma, verbose=0)
    model.learn(total_timesteps=total_timesteps, log_interval=4, progress_bar=True)
    model.save(os.path.join(save_path, model_name))

    del model

def CollectData(model_name: str, num_episodes=10):
    model = SAC.load(os.path.join("./trainings", model_name))

    class PendulumRewardWrapper(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env=env)

        def reset(self, **kwargs):
            obs = self.env.reset(**kwargs)
            return obs

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            reward = 1 - obs[0]**2 - obs[1]**2  ######## MODIFY FOR REWARD FUNCTION
            return obs, reward, terminated, truncated, info

    env = gym.make("InvertedPendulum-v4", render_mode='rgb_array')
    env = PendulumRewardWrapper(env)
    env = gym.wrappers.RecordVideo(env, video_folder="./videos", disable_logger=True)

    for _ in range(num_episodes):
        obs, info = env.reset()
        data = []
        total_reward = 0
        terminated, truncated = False, False
        rewards = []
        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            data.append(np.concatenate((action, obs, np.array([reward]), np.array([total_reward]))))
        names = ["action", "cart_pos", "pole_angle", "cart_velocity", "pole_ang_vel", "reward", "total_reward"]
        P = pd.DataFrame(data, columns = names)
        name = model_name + "_" + str(_)
        saveFile = "recordings/" + name
        P.to_csv(saveFile + ".csv")
    env.close()
    return rewards
