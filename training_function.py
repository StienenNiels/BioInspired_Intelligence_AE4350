import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd
import numpy as np
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
        # reward = 1                                                            ######## MODIFY FOR REWARD FUNCTION
        # reward = 1 -obs[:,0]**2 -obs[:,1]**2                                  ######## MODIFY FOR REWARD FUNCTION
        reward = 1 -obs[:,0]**2 -obs[:,1]**2 -obs[:,2]**2/10 -obs[:,3]**2/10  ######## MODIFY FOR REWARD FUNCTION
        return obs, reward, done, info


def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        return env
    return _init

class RewardCallback(BaseCallback):
    """
    Custom callback for logging the average reward per episode during training.
    """

    def __init__(self, log_dir, model_name, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.log_dir = log_dir
        self.model_name = model_name
        self.csv_file = None
        self._create_csv()

    def _create_csv(self):
        """
        Create a CSV file for logging rewards.
        """
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        self.csv_file = os.path.join(self.log_dir, self.model_name + ".csv")
        with open(self.csv_file, mode="w") as file:
            file.write("episode,average_reward\n")

    def _on_step(self) -> bool:
        # VecMonitor automatically adds episode reward to info
        infos = self.locals["infos"]
        for info in infos:
            if "episode" in info.keys():
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])

                if len(self.episode_rewards) % 5 == 0:
                    avg_reward = np.mean(self.episode_rewards[-5:])
                    self._log_to_csv(len(self.episode_rewards), avg_reward)
        return True

    def _log_to_csv(self, episode, avg_reward):
        """
        Log the average reward to a CSV file.
        """
        with open(self.csv_file, mode="a") as file:
            file.write(f"{episode},{avg_reward}\n")


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

    # Create the SAC model
    model = SAC('MlpPolicy', env, learning_rate=learning_rate, buffer_size=buffer_size, learning_starts=learning_starts, batch_size=batch_size, tau=tau, gamma=gamma, verbose=0)
    
    # Create the callback
    log_dir = "./logs"  # Directory where the CSV file will be saved
    callback = RewardCallback(log_dir=log_dir, model_name=model_name)

    # Train the model
    model.learn(total_timesteps=total_timesteps, log_interval=4, progress_bar=True, callback=callback)
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
            # reward = 1                                                    ######## MODIFY FOR REWARD FUNCTION
            # reward = 1 -obs[0]**2 -obs[1]**2                              ######## MODIFY FOR REWARD FUNCTION
            reward = 1 -obs[0]**2 -obs[1]**2 -obs[2]**2/10 -obs[3]**2/10  ######## MODIFY FOR REWARD FUNCTION
        
            return obs, reward, terminated, truncated, info

    env = gym.make("InvertedPendulum-v4", render_mode='rgb_array')
    env = PendulumRewardWrapper(env)
    env = gym.wrappers.RecordVideo(env, video_folder="./videos", name_prefix=model_name, disable_logger=True)

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
