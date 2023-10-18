from gymnasium import spaces, envs
import gymnasium as gym
import mss
import torch
import torch.nn as nn
import numpy as np
import keyboard
import os
from stable_baselines3.common import env_checker
from gymnasium import Env, spaces
import time
from PIL import Image
import matplotlib.pyplot as plt

from VideoCapture import Video_Capture


def get_img_dims(monitor_id=1):
    '''
    The image transformations (rescale, crop, etc.) only work on a 1920 X 1080 resolution. Hence, we need to assert that in the Environment constructor.
    This function extracts these dimensions of a screenshot of the monitor to confirm the resolution. We do not need to consider number of channels
    Args: None
    Returns:
        im1_height (int) : Height of the image of a sample screenshot taken
        im1_width (int) : Width of the image of a sample screenshot taken
    '''

    sc_1 = mss.mss()
    monitor = sc_1.monitors[monitor_id]
    im_1 = sc_1.grab(monitor)
    return im_1.height, im_1.width


class DinoEnv(Env):
    '''
    The environment is created based on the new version of the gym environment (https://gymnasium.farama.org/index.html) - a maintained fork of OpenAI's gym environment.
    The environment creates a state  as follows:
        The environment creates a state by cropping a screenshot of (137X345) pixels and extracts only the first channel. Since the screenshot is a grey scale image, all channels are the same values.
        It then checks if it night time or day time. To simplify the exercise, we invert all the pixels if it is night.
        Contrast the image. Any pixel less than 85 is made 0 and all pixels greater than or equal to 85 is made white.This is the final processed frame.
        A state is a 4-frame numpy tensor of shape (4X137X345) with the latest frame on top of the stack and the last 3 frames arranged in order of recency as the other layers of this stack.
    The done status is True if two frames are exactly equal to each other. This has to be the raw pixels of the image as numpy arrays don't provide the requisite output.
    '''

    def __init__(self, seed=42, sleeptime=0.02, render_mode='human'):
        super().__init__()
        self.metadata = {"render_modes": ["human", "rgb_array"]}
        self.seed = seed
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        # The crop image is based on dimensions 1920 X 1080.
        assert get_img_dims() == (1080, 1920)
        self.dc = Video_Capture()
        self.frame = Image.fromarray(
            (np.zeros((137, 345, 4))*255).astype(np.uint8))
        self.dc.prev_frame = None
        self.dc.done = False
        self.obs_shape = (4, 137, 345)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=self.obs_shape, dtype=np.uint8)
        self.action_space = spaces.Discrete(3)
        self.obs = np.zeros(self.obs_shape).astype(np.uint8)
        self.info = {}
        self.sleeptime = sleeptime

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.obs = np.zeros(self.obs_shape).astype(np.uint8)
        self.dc.done = False
        time.sleep(2)
        keyboard.send('space')
        time.sleep(1)
        return self.obs, {}

    def get_obs(self):
        snapshot, self.frame = self.dc.capture_frame()
        self.obs = np.vstack((snapshot, self.obs[:-1])).astype(np.uint8)

    def step(self, action):
        time.sleep(self.sleeptime)
        if action > 0:
            if action == 1:
                keyboard.send('up')
            else:
                keyboard.send('down')
        self.get_obs()
        return self.obs, 1, self.dc.done, False, self.info

    def render(self, mode='rgb_array', close=False):
        im = self.frame
        if mode == 'human':
            plt.imshow(np.asarray(im))
            plt.axis('off')
        elif mode == 'rgb_array':
            return np.asarray(im)
