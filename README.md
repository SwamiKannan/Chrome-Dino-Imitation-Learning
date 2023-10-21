# The Chrome Dinosaur Game

<p align="center">
 <img src="https://github.com/SwamiKannan/Chrome-Dino-Imitation-Learning/blob/main/cover.png" width=60%>
</p>

 ## Automated the playing of the chrome Dino game using Imitation Learning
 I was trying to implement an RL model to play the Chrome Dino game. This game is available @ chrome://dino in a Chrome browser or brave://dino in a Brave browser. An online version of the game is available at <a href="https://chromedino.com/">the T-Rex Chrome Dino Game website </a>.

However, I tried using multiple algorithms including DQN, A3C, etc. and even after 10K steps, it could barely cross one obstacle ("cactus"). I realized that it took almost 90K timesteps to get even a remotely well functioning to clear almost 400 frames of play. A large part of the learning was wasted because the model was trying to extract the necessary features and predict the action in near-real time which would hurt its efficiency (as the game was running pretty fast as 60 fps). This led to some kind of cold start problem due to three objectives (not in the typical DL definition):
1. Extract the relevant features (What is happening?)
2. Create the action and identify rewards, backpropogate the value function (What should the agent do?)
3. Do all this while the game is running leading to severe lags.

I felt this cold start problem could be solved by imitation learning i.e. training the model based on our own gameplay initially which could then be uploaded as a state value prediction model (for DQNs) or an action-prediction model (for A3C) as a starting point for Deep RL algorithms.

This also jives with my larger philosophy that a single model may not be proficient at doing multiple activities at the same time.

## Statistics of data collection:
1. No. of games played: __25__
2. No. of frames processed: __55932__
3. Actions collected:
    1.  Run: __46545__ frames
    2.  Jump: __8286__ frames
    3.  Duck: __1101__ frames
## How to run this repo:
1. Capture Frames:
    1. Setup the Dino game in your browser (chrome://dino or brave://dino for Chrome and Brave respectively). Make sure that the screen is maximized horizontally and no vertical scroll bars are on the window.
    2. Run <a href="https://github.com/SwamiKannan/Chrome-Dino-Imitation-Learning/blob/main/src/DataCapture/capture_data.py"> and immediately make the Dino screen active.
    3. Press space to start the game.
    4. Use only "up" and "down" arrows to control the game. Using the "space" key does not capture the jump.
2. Process the frames:
   1. Run the following codes in this specific order:
       1. <a href="https://github.com/SwamiKannan/Chrome-Dino-Imitation-Learning/blob/main/src/DataPreprocessing/data_preprocessing.py"> data_processing.py </a>
       2. <a href="https://github.com/SwamiKannan/Chrome-Dino-Imitation-Learning/blob/main/src/DataPreprocessing/train_val_test.py">train_val_test.py</a>
3. Train the model:
   1. Run <a href="https://github.com/SwamiKannan/Chrome-Dino-Imitation-Learning/blob/main/src/Model/train.py">train.py</a>
4. Test the models:
   1. Run <a href="https://github.com/SwamiKannan/Chrome-Dino-Imitation-Learning/blob/main/src/Model/test.py">test.py</a>
## Output:




https://github.com/SwamiKannan/Chrome-Dino-Imitation-Learning/assets/65940566/8e5e2d52-60c8-46ca-be8d-f803570a9d18


## Credits for cover image:

Cover Image generated using <a href="https://www.segmind.com/models/sdxl1.0-txt2img">Segmind's Stable Diffusion XL 1.0 model</a>
### Prompt
<img src="https://github.com/SwamiKannan/Chrome-Dino-Imitation-Learning/blob/main/prompt.png">
