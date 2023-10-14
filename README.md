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



Cover Image generated using 
