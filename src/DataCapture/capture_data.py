import mss
import os
import time
import keyboard
import torch
import pickle
import numpy as np
import json

# height=1080
# width=1920
# x1=0.365*width
# x2=0.57*width
# y1=0.1*height
# y2=0.225*height

# width_calc=int(x2-x1)
# height_calc=int(y2-y1)

# bbox={"top": int(y1), "left": int(x1), "width": width_calc, "height": height_calc}

# print(width_calc)
# print(height_calc)

# sc=mss.mss()
# keystrokes={'up':1, 'down':2}

class DataCapture():
    def __init__(self,frame_count=4):
        self.height=1080
        self.obs_path=os.path.join('data','raw_data','states')
        self.act_path=os.path.join('data','raw_data','actions')
        self.width=1920
        self.x1=0.36*self.width
        self.x2=0.54*self.width
        self.y1=0.1*self.height
        self.y2=0.226853*self.height
        self.width_calc=int(self.x2-self.x1)
        self.height_calc=int(self.y2-self.y1)
        self.sc=mss.mss()
        self.bbox={"top": int(self.y1), "left": int(self.x1), "width": self.width_calc, "height": self.height_calc}
        self.frame_count=frame_count
        self.check_night=int(self.width_calc*self.height_calc*255/2)
        try: #This is to keep a unique name for each list of observations / actions for each game
            with open('capture_stats.json','r') as f:
                self.stats_dict=json.load(f)
                self.game_count=int(self.stats_dict['game_count'])
                self.act_counts=int(self.stats_dict['act_count'])
                self.jump_counts=int(self.stats_dict['jump_count'])
                self.duck_counts=int(self.stats_dict['duck_count'])
                self.run_counts=int(self.stats_dict['run_count'])
        except FileNotFoundError:
                self.game_count=0
                self.act_counts=0
                self.jump_counts=0
                self.duck_counts=0
                self.run_counts=0
                self.stats_dict={}
        except Exception as e:
             print('Exception',e)

    def capture_frame(self): # Capture one frame, process it and return the processed tensor array and the done status
        frame=self.sc.grab(self.bbox)
        raw_frame=frame.raw
        if self.prev_frame==raw_frame:
            #if self.prev_frame==self.prev2_frame:
            self.done=True
        #self.prev2_frame=self.prev_frame
        self.prev_frame=raw_frame
        frame_tensor=torch.from_numpy(np.where((255-np.array(frame)[:,:,0] if np.sum(np.array(frame)[:,:,0])<self.check_night else np.array(frame)[:,:,0])<85,0,1)).expand(1,-1,-1)
        '''The command above does three things:
        1. Convert night to day (i.e. inverts the image if the number of white pixels of the crop is less than 50% of total pixels)
        2. Make a binary numpy with values less than 85 in the RGB values as 0 and the rest as 1. This removes clouds, moons and stars which are irrelevant
        3. Convert the numpy array to torch.tensor. 
            numpy is faster than torch on where() and sum() commands. 
            Further, we can also convert an mss screenshot to numpy but not pytorch tensor.
            Hence, while we need an output of a pytorch tensor, we use numpy as an intermediary
        4. Expand the dims to allow stacking of this observation on the observation stack for the game

        '''
        return frame_tensor
    
    def get_stats(self, game_actions_list):
        act_reference={0:'run_count', 1:'jump_count',2:"duck_count"}
        frames_count=len(game_actions_list)
        a=np.unique(game_actions_list,return_counts=True)
        for act,act_count in zip(a[0].tolist(),a[1].tolist()):
            if act_reference[act] in self.stats_dict:
                self.stats_dict[act_reference[act]]+=act_count
            else:
                self.stats_dict[act_reference[act]]=act_count
        self.game_count+=1
        self.act_counts+=frames_count
        self.stats_dict['game_count']=self.game_count
        self.stats_dict['act_count']=self.act_counts
        

    
    def capture_game(self): # Capture all the obs frames and actions for one game
        self.prev_frame=None
        self.obs=torch.zeros((self.frame_count,self.height_calc,self.width_calc))
        observations_game,actions_game=[],[]
        time.sleep(5)
        self.done=False
        while not self.done:
            tensor_frame=self.capture_frame()
            if self.done:
                print('Done')
                return observations_game, actions_game
            if keyboard.is_pressed('up'):
                key_stroke=1
            elif keyboard.is_pressed('down'):
                key_stroke=2
            else:
                key_stroke=0
            self.obs=torch.vstack((tensor_frame,self.obs[:-1]))
            observations_game.append(self.obs)
            actions_game.append(key_stroke)

    
    def play_game(self):
        print('Starting')
        new_game=True
        while new_game:
            if keyboard.is_pressed('space'):
                game_states,game_actions=self.capture_game()
                games_states_uint=[state.bool() for state in game_states] #converting to unsigned int to reduce storage space
                self.get_stats(game_actions)
                obs_path=os.path.join(self.obs_path,'states'+str(self.game_count)+'.pkl')
                with open(obs_path,'wb') as f:
                    pickle.dump(games_states_uint,f)
                act_path=os.path.join(self.act_path,'actions'+str(self.game_count)+'.pkl')
                with open(act_path,'wb') as f:
                    pickle.dump(game_actions,f)
                self.get_stats(game_actions)
                with open('capture_stats.json','w') as f:
                    json.dump(self.stats_dict,f,indent=6)
                print('Game saved')
            if keyboard.is_pressed('q'):
                    print('Game stats so far:')
                    print(f'Total Games:{self.game_count}\nTotal Frames:{self.act_counts}\nTotal Runs:{self.stats_dict["run_count"]}\nTotal Jumps:{self.stats_dict["jump_count"]}\nTotal Ducks:{self.stats_dict["duck_count"]}\n')
                    new_game=False






        
    




