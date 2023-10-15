import numpy as np
import pickle
import paths
import os
import torch

import sys
sys.path.append(os.path.dirname(os.getcwd()))
from paths import create_paths

import imblearn

#create all requisite paths that are required for storing the data
PATHS=paths.create_paths()
oversample=imblearn.over_sampling.SMOTE()

def load_data(paths):
    '''
    Load all the data collected. There are two things to consider:
    1. The time taken to process a frame in the forward pass + the time taken to act (env.step()) > time for the data capture and appending to array (as we did in data collection).
    Hence, we take the current state (frame) and the action pertaining to the NEXT state (frame). Hence, our X,y will be current_frame, action taken on next_frame.
    As a proxy, our model may have next state prediction also built-in
    2. We do not want to train the dino on the frames where we made a mistake and the game ended. Hence, we will not take the last 5 frames of every game.

    Args:
    paths (list): List of paths that we are using for the project from where we can reference the path where we want to store the augmented data

    Returns:
        states_space: A consolidated list of all the states (each state is 4 frames of the game)
        actions_space: A consolidated list of all the corresponding actions (as per the logic provided above)

    '''
    states_space=[]
    actions_space=[]
    for state, action in zip(os.listdir(paths[-2]),os.listdir(paths[-1])):
        with open(os.path.join(paths[-2],state),'rb') as f:
            state_sample=pickle.load(f)
        with open(os.path.join(paths[-1],action),'rb') as f:
            action_sample=pickle.load(f)
        assert len(state_sample[:-5])==len(action_sample[1:-4])
        sum+=len(state_sample[:-5])
        for state_instance, action_instance in zip(state_sample[:-5], action_sample[1:-4]):
            states_space.append(state_instance)
            actions_space.append(action_instance)
        del state_sample, action_sample
    return states_space, actions_space

states_space, actions_space=load_data(PATHS)
assert len(states_space)==len(actions_space)

def data_augmentation(state_space, action_space,smote_object,paths_list, return_data=False):
    '''
    The data captured is heavily imbalanced - Up to a score of 3000, around 80% of the frames only have a run action 15% of the frames have a jump action and 5% have a duck action.
    Hence to balance out the data, we need to augment the data for 'jump' and 'duck'. We use the SMOTE (https://arxiv.org/abs/1106.1813) methodology to augment the data. My initial data 
    led to an augmentation dataset of almost 2.5X.

    However, we use SMOTE augmentation algorithm from imblearn which only takes a list of 2-D arrays to augment. It also does not accept boolean dtype arrays.
    Hence, we need to transform this to np.byte dtype for the augmentation process.

    Post augmentation, we reshape the arrays and change the bool type back to "bool". Change the dtype is only for memory capacity reasons. Not mandatory. 

    Args:
    state_space (list): A list of numpy arrays with each numpy array having a shape of (4,139,145), dtype of 'bool' and representing one sample of the game
    action_space (list): A list of actions, each action depicted as 0,1 or 2
    smote_object (imblearn.over_sampling._smote.base.SMOTE object): Instantiated object of an SMOTE oversampling class (from imblearn library: pip install imblearn)
    paths_list (list): List of paths that we are using for the project from where we can reference the path where we want to store the augmented data
    return_data (bool): We are writing all the augmented data to disk by default. However, if this value is True, then we return the augmented states and action data. Else None objects are returned
        (default is False)

    Returns:
        None,None: if return_data is False
        augmented state_space, augmented action_space if return_data is True
    '''
    states_shape_reshape=[x.reshape(-1).numpy().astype(np.byte) for x in state_space]
    del state_space #to conserve memory; optional if you have enough memory / virtual memory
    states_shape_reshape,actions_space=smote_object.fit_resample(states_shape_reshape,action_space)
    #For my data, states_shape_reshape had a size of almost 50+ GB. Feel free to save this to disk using pickle if you need to.
    states = [np.array(state).astype(bool) for state in states_shape_reshape]
    with open(os.path.join(paths_list[-3],'states_processed.pkl'),'wb') as f:
        pickle.dump(states)
    with open(os.path.join(paths_list[-3],'actions_processed.pkl'),'wb') as f:
        pickle.dump(actions_space)
    print('Data processed and stored')
    if return_data:
        return (states, actions_space)
    else:
        return (None,None)

oversample=imblearn.over_sampling.SMOTE()   
augmented_states, augmented_actions=data_augmentation(states_space, actions_space,smote_object=oversample,paths_list=PATHS, return_data=False)
