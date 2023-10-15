import numpy as np
import pickle
import os

import sys
sys.path.append(os.path.dirname(os.getcwd()))
from paths import create_paths

from sklearn.model_selection import train_test_split

paths_list=create_paths()

#To create the train-val-test split, we are going to use only the index of the features. Shuffling and split a large number of (4,139,345) numpy arrays is time consuming.
# So we will just shuffle the indices and then finally index the complete feature set. 

with open(os.path.join(paths_list[-3],'actions_processed.pkl'),'rb') as f:
    y=pickle.load(f)

def shuffle(action_space,n=1000):
    '''
    Args:
    action_space (list): A list of actions, each action depicted as 0,1 or 2
    n (int): Number of times you want the action_space to be shuffled. Default: 1000

    Returns:
        X: A shuffled index of the action_space
        y: The shuffled action_space whose indices are X
    '''

    X=np.array(range(len(action_space))).reshape(-1,1)
    y=np.array(action_space).reshape(-1,1)
    data=np.hstack((X,y))
    for i in range(n):
        np.random.shuffle(data)
        if i%50==0:
            print(i+1, 'shuffles done')
    X,y = data[:,0],data[:,1]
    return X,y

X,y = shuffle(y)
X_train, X_test, y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y) #Split training and test
X_val, X_tst, y_val,y_tst=train_test_split(X_test,y_test,test_size=0.5,stratify=y_test) #Split test into validations and final test sets

#Now we finally index the train data set.
with open(os.path.join(paths_list[-3],'states_processed.pkl'),'rb') as f:
    states=pickle.load(f)
states=[np.array(state).reshape(4, 137, 345) for state in states]
X_train_states=[states[X] for X in X_train]
X_val_states=[states[X] for X in X_val]
X_tst_states=[states[X] for X in X_tst]

#Save all the datasets to disk
with open(os.path.join(paths_list[-3],'X_train.pkl'),'wb') as f:
    pickle.dump(X_train_states,f)
with open(os.path.join(paths_list[-3],'X_val.pkl'),'wb') as f:
    pickle.dump(X_val_states,f)
with open(os.path.join(paths_list[-3],'X_test.pkl'),'wb') as f:
    pickle.dump(X_tst_states,f)

with open(os.path.join(paths_list[-3],'y_train.pkl'),'wb') as f:
    pickle.dump(y_train,f)
with open(os.path.join(paths_list[-3],'y_val.pkl'),'wb') as f:
    pickle.dump(y_val,f)
with open(os.path.join(paths_list[-3],'y_test.pkl'),'wb') as f:
    pickle.dump(y_tst,f)

