#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import os
import random
import gym

#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute,Reshape,merge)
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam
from keras import losses
from keras import backend

import graphviz
from keras.utils import plot_model

import deeprl_hw2 as tfrl
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.objectives import mean_huber_loss
from deeprl_hw2.preprocessors import HistoryPreprocessor
from deeprl_hw2.policy import LinearDecayGreedyEpsilonPolicy
from deeprl_hw2.core import ReplayMemory
from deeprl_hw2.core import Preprocessor

def create_linear_network(window, input_shape, num_actions, model_name):
    model = Sequential()
    model.add(Flatten(input_shape=(window,input_shape[0],input_shape[1])))
    model.add(Dense(num_actions, init='normal', use_bias=True, bias_initializer='normal'))
    return model

def create_DQN(window, input_shape, num_actions, model_name):
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4), init="normal",
                            border_mode='same', input_shape=(window,input_shape[0],input_shape[1])))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2), init="normal",
                            border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), init="normal",
                            border_mode='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512, init="normal"))
    model.add(Activation('relu'))
    model.add(Dense(num_actions, init="normal"))

    return model

def create_Duling_DQN(window, input_shape, num_actions, model_name):
    model = Sequential()

    #input
    inputLayer = Input(shape=(window,input_shape[0],input_shape[1]))
    
    #convolution
    conv1 = Convolution2D(32, 8, 8, subsample=(4, 4), init="normal", border_mode='same',
                                activation='relu')(inputLayer)
    conv2 = Convolution2D(64, 4, 4, subsample=(2, 2), init="normal", border_mode='same',
                                activation='relu')(conv1)

    conv3 = Convolution2D(64, 3, 3, subsample=(1, 1), init="normal", border_mode='same',
                                activation='relu')(conv2)
    
    #flatten
    flat = Flatten()(conv3)

    #fully connected advantage
    dense1 = Dense(512, init="normal", activation='relu')(flat)
    advantage = Dense(num_actions, init="normal")(dense1)

    #fully connected value
    dense2 = Dense(512, init="normal", activation='relu')(flat)
    value = Dense(1, init="normal")(dense2)

    qFun = merge([advantage, value], mode = lambda x: x[0]-backend.mean(x[0])+x[1], output_shape = (num_actions,))

    return Model(input=[inputLayer], output=[qFun])

def create_model(window, input_shape, num_actions, model_name): 
    """Create the Q-network model.

    Use Keras to construct a keras.models.Model instance (you can also
    use the SequentialModel class).

    We highly recommend that you use tf.name_scope as discussed in
    class when creating the model and the layers. This will make it
    far easier to understnad your network architecture if you are
    logging with tensorboard.

    Parameters
    ----------
    window: int
      Each input to the network is a sequence of frames. This value
      defines how many frames are in the sequence.
    input_shape: tuple(int, int)
      The expected input image size.
    num_actions: int
      Number of possible actions. Defined by the gym environment.
    model_name: str
      Useful when debugging. Makes the model show up nicer in tensorboard.

    Returns
    -------
    keras.models.Model
      The Q-model.
    """
    #Select model to build
    print("Selecting Model To Build")

    if(model_name == "Linear"):
        #online model
        model0 = create_linear_network(window, input_shape, num_actions, model_name+"_online")
        #this type of network has no target
        model1 = None
    elif(model_name == "LinearERTF" or model_name == "DoubleLinear"):
        #online model/ modelA
        model0 = create_linear_network(window, input_shape, num_actions, model_name+"_online")
        #target model/ modelB
        model1 = create_linear_network(window, input_shape, num_actions, model_name+"_target")
    elif(model_name == "DQN" or model_name == "DDQN"):
        #online model
        model0 = create_DQN(window, input_shape, num_actions, model_name+"_online")
        #target model
        model1 = create_DQN(window, input_shape, num_actions, model_name+"_target")
    elif(model_name == "Duling"):
        #online model
        model0 = create_Duling_DQN(window, input_shape, num_actions, model_name+"_online")
        #target model
        model1 = create_Duling_DQN(window, input_shape, num_actions, model_name+"_target")
    else:
        raise ValueError("Invalid network type.")

    print("Built "+model_name+" model.")
    
    print("Saving image of model(s).")
    plot_model(model0, to_file=model_name+'_model0.png')
    if(model1 != None):
        plot_model(model1, to_file=model_name+'_model1.png')


    return [model0,model1];


def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    return parent_dir


def main():
    parser = argparse.ArgumentParser(description='Run DQN on Atari Breakout')
    parser.add_argument('--env', default='SpaceInvaders-v0', help='Atari env name')
    parser.add_argument('-o', '--output', default='atari-v0', help='Directory to save data to')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--type', default="DQN", help='Type of network to train. ()')

    args = parser.parse_args()

    #check if valid network type
    network_types = ["Linear", "LinearERTF", "DoubleLinear", "DQN", "DDQN", "Duling"]
    if(not(args.type in network_types)):
        raise ValueError("Invalid network type.")

    NETWORK_TYPE = args.type

    #set up environment model
    env=gym.make(str(args.env))
    NUM_ACTIONS = env.action_space.n  
    
    #make dqn agent
    
    FRAMES_PER_STATE = 4
    INPUT_SHAPE = (84,84)
    GAMMA = .99
    if (NETWORK_TYPE=="Linear" or NETWORK_TYPE=="LinearERTF" or NETWORK_TYPE=="DoubleLinear"):
        NUM_ITERATIONS = 5000000
    else:
        NUM_ITERATIONS = 3000000
    TARGET_UPDATE_FREQ =  10000
    BATCH_SIZE = 32
    REPLAY_MEM_SIZE = 1000000
    REPLAY_START_SIZE = 50000
    MAX_EPISODE_LEN = 1000
    REWARD_SAMPLE = 10000
    HELD_OUT_STATES_SIZE=1000
    RESUME_TRAINING=1
    """
    FRAMES_PER_STATE = 4
    INPUT_SHAPE = (84,84)
    GAMMA = .99
    NUM_ITERATIONS = 20000
    TARGET_UPDATE_FREQ =  1000
    BATCH_SIZE = 32
    REPLAY_MEM_SIZE = 1000000
    REPLAY_START_SIZE = 1000
    MAX_EPISODE_LEN = 100
    REWARD_SAMPLE = 1000
    HELD_OUT_STATES_SIZE = 1000

    """
    #retuns a list of models ie: [Online,None] or [Online,Target] or [OnlineA,OnlineB]
    models = create_model(FRAMES_PER_STATE, INPUT_SHAPE, NUM_ACTIONS, NETWORK_TYPE)
    
    if RESUME_TRAINING==1:
        policy = LinearDecayGreedyEpsilonPolicy(0.5,.05,int(1e6))
        weightstr="saved_weights/"+NETWORK_TYPE+"model2.h5"
        models[0].load_weights(weightstr)
        REPLAY_START_SIZE = 500000
    else:
        policy = LinearDecayGreedyEpsilonPolicy(1,.05,int(1e6))
        
    history = HistoryPreprocessor(FRAMES_PER_STATE-1)
    preprocessor = Preprocessor()
    if(NETWORK_TYPE != "Linear"):
        memory = ReplayMemory(REPLAY_MEM_SIZE,FRAMES_PER_STATE)
    else:
        memory = None
    held_out_states = ReplayMemory(HELD_OUT_STATES_SIZE,FRAMES_PER_STATE)
    
    agent = DQNAgent(models[0],models[1],preprocessor,history,memory,policy,GAMMA,TARGET_UPDATE_FREQ,
        BATCH_SIZE,REPLAY_START_SIZE,NUM_ACTIONS,NETWORK_TYPE,REWARD_SAMPLE,held_out_states,HELD_OUT_STATES_SIZE)

    #compile agent
    adam = Adam(lr=0.0001)
    loss = mean_huber_loss
    agent.compile(adam,loss)
    agent.fit(env, NUM_ITERATIONS, MAX_EPISODE_LEN)

    print("Saved model to disk")


if __name__ == '__main__':
    main()
