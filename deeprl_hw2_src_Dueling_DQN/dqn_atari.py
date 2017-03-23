#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import os
import random
import gym

import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute,Reshape,merge)
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam
from keras import losses
from keras import backend
import pydot
import graphviz
from keras.utils import plot_model

import deeprl_hw2 as tfrl
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.objectives import mean_huber_loss
from deeprl_hw2.preprocessors import HistoryPreprocessor
from deeprl_hw2.policy import LinearDecayGreedyEpsilonPolicy
from deeprl_hw2.core import ReplayMemory


def create_model(window, input_shape, num_actions,
                 model_name='q_network'):  # noqa: D103
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


def main():  # noqa: D103
    parser = argparse.ArgumentParser(description='Run DQN on Atari Breakout')
    parser.add_argument('--env', default='Breakout-v0', help='Atari env name')
    parser.add_argument(
        '-o', '--output', default='atari-v0', help='Directory to save data to')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')

    args = parser.parse_args()
    #args.input_shape = tuple(args.input_shape)

    #args.output = get_output_folder(args.output, args.env)

    #set up environment model
    env=gym.make(str(args.env))
    NUM_ACTIONS = env.action_space.n 
    


    #make dqn agent
    FRAMES_PER_STATE = 4
    INPUT_SHAPE = (84,84)
    GAMMA = .99
    NUM_ITERATIONS =  5000000
    TARGET_UPDATE_FREQ = 10000
    TRAIN_FREQ = 0
    BATCH_SIZE = 32
    REPLAY_MEM_SIZE =  1000000
    REPLAY_START_SIZE = 50000
    MAX_EPISODE_LEN = 1000

    model = create_model(FRAMES_PER_STATE, INPUT_SHAPE, NUM_ACTIONS,
                 model_name='linear q_network')

    plot_model(model, to_file='model.png')


    target = create_model(FRAMES_PER_STATE, INPUT_SHAPE, NUM_ACTIONS,
                 model_name='linear q_network target')

    preprocessor = HistoryPreprocessor(FRAMES_PER_STATE-1)
    memory = ReplayMemory(REPLAY_MEM_SIZE,FRAMES_PER_STATE)
    policy = LinearDecayGreedyEpsilonPolicy(1, .05, int(1e6))
    agent = DQNAgent(model,target,preprocessor,memory,policy,GAMMA,TARGET_UPDATE_FREQ,TRAIN_FREQ,BATCH_SIZE,REPLAY_START_SIZE,NUM_ACTIONS)

    #compile agent
    adam = Adam(lr=0.0001)
    loss = mean_huber_loss
    agent.compile(adam,loss)
    agent.fit(env, NUM_ITERATIONS, MAX_EPISODE_LEN)

if __name__ == '__main__':
    main()
