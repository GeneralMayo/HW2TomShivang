import argparse
import os
import random
import gym
from gym import wrappers

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


def main():
    parser = argparse.ArgumentParser(description='Run DQN on Atari Breakout')
    parser.add_argument('--env', default='Breakout-v0', help='Atari env name')
    parser.add_argument(
        '-o', '--output', default='atari-v0', help='Directory to save data to')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')

    args = parser.parse_args()
    # args.input_shape = tuple(args.input_shape)

    # args.output = get_output_folder(args.output, args.env)

    # set up environment model
    env1= gym.make(str(args.env))
    window = 4
    input_shape = (84,84)
    num_actions = env1.action_space.n 
    GAMMA = .99
    #if (NETWORK_TYPE=="Linear" or NETWORK_TYPE=="LinearERTF" or NETWORK_TYPE=="DoubleLinear"):
    #    NUM_ITERATIONS = 5000000
    #else:
    NUM_ITERATIONS = 3000000
    TARGET_UPDATE_FREQ =  10000
    BATCH_SIZE = 32
    REPLAY_MEM_SIZE = 1000000
    REPLAY_START_SIZE = 50000
    MAX_EPISODE_LEN = 1000
    REWARD_SAMPLE = 10000
    HELD_OUT_STATES_SIZE=1000
    
    
    # load json and create model
    
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

    model= Model(input=[inputLayer], output=[qFun])
    """
    json_file = open('Dulingmodel1.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    """
    # load weights into new model
    model.load_weights("Dulingmodel2.h5")
    print("Loaded model from disk")

    
    NUM_ACTIONS = env1.action_space.n  # env.get_action_space().num_actions()

    # make dqn agent
    FRAMES_PER_STATE = 4
    MAX_EPISODE_LEN = 10000

    history = HistoryPreprocessor(FRAMES_PER_STATE - 1)

    policy = LinearDecayGreedyEpsilonPolicy(1, .05, int(1e6))

    preprocessor = Preprocessor()
    # evaluate loaded model on test data
    #compile agent
    adam = Adam(lr=0.0001)
    loss = mean_huber_loss
    model.compile(loss=loss, optimizer=adam)
    max_episode_length=MAX_EPISODE_LEN
    num_episodes =100

    """Test your agent with a provided environment.
    You shouldn't update your network parameters here. Also if you
    have any layers that vary in behavior between train/test time
    (such as dropout or batch norm), you should set them to test.
    Basically run your policy on the environment and collect stats
    like cumulative reward, average episode length, etc.
    You can also call the render function here if you want to
    visually inspect your policy.
    """
    cumulative_reward = np.zeros(num_episodes)
    actions = np.zeros(env1.action_space.n)
    no_op_max = 30

    for episodes in range(num_episodes):
        if episodes<4:
            env = wrappers.Monitor(env1, str(episodes) + '/',force=True)
        else:
            env=env1
            
        steps = 0
        
        # get initial state
        history.reset()
        history.process_state_for_network(env.reset())
        state = history.frames
        for i in range(no_op_max):
            (next_state, reward, is_terminal, info) = env.step(0)
            history.process_state_for_network(next_state)
            next_state = history.frames
            actions[0] += 1
            steps = steps + 1
            if is_terminal:
                state=env.reset()
            else:
                state=next_state


        while steps < max_episode_length:
            state = preprocessor.process_state_for_network(state)
            q_vals = model.predict(state)
            action = np.argmax(q_vals[0])
            actions[action] += 1
            #print (action)
            (next_image, reward, is_terminal, info) = env.step(action)
            cumulative_reward[episodes] = cumulative_reward[episodes] + reward
            history.process_state_for_network(next_image)
            next_state = history.frames
            state = next_state
            steps = steps + 1
            if is_terminal:
                print (steps)
                break

        print (actions)
        avg_reward = np.mean(cumulative_reward) 
       
      
    print (cumulative_reward,avg_reward)
  

if __name__ == '__main__':
    main()
