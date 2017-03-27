import argparse
import os
import random
import gym
from gym import wrappers

import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute,Reshape)
from keras.models import Model
from keras.models import Sequential
from keras.models import model_from_json
from keras.optimizers import Adam
from keras import losses
#import pydot
import graphviz
from keras.utils import plot_model

import deeprl_hw2 as tfrl
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.objectives import mean_huber_loss
from deeprl_hw2.preprocessors import HistoryPreprocessor
from deeprl_hw2.policy import LinearDecayGreedyEpsilonPolicy
from deeprl_hw2.core import ReplayMemory

def main():
    # load json and create model
    json_file = open('/home/shivang/Desktop/HW2TomShivang/deeprl_hw2_src_DQNv2/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model.h5")
    print("Loaded model from disk")

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
    NUM_ACTIONS = env1.action_space.n  # env.get_action_space().num_actions()

    # make dqn agent
    FRAMES_PER_STATE = 4
    MAX_EPISODE_LEN = 1000

    preprocessor = HistoryPreprocessor(FRAMES_PER_STATE - 1)

    policy = LinearDecayGreedyEpsilonPolicy(1, .05, int(1e6))

    preprocessor = HistoryPreprocessor(FRAMES_PER_STATE-1)
    # evaluate loaded model on test data
    #compile agent
    adam = Adam(lr=0.0001)
    loss = mean_huber_loss
    model.compile(loss=loss, optimizer=adam)
    max_episode_length=MAX_EPISODE_LEN
    num_episodes = 20

    """Test your agent with a provided environment.
    You shouldn't update your network parameters here. Also if you
    have any layers that vary in behavior between train/test time
    (such as dropout or batch norm), you should set them to test.
    Basically run your policy on the environment and collect stats
    like cumulative reward, average episode length, etc.
    You can also call the render function here if you want to
    visually inspect your policy.
    """
    cumulative_reward = 0
    actions = np.zeros(env1.action_space.n)
    no_op_max = 30

    for episodes in range(num_episodes):
        if episodes<4:
            env = wrappers.Monitor(env1, '/home/shivang/Desktop/HW2TomShivang/Video_evaluation/' + str(episodes) + '/',
                               force=True)
        else:
            env=env1
        # get initial state
        preprocessor.reset()
        preprocessor.process_state_for_network(env.reset())
        state = preprocessor.frames
        steps = 0
        q_vals_eval = np.zeros(no_op_max)
        for i in range(no_op_max):
            q_vals = model.predict(state)
            (next_state, reward, is_terminal, info) = env.step(0)
            preprocessor.process_state_for_network(next_state)
            next_state = preprocessor.frames
            actions[0] += 1
            steps = steps + 1
            q_vals_eval[i] = q_vals_eval[i] + max(q_vals[0])
            if is_terminal:
                state = env.reset()
            else:
                state = next_state

        while steps < max_episode_length:
            q_vals = model.predict(state)
            action = np.argmax(q_vals[0])
            actions[action] += 1
            (next_state, reward, is_terminal, info) = env.step(action)
            # reward = self.preprocessor.process_reward(reward)
            cumulative_reward = cumulative_reward + reward
            preprocessor.process_state_for_network(next_state)
            next_state = preprocessor.frames
            state = next_state
            steps = steps + 1
            if is_terminal:
                break

    print (actions)
    avg_reward = cumulative_reward / num_episodes
    avg_qval = np.mean(q_vals_eval) / num_episodes
    print (avg_reward)
    print (avg_qval)


if __name__ == '__main__':
    main()


"""
import gym
from gym import wrappers
env1 = gym.make('Breakout-v0')
#env = wrappers.Monitor(env, '/home/shivang/Desktop/HW2TomShivang/Video_evaluation/',force=True)
for i_episode in range(5):
    env = wrappers.Monitor(env1, '/home/shivang/Desktop/HW2TomShivang/Video_evaluation/'+str(i_episode)+'/', force=True)
    observation = env.reset()
    for t in range(1000):
        env.render()
        #print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

"""