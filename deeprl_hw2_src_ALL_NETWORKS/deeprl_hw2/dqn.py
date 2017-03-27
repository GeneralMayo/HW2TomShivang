import numpy as np
import matplotlib.pyplot as plt
import time
import random

"""Main DQN agent."""

class DQNAgent:
    """Class implementing DQN.
    This is a basic outline of the functions/parameters you will need
    in order to implement the DQNAgnet. This is just to get you
    started. You may need to tweak the parameters, add new ones, etc.
    Feel free to change the functions and funciton parameters that the
    class provides.
    We have provided docstrings to go along with our suggested API.
    Parameters
    ----------
    q_network: keras.models.Model
      Your Q-network model.
    preprocessor: deeprl_hw2.core.Preprocessor
      The preprocessor class. See the associated classes for more
      details.
    memory: deeprl_hw2.core.Memory
      Your replay memory.
    gamma: float
      Discount factor.
    target_update_freq: float
      Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
    num_burn_in: int
      Before you begin updating the Q-network your replay memory has
      to be filled up with some number of samples. This number says
      how many.
    train_freq: int
      How often you actually update your Q-Network. Sometimes
      stability is improved if you collect a couple samples for your
      replay memory, for every Q-network update that you run.
    batch_size: int
      How many samples in each minibatch.
    """
    def __init__(self,
                 q_network,
                 target_q_network,
                 preprocessor,
                 history,
                 memory,
                 policy,
                 gamma,
                 target_update_freq,
                 batch_size,
                 replay_start_size,
                 num_actions,
                 network_type,
                 reward_samp):
        self.q_network = q_network
        self.target_q_network = target_q_network
        self.preprocessor = preprocessor
        self.history=history
        self.memory = memory
        self.policy = policy
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.replay_start_size=replay_start_size
        self.num_actions = num_actions
        self.network_type = network_type
        self.reward_samp = reward_samp
        self.update_policy = None

    def compile(self, optimizer, loss_func):
        """Setup all of the TF graph variables/ops.
        This is inspired by the compile method on the
        keras.models.Model class.
        This is a good place to create the target network, setup your
        loss function and any placeholders you might need.
        
        You should use the mean_huber_loss function as your
        loss_function. You can also experiment with MSE and other
        losses.
        The optimizer can be whatever class you want. We used the
        keras.optimizers.Optimizer class. Specifically the Adam
        optimizer.
        """
        self.q_network.compile(loss=loss_func,optimizer=optimizer)

        #compile target network if this strategy is being used
        if(self.target_q_network != None):
            self.target_q_network.compile(loss=loss_func,optimizer=optimizer)

    def calc_q_values(self, state):
        """
        Given a preprocessed state (or batch of states) calculate the Q-values.
        Basically run your network on these states.
        Return
        ------
        Q-values for the state(s)
        """
        return self.q_network.predict(state)

    def select_action(self, state, **kwargs):
        """Select the action based on the current state.
        You will probably want to vary your behavior here based on
        which stage of training your in. For example, if you're still
        collecting random samples you might want to use a
        UniformRandomPolicy.
        If you're testing, you might want to use a GreedyEpsilonPolicy
        with a low epsilon.
        If you're training, you might want to use the
        LinearDecayGreedyEpsilonPolicy.
        This would also be a good place to call
        process_state_for_network in your preprocessor.
        Returns
        --------
        selected action
        """

        #get q-values
        state = self.preprocessor.process_state_for_network(state)
        q_vals = self.calc_q_values(state)
        return self.policy.select_action(q_vals, self.num_actions)


    def getInputAndNextStates(self, minibatch):
        inputStates = np.zeros((self.batch_size, 4, 84,84))
        nextStates = np.zeros((self.batch_size, 4, 84,84))
        
        for sampleIdx in range(self.batch_size):
            states=(minibatch[sampleIdx].states)
            #print (len(states[0][0]))
          
            s_t=list()
            s_t1=list()
            
            for i in range(4):
                s_t.append(states[i])
                s_t1.append(states[i+1])
            
            s_t = self.preprocessor.process_state_for_network(s_t)
            inputStates[sampleIdx] = s_t
            s_t1 = self.preprocessor.process_state_for_network(s_t1)
            nextStates[sampleIdx] = s_t1

        return [inputStates, nextStates]

    def Linear_update_policy(self, s_t, q_t, a_t, r_t, s_t1, is_terminal):
        #Note: "batch size" is 1 for this network

        #get target
        target = q_t
        if(is_terminal):
            target[0][a_t] = r_t
        else:
            q_t1 = self.calc_q_values(s_t1)
            target[0][a_t] = self.gamma*max(q_t1[0]) + r_t

        #update weights/ return loss
        return self.q_network.train_on_batch(s_t,target)

    def DQN_and_LinearERTF_update_policy(self):
        #get minibatch
        minibatch = self.memory.sample(self.batch_size)
        minibatch = self.preprocessor.process_batch(minibatch)

        #init state inputs + state targets
        [inputStates,nextStates] = self.getInputAndNextStates(minibatch)

        #forward propegation    
        targets = self.calc_q_values(inputStates)
        q_t1_All = self.target_q_network.predict(nextStates)
        
        #modify target for particular action chosen by online network and reward gained for this action
        for sampleIdx in range(self.batch_size):
            a_t=minibatch[sampleIdx].a_t 
            r_t=minibatch[sampleIdx].r_t
            is_terminal=minibatch[sampleIdx].is_terminal

            if(is_terminal):
                targets[sampleIdx][a_t] = r_t
            else:
                targets[sampleIdx][a_t] = self.gamma*max(q_t1_All[sampleIdx]) +r_t

        #update weights
        loss = self.q_network.train_on_batch(inputStates,targets)
        
        return loss

    def DoubleLinear_update_policy(self):
        #choose online/ target network
        if(random.randint(0,1)==1):
            temp = self.q_network
            self.q_network = self.target_q_network
            self.target_q_network = temp

        #follow DDQN policy
        self.DDQN_update_policy()

    def DDQN_update_policy(self):
        #get minibatch
        minibatch = self.memory.sample(self.batch_size)
        minibatch = self.preprocessor.process_batch(minibatch)

        [inputStates,nextStates] = self.getInputAndNextStates(minibatch)

        #forward propegation
        targets = self.calc_q_values(inputStates)
        q_t1_online_All = self.calc_q_values(nextStates)
        q_t1_target_All = self.target_q_network.predict(nextStates)

        #modify target for particular action chosen by online network
        for sampleIdx in range(self.batch_size):
            a_t=minibatch[sampleIdx].a_t 
            r_t=minibatch[sampleIdx].r_t
            is_terminal=minibatch[sampleIdx].is_terminal

            if(is_terminal):
                targets[sampleIdx][a_t] = r_t
            else:
                #best action according to online network will be the action which target network chooses
                a_t1_online = np.argmax(q_t1_online_All[sampleIdx])
                targets[sampleIdx][a_t] = self.gamma*q_t1_target_All[sampleIdx][a_t1_online]+r_t


        #update weights
        loss = self.q_network.train_on_batch(inputStates,targets)

        return loss

    def save_weights_on_interval(self, curiter, totaliter):
        if (curiter == 0):
            self.q_network.save_weights("weights0")
        elif (curiter == int(totaliter / 3)):
            self.q_network.save_weights("weights1")
        elif (curiter == int((totaliter / 3)) * 2):
            self.q_network.save_weights("weights2")
        elif (curiter == totaliter - 1):
            self.q_network.save_weights("weights3")

    def populate_replay_memory(self, env):
        print("Populating replay mem ...")
        #initial state for replay memory filling
        self.history.process_state_for_network(env.reset())
        s_t=self.history.frames

        # populate replay memory
        for iter in range(self.replay_start_size):
            # select action
            a_t = env.action_space.sample()
            
            # get next state, reward, is terminal
            (image, r_t, is_terminal, info) = env.step(a_t)
            self.history.process_state_for_network(image)
            s_t1=self.history.frames
            
            # store sample in memory
            self.memory.append(self.preprocessor.process_state_for_memory(s_t), a_t,
                               r_t, self.preprocessor.process_state_for_memory(s_t1), is_terminal)
            
            # update new state
            if (is_terminal):
                self.history.reset()
                self.history.process_state_for_network(env.reset())
                s_t = self.history.frames
            else:
                s_t = s_t1
        print("Done populating replay mem ...")

    def update_target(self,iteration):
        if(iteration % self.target_update_freq == 0 and
            self.network_type != "Linear" and       #has no target network
            self.network_type != "DoubleLinear"):   #target update replaced with "coin-flip"

            print ("Updating target Q network")
            self.target_q_network.set_weights(self.q_network.get_weights())
        

    def set_update_policy_function(self):
        #Select function to update this particular network's policy
        print("Selecting Update Function")

        if(self.network_type == "Linear"):
            self.update_policy = self.Linear_update_policy
        elif(self.network_type == "LinearERTF" or self.network_type == "DQN"):
            self.update_policy = self.DQN_and_LinearERTF_update_policy
        elif(self.network_type == "DoubleLinear"):
            self.update_policy = self.DoubleLinear_update_policy
        elif(self.network_type == "DDQN"):
            self.update_policy = self.DDQN_update_policy
        elif(self.network_type == "Duling"):
            #Note: Duling DQN can be trained with either normal DQN or DDQN update policy
            self.update_policy = self.DQN_and_LinearERTF_update_policy
            #self.update_policy = self.DDQN_update_policy
        else:
            raise ValueError("Invalid network type.")

    def fit(self, env, num_iterations, max_episode_length,num_episodes=20):
        """Fit your model to the provided environment.
        Its a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.
        You should probably also periodically save your network
        weights and any other useful info.
        This is where you should sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.
        Parameters
        ----------
        env: gym.Env
          This is your Atari environment. You should wrap the
          environment using the wrap_atari_env function in the
          utils.py
        num_iterations: int
          How many samples/updates to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """
        
        #populate replay memory if network has one
        if(self.memory != None):
            self.populate_replay_memory(env)

        #set update policy function
        self.set_update_policy_function()

        #get initial state
        self.history.process_state_for_network(env.reset())
        s_t = self.history.frames

        #init metric vectors (to eventually plot)
        allLoss=np.zeros(num_iterations)
        rewards=np.zeros(int(np.ceil(num_iterations/self.reward_samp)))
        avg_qvals_iter=np.zeros(int(np.ceil(num_iterations/self.reward_samp)))

        #iterate through environment samples
        for iteration in range(num_iterations):
            
            #check if target needs to be updated (all network types handeled appropriately)
            self.update_target(iteration)

            # this function saves weights 0/3, 1/3, 2/3, and 3/3 of the way through training
            self.save_weights_on_interval(iteration, num_iterations)

            #select action
            if(self.network_type == "Linear"):
                s_t = self.preprocessor.process_state_for_network(s_t)
                q_t = self.calc_q_values(s_t)
                a_t = self.policy.select_action(q_t, self.num_actions)
            else:
                a_t = self.select_action(s_t)
         
            #get next state, reward, is terminal
            (image, r_t, is_terminal, info) = env.step(a_t)
            self.history.process_state_for_network(image)
            s_t1 = self.history.frames

            #store sample in memory
            if(self.network_type != "Linear"):
                self.memory.append(self.preprocessor.process_state_for_memory(s_t), a_t,
                               r_t, self.preprocessor.process_state_for_memory(s_t1),is_terminal)

            #update policy
            if(self.network_type != "Linear"):
                loss = self.update_policy()
            else:
                s_t1 = self.preprocessor.process_state_for_network(s_t1)
                loss = self.update_policy(s_t,q_t,a_t,r_t,s_t1,is_terminal)

            allLoss[iteration] = loss

            if (iteration==0):
                print ("Training Starts")
                with open('testlog.txt',"a") as f:
                    f.write("Training Starts\n")

            if (iteration % self.reward_samp == 0):
                print("Iteration: "+str(iteration))
                """
                print("Start Evaluation\n")
                with open('testlog.txt', "a") as f:
                    f.write("Start Evaluation\n")
                cum_reward, avg_qvals= self.evaluate(env, num_episodes,max_episode_length)
                rewards[int(iteration / reward_samp)] = cum_reward
                avg_qvals_iter[int(iteration / reward_samp)] = avg_qvals
                prtscn="At iteration : " + str(iteration) + " , Average Reward = " + str(cum_reward)+ " , Average Q value = " +str(avg_qvals)+" , Loss = " +str(loss)+"\n"
                print (prtscn)
                with open('testlog.txt', "a") as f:
                    f.write(prtscn)
                """
            #update new state
            if (is_terminal):
                self.history.reset()
                self.history.process_state_for_network(env.reset())
                s_t = self.history.frames
            else:
                s_t = s_t1

        print("DONE TRAINING")
        np.save("loss_linear_MR_TF", allLoss)
        np.save("reward_linear_MR_TF", rewards)

        fig = plt.figure()
        plt.plot(allLoss)
        plt.ylabel('Loss function')
        fig.savefig('Loss.png')
        plt.clf()
        plt.plot(rewards)
        plt.ylabel('Average Reward')
        fig.savefig('reward.png')
        plt.clf()
        plt.plot(avg_qvals_iter)
        plt.ylabel('Average Q value')
        fig.savefig('q_value.png')


    def evaluate(self, env, num_episodes,max_episode_length):
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
        actions = np.zeros(env.action_space.n)
        no_op_max=30

        for episodes in range(num_episodes):
            # get initial state
            self.history.reset()
            self.history.process_state_for_network(env.reset())
            state = self.history.frames
            steps = 0
            q_vals_eval=np.zeros(no_op_max)
            for i in range(no_op_max):
                state = self.preprocessor.process_state_for_network(state)
                q_vals = self.calc_q_values(state)
                (next_image, reward, is_terminal, info) = env.step(0)
                self.history.process_state_for_network(next_image)
                next_state = self.history.frames
                actions[0] += 1
                steps = steps + 1
                q_vals_eval[i]=q_vals_eval[i]+max(q_vals[0])
                if is_terminal:
                    self.history.process_state_for_network(env.reset())
                    state = self.history.frames
                else:
                    state=next_state

            while steps < max_episode_length:
                state = self.preprocessor.process_state_for_network(state)
                q_vals = self.calc_q_values(state)
                action = np.argmax(q_vals[0])
                actions[action] += 1
                (next_image, reward, is_terminal, info) = env.step(action)
                cumulative_reward = cumulative_reward + reward
                self.history.process_state_for_network(next_image)
                next_state = self.history.frames
                state = next_state
                steps = steps + 1
                if is_terminal:
                    break

        print (actions)
        avg_reward = cumulative_reward / num_episodes
        avg_qval=np.mean(q_vals_eval)/num_episodes
        return avg_reward, avg_qval
