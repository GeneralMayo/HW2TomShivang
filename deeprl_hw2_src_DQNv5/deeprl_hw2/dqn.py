import numpy as np
import matplotlib.pyplot as plt
import time

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
                 held_out_states,
                 held_out_states_size,
                 gamma,
                 target_update_freq,
                 num_burn_in,
                 train_freq,
                 batch_size,
                 replay_start_size,
                 num_actions):
        self.q_network = q_network
        self.target_q_network = target_q_network
        self.preprocessor = preprocessor
        self.history=history
        self.memory = memory
        self.policy = policy
        self.held_out_states=held_out_states
        self.held_out_states_size=held_out_states_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.num_burn_in = num_burn_in
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.replay_start_size=replay_start_size
        self.num_actions = num_actions

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
        self.target_q_network.compile(loss=loss_func,optimizer=optimizer)

    def calc_q_values(self, state):
        """
        Given a preprocessed state (or batch of states) calculate the Q-values.
        Basically run your network on these states.
        Return
        ------
        Q-values for the state(s)
        """

        #How many states???
        #q_values = np.zeros(...,self.policy.num_actions)

        #iterate through states
        #  q_values[sIdx] = self.q_network.predict(state[sIdx])

        #return q_values
        #print (state.shape)
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
        #calc_q_values(state)
        state=self.preprocessor.process_state_for_network(state)
        #print (state1.shape)
        q_vals = self.calc_q_values(state)
        return self.policy.select_action(q_vals,self.num_actions)


    def update_policy(self):
        """Update your policy.
        Behavior may differ based on what stage of training your
        in. If you're in training mode then you should check if you
        should update your network parameters based on the current
        step and the value you set for train_freq.
        Inside, you'll want to sample a minibatch, calculate the
        target values, update your network, and then update your
        target values.
        You might want to return the loss and other metrics as an
        output. They can help you monitor how training is going.
        """

        #get minibatch
        #print (len(self.memory.M))
        minibatch = self.memory.sample(self.batch_size)
        #print (len(minibatch))
        minibatch = self.preprocessor.process_batch(minibatch)

        #init state inputs + state targets
        inputStates = np.zeros((self.batch_size, 4, 84,84))
        nextStates = np.zeros((self.batch_size, 4, 84,84))
        targets = np.zeros((self.batch_size,self.num_actions))
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
            
        targets = self.calc_q_values(inputStates)
        q_t1_All = self.target_q_network.predict(nextStates)
        
        for sampleIdx in range(self.batch_size):
            a_t=minibatch[sampleIdx].a_t   #This is action index
            r_t=minibatch[sampleIdx].r_t
            is_terminal=minibatch[sampleIdx].is_terminal

            if(is_terminal):
                targets[sampleIdx][a_t] = r_t
            else:
                targets[sampleIdx][a_t] = self.gamma*max(q_t1_All[sampleIdx]) +r_t
        loss = self.q_network.train_on_batch(inputStates,targets)
        
        #update target if ready                                                 
        return loss

    

    def save_weights_on_interval(self, curiter, totaliter):
        #function to save model and weights at different stages of training
        #I am doing it every 500k so that we can decide later whether we want the results till 1.5M or 3M 
        #If it doesn't show improvement in the first 1.5M but shows promise 
        # we can let it run for more.
        if (curiter % 500000==0):
            stage=int(curiter/500000)
            modelstr="model"+str(stage)+".json"
            weightstr="model"+str(stage)+".h5"
            model_json = self.q_network.to_json()
            with open(modelstr, "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            self.q_network.save_weights(weightstr)
       
            

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
        reward_samp = 10000
        time1=time.time()
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
            
            #store held out states to be used for Q value evaluation
            if iter<self.held_out_states_size:
                self.held_out_states.append(self.preprocessor.process_state_for_memory(s_t), a_t,
                               r_t, self.preprocessor.process_state_for_memory(s_t1), is_terminal)
                
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

        #print (len(self.memory.M))
        #get initial state
        self.history.process_state_for_network(env.reset())
        s_t = self.history.frames

        #init metric vectors (to eventually plot)
        allLoss=np.zeros(num_iterations)
        rewards=np.zeros(int(np.ceil(num_iterations/reward_samp)))
        avg_qvals_iter=np.zeros(int(np.ceil(num_iterations/reward_samp)))

        #iterate through environment samples
        for iteration in range(num_iterations):

            #if (iteration % reward_samp == 0):
            #print (iteration, time1-time.time())
            #check if target  needs to be updated
            if(iteration % self.target_update_freq == 0):
                print ("Updating target Q network")
                self.target_q_network.set_weights(self.q_network.get_weights())

            # this function saves weights 0/3, 1/3, 2/3, and 3/3 of the way through training
            self.save_weights_on_interval(iteration, num_iterations)

            #select action
            a_t = self.select_action(s_t)
         
            #print ((q_t[0]))
            #get next state, reward, is terminal
            (image, r_t, is_terminal, info) = env.step(a_t)
            #print (len(s_t))
            self.history.process_state_for_network(image)
            #print (len(s_t))
            s_t1 = self.history.frames
            #print (len(s_t1))
            #print (len(s_t))
            #store sample in memory
            self.memory.append(self.preprocessor.process_state_for_memory(s_t), a_t,
                               r_t, self.preprocessor.process_state_for_memory(s_t1),is_terminal)


            #update policy
            loss = self.update_policy()
            allLoss[iteration] = loss

            if (iteration==0):
                print ("Training Starts")
                with open('testlog.txt',"a") as f:
                    f.write("Training Starts\n")

            if (iteration % reward_samp == 0):
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
                
            #update new state
            if (is_terminal):
                self.history.reset()
                self.history.process_state_for_network(env.reset())
                s_t = self.history.frames
            else:
                s_t = s_t1

        #print("DONE")
        np.save("loss", allLoss)
        np.save("rewards", rewards)
        np.save("q_vals",avg_qvals_iter)
        
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
            
            steps = 0
            q_vals_eval=np.zeros(self.held_out_states_size)
            held_out=list(self.held_out_states.M) # convert the deque into a list of samples
            #print ((held_out))
            for i in range(self.held_out_states_size):                
                state = held_out[i].states[0:4]  #get the initial state from the sample
                state = self.preprocessor.process_state_for_network(state) #conversion into float
                q_vals = self.calc_q_values(state)              
                q_vals_eval[i]=q_vals_eval[i]+max(q_vals[0])  #take the max over actions and add to the current state
                
            # get initial state
            self.history.reset()
            self.history.process_state_for_network(env.reset())
            state = self.history.frames
            for i in range(no_op_max):
                (next_state, reward, is_terminal, info) = env.step(0)
                self.history.process_state_for_network(next_state)
                next_state = self.history.frames
                actions[0] += 1
                steps = steps + 1
                if is_terminal:
                    state=env.reset()
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
