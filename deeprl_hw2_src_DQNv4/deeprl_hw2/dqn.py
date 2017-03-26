import numpy as np
import matplotlib.pyplot as plt
import time

class Time:
    def __init__(self):
        self.sampleMem = 0
        self.processBatch = 0
        self.getInputsTargets = 0
        self.forwardProp = 0 
        self.trainOnBatch = 0
        self.processStateForNetwork = 0
        self.storeInMem = 0
        self.updatePolicy = 0

    def print_times(self):
        print("sampleMem = "+str(self.sampleMem)+"\n"\
            +"processBatch = "+str(self.processBatch)+"\n"\
            +"getInputsTargets = "+str(self.getInputsTargets)+"\n"\
            +"forwardProp = "+str(self.forwardProp)+"\n"\
            +"trainOnBatch = "+str(self.trainOnBatch)+"\n"\
            +"processStateForNetwork "+str(self.processStateForNetwork)+"\n"\
            +"storeInMem "+str(self.storeInMem)+"\n"\
            +"updatePolicy "+str(self.updatePolicy)+"\n")
            

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
                 num_actions,
                 time):
        self.q_network = q_network
        self.target_q_network = target_q_network
        self.preprocessor = preprocessor
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
        self.time = time
        self.minybatch = None

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

    def calc_q_values(self, batch):
        """
        Given a preprocessed state (or batch of states) calculate the Q-values.
        Basically run your network on these states.
        Return
        ------
        Q-values for the state(s)
        """


        return self.q_network.predict(batch)

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

        q_vals = calc_q_values(state)
        return self.policy.select_action(q_vals)

    def populate_replay_mem(self, env):
        #initial state for replay memory filling
        self.preprocessor.process_state_for_network(env.reset())
        s_t = self.preprocessor.frames
        
        for iter in range(self.replay_start_size):
            if(iter % 10000 == 0):
                print("Iter: "+str(iter))
            # select action
            a_t = env.action_space.sample()
            # get next state, reward, is terminal
            (s_t1, r_t, is_terminal, info) = env.step(a_t)
            self.preprocessor.process_state_for_network(s_t1)
            s_t1 = self.preprocessor.frames

            # store sample in memory
            self.memory.append(self.preprocessor.process_state_for_memory(s_t), a_t,
                               r_t, self.preprocessor.process_state_for_memory(s_t1), is_terminal)
            # update new state
            if (is_terminal):
                self.preprocessor.reset()
                self.preprocessor.process_state_for_network(env.reset())
                s_t = self.preprocessor.frames
            else:
                s_t = s_t1

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
        startTime = time.time()
        #minibatch = self.memory.sample(self.batch_size)
        minibatch = self.minibatch
        self.time.sampleMem += time.time() -startTime

        startTime = time.time()
        minibatch = self.preprocessor.process_batch(minibatch)
        self.time.processBatch += time.time() -startTime

        #init state inputs + state targets
        exampleState = minibatch[0].s_t
        inputStates = np.zeros((self.batch_size, exampleState.shape[1], exampleState.shape[2], exampleState.shape[3]))
        nextStates = np.zeros((self.batch_size, exampleState.shape[1], exampleState.shape[2], exampleState.shape[3]))

        startTime = time.time()
        #make compute state inputs and targets
        for sampleIdx in range(self.batch_size):
          s_t=minibatch[sampleIdx].s_t
          s_t1=minibatch[sampleIdx].s_t1
          assert((s_t[0][0][0][0]).dtype=="float64")
          assert((s_t1[0][0][0][0]).dtype=="float64")
          
          inputStates[sampleIdx] = s_t
          nextStates[sampleIdx] = s_t1
        
        startTime2 = time.time()
        targets = self.calc_q_values(inputStates)
        q_t1_All = self.target_q_network.predict(nextStates)
        self.time.forwardProp += time.time()-startTime2

        for sampleIdx in range(self.batch_size):
            a_t=minibatch[sampleIdx].a_t   
            r_t=minibatch[sampleIdx].r_t
            is_terminal=minibatch[sampleIdx].is_terminal

            if(is_terminal):
                targets[sampleIdx][a_t] = r_t
            else:
                targets[sampleIdx][a_t] = self.gamma*max(q_t1_All[sampleIdx]) + r_t
        
        self.time.getInputsTargets += time.time() -startTime

        #update weights
        startTime = time.time()
        loss = self.q_network.train_on_batch(inputStates,targets)
        self.time.trainOnBatch += time.time() -startTime

        #update target if ready
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

        # populate replay memory
        print("Start Populating")
        self.populate_replay_mem(env)
        print("Done Populating")

        #get initial state
        self.preprocessor.process_state_for_network(env.reset())
        s_t = self.preprocessor.frames

        #init metric vectors (to eventually plot)
        allLoss=np.zeros(num_iterations)
        rewards=np.zeros(int(np.ceil(num_iterations/reward_samp)))
        avg_qvals_iter=np.zeros(int(np.ceil(num_iterations/reward_samp)))

        self.minibatch = self.memory.sample(self.batch_size)

        totalTime = time.time()
        #iterate through environment samples
        for iteration in range(num_iterations):
            #check if target  needs to be updated
            if(iteration % self.target_update_freq == 0):
                self.target_q_network.set_weights(self.q_network.get_weights())

            # this function saves weights 0/3, 1/3, 2/3, and 3/3 of the way through training
            self.save_weights_on_interval(iteration, num_iterations)

            #select action
            #print("A")
            q_t = self.calc_q_values(s_t)
            a_t = self.policy.select_action(q_t[0],self.num_actions)
            #print ((q_t[0]))
            #get next state, reward, is terminal
            (s_t1, r_t, is_terminal, info)= env.step(a_t)
            #print("B")

            startTime = time.time()
            self.preprocessor.process_state_for_network(s_t1)
            s_t1 = self.preprocessor.frames
            self.time.processStateForNetwork += time.time() -startTime
            #print("C")

            #store sample in memory
            startTime = time.time()
            self.memory.append(self.preprocessor.process_state_for_memory(s_t), a_t,
            r_t, self.preprocessor.process_state_for_memory(s_t1),is_terminal)
            self.time.storeInMem += time.time() -startTime

            #print("D")
            #update policy
            #startTime = time.time()
            #loss = self.update_policy()
            #self.time.updatePolicy += time.time() -startTime
            #allLoss[iteration] = loss

            #print("E")
            if (iteration==0):
                print ("Training Starts")
                with open('testlog.txt',"a") as f:
                    f.write("Training Starts\n")

            """
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
            """

            if(iteration % reward_samp == 0):
                self.time.print_times()

            if(iteration % 1000 == 0):
                print("Num Samples: = "+str(self.memory.num_samples))

            #update new state
            if(is_terminal):
                self.preprocessor.reset()
                self.preprocessor.process_state_for_network(env.reset())
                s_t = self.preprocessor.frames
            else:
                s_t = s_t1

        self.time.print_times()
        print("Total Time: "+str(time.time()-totalTime))
        #print("DONE")
        #np.save("loss_linear_MR_TF", allLoss)
        #np.save("reward_linear_MR_TF", rewards)

        #fig = plt.figure()
        #plt.plot(allLoss)
        #plt.ylabel('Loss function')
        #fig.savefig('Loss.png')
        #plt.clf()
        #plt.plot(rewards)
        #plt.ylabel('Average Reward')
        #fig.savefig('reward.png')
        #plt.clf()
        #plt.plot(avg_qvals_iter)
        #plt.ylabel('Average Q value')
        #fig.savefig('q_value.png')


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
            self.preprocessor.reset()
            self.preprocessor.process_state_for_network(env.reset())
            state = self.preprocessor.frames
            steps = 0
            q_vals_eval=np.zeros(no_op_max)
            for i in range(no_op_max):
                q_vals = self.calc_q_values(state)
                (next_state, reward, is_terminal, info) = env.step(0)
                self.preprocessor.process_state_for_network(next_state)
                next_state = self.preprocessor.frames
                actions[0] += 1
                steps = steps + 1
                q_vals_eval[i]=q_vals_eval[i]+max(q_vals[0])
                if is_terminal:
                    state=env.reset()
                else:
                    state=next_state

            while steps < max_episode_length:
                q_vals = self.calc_q_values(state)
                action = np.argmax(q_vals[0])
                actions[action] += 1
                (next_state, reward, is_terminal, info) = env.step(action)
                #reward = self.preprocessor.process_reward(reward)
                cumulative_reward = cumulative_reward + reward
                self.preprocessor.process_state_for_network(next_state)
                next_state = self.preprocessor.frames
                state = next_state
                steps = steps + 1
                if is_terminal:
                    break

        print (actions)
        avg_reward = cumulative_reward / num_episodes
        avg_qval=np.mean(q_vals_eval)/num_episodes
        return avg_reward, avg_qval
