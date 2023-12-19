import os
import numpy as np
import copy
from muavenv.global_vars import MODEL_FOLDER, LOCAL_OBS_NAMES, PROB_OBS_DELAY, SEED
import torch as T
#T.manual_seed(SEED)
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions import MultivariateNormal
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done, backupdate_batch=None, actor=None, critic=None, train_cfg=None, n_agents=None):
        #print(self.states)
        #print()

        n_stored_states = len(self.states)
        '''
        If a backupdate is needed (i.e., observation delay is taken into account and the current state is associated with an old AoI
        which is already in memory with other observations), then add new elements to the stored vectors:
        '''
        if backupdate_batch==None:
            self.states.append(state)
            self.actions.append(action)
            self.probs.append(probs)
            self.vals.append(vals)
            self.rewards.append(reward)
            self.dones.append(done)
        # Otherwise replace the old stored element with the associated 'time' (which has been received now because of the observation delay):
        else:
            """
            _________________________________________________________________________________________________________________________________
            We do not perform backtrack on actions since they have already been executed in the past and obviously they cannot change;
            'probs' instead depends on both the action and the distribution computed by the Actor based on the state: this means
            that we need to compute the new 'probs' with a new distribution (from the Actor) by keeping the same action already performed
            in the past. Obviously this is valid only in the case in which the execution is centralized since if the execution is decentralized,
            then the state feeded into the actor is local and for this reason it is obviously the same that the agent observed in the past (it
            cannot be subject to a communication delay between agents).
            'vals' depends on the the state and since it is being backtracked, it needs to be recomputed by the Critic.  
            'rewards' and 'done' are now obviously different and indeed they are assigned to their 'backupdated' value.
            Thus, we can obviously only modify features that depends on the observations, i.e. (past) observationst themselves and (past) rewards. 
            
            Since 'states', 'probs', 'vals', 'actions', 'rewards' and 'dones' are updated by appending the related new value at each iteration,
            then from them it is not possible to extract the observation i-th associated with the AoI contained inside the ENode memory:
            indeed the ENode memory contains all the (MIXED) observations of all the agents associated with the related AoI. To solve this
            'issue', it is needed to scroll an index starting from the 'backupdate_batch' times the number of the agents: in such a way it
            will be possible to associate each observation to the correct 'action', 'prob', 'val', 'reward' and 'done'. In order to avoid
            to 'lose' the old observations associated with some specific agent, the batch_size associated with the features stored
            here in PPOMemory is automatically set equal to 'batch_size = batch_size*n_agents': indeed, in this way it will be possible to store
            all the (MIXED) observations of all the agents at each available and stored AoI (for a total number of stored AoIs only equal to 'batch_size': each of them, as already
            explained, will provide the (MIXED) observation of each agent at the Enode side).

            # To avoid the latter procedure you could use a dictionary by changing also the way you use to selec states, probs (etc.).
            _________________________________________________________________________________________________________________________________
            """

            backupdate_batch_start = backupdate_batch*n_agents
            global_states = state[0]
            local_states = state[1]

            # Redo what is inside 'choose_action' except for the action selection (indeed the old selected actions will be kept):
            backupdate_batch_i = backupdate_batch_start
            for obs_idx, obs_i in enumerate(global_states):
                '''
                Check if the number of the current observations belongs to an agent whose observation
                has not been stored (due to the buffer capacity):
                '''
                if backupdate_batch_i>=n_stored_states:
                    break

                # CTDE learning paradigm case:
                if train_cfg.ct_de_paradigm:
                    actor_state = T.tensor([local_states[obs_idx]], dtype=T.float).to(actor.device)
                else:
                    actor_state = T.tensor([global_states[obs_idx]], dtype=T.float).to(actor.device)

                critic_state = T.tensor([global_states[obs_idx]], dtype=T.float).to(actor.device)

                dist = actor(actor_state)
                value = critic(critic_state)

                current_past_action = self.actions[backupdate_batch_i]
                current_past_action = T.tensor(current_past_action, dtype=T.float).to(actor.device)

                current_backupdated_probs = T.squeeze(dist.log_prob(current_past_action)).item()
                current_backupdated_value = T.squeeze(value).item()

                self.probs[backupdate_batch_i] = current_backupdated_probs
                self.vals[backupdate_batch_i] = current_backupdated_value
                self.rewards[backupdate_batch_i] = reward[obs_idx]
                self.dones[backupdate_batch_i] = done # -> 'done' is the same for all the agents, thus there is only one 'done' for all the agents

                backupdate_batch_i += 1

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class ActorNetwork(nn.Module):
    def __init__(self, train_cfg: "TrainingConfig", n_actions, input_dims, lr,
            fc1_dims=256, fc2_dims=256, chkpt_dir_save=MODEL_FOLDER, chkpt_dir_load=MODEL_FOLDER):
        super(ActorNetwork, self).__init__()
        self.train_cfg = train_cfg
        self.n_actions = n_actions
        
        if chkpt_dir_save!=None:
            if not os.path.isdir(chkpt_dir_save):
                os.mkdir(chkpt_dir_save)
            self.checkpoint_file_save = os.path.join(chkpt_dir_save, 'actor_torch_ppo')
        if chkpt_dir_load!=None:
            self.checkpoint_file_load = os.path.join(chkpt_dir_load, 'actor_torch_ppo')
        
        # Discrete action space case:
        if self.train_cfg.action_space=='discrete':
            self.actor = nn.Sequential(
                    nn.Linear(*input_dims, fc1_dims),
                    nn.Tanh(),
                    nn.Linear(fc1_dims, fc2_dims),
                    nn.Tanh(), 
                    nn.Linear(fc2_dims, n_actions),
                    nn.Softmax(dim=-1)
            )
        # Continuous action space case:
        else:
            self.actor = nn.Sequential(
                    nn.Linear(*input_dims, fc1_dims),
                    nn.Tanh(),
                    nn.Linear(fc1_dims, fc2_dims),
                    nn.Tanh(), 
                    nn.Linear(fc2_dims, n_actions)
            )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        # Discrete action space case:
        if self.train_cfg.action_space=='discrete':
            dist = self.actor(state)
            dist = Categorical(dist)
        # Continous action space case:
        else:
            mu = self.actor(state)

            def set_action_var(sigma):
                std = T.full((self.n_actions,), pow(sigma, 2))
                return std

            var = set_action_var(self.train_cfg.sigma)
            var = T.diag(var).to(self.device)
            dist = MultivariateNormal(mu, var)
        
        return dist

    def save_checkpoint(self):
        # Both parameters and persistent buffers (e.g. running averages) are included
        T.save(self.state_dict(), self.checkpoint_file_save) # -> 'state_dict()' returns a dictionary containing a whole state of the module.

    def load_checkpoint(self):
        print(self.checkpoint_file_load)
        # 'load_state_dict()' copies parameters and buffers from state_dict into this module and its descendants
        self.load_state_dict(T.load(self.checkpoint_file_load)) # strict=False -> to only load matching weights in the dictionary

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, lr, fc1_dims=256, fc2_dims=256,
            chkpt_dir_save=MODEL_FOLDER, chkpt_dir_load=MODEL_FOLDER): 
        super(CriticNetwork, self).__init__()
        
        if chkpt_dir_save!=None:
            if not os.path.isdir(chkpt_dir_save):
                os.mkdir(chkpt_dir_save)
            self.checkpoint_file_save = os.path.join(chkpt_dir_save, 'critic_torch_ppo')
        if chkpt_dir_load!=None:
            self.checkpoint_file_load = os.path.join(chkpt_dir_load, 'critic_torch_ppo')

        self.critic = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.Tanh(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.Tanh(),
                nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file_save)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file_load))

class Agent:
    def __init__(self, train_cfg: "TrainingConfig", chkpt_dir_save, chkpt_dir_load, n_actions,
            actor_input_dims, critic_input_dims, gamma=0.99, lr=0.0003, gae_lambda=0.95, policy_clip=0.2, c_value=0.5, #input_dims
            c_entropy=0., batch_size=64, n_epochs=10):
        self.train_cfg = train_cfg
        self.n_actions = n_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.c_value = c_value
        self.c_entropy = c_entropy
        self.n_epochs = n_epochs

        self.actor = ActorNetwork(train_cfg=train_cfg, n_actions=n_actions, input_dims=actor_input_dims, # input_dims=input_dims
                                  lr=lr, fc1_dims=train_cfg.A_fc1_dims_ppo, fc2_dims=train_cfg.A_fc2_dims_ppo, chkpt_dir_save=chkpt_dir_save, chkpt_dir_load=chkpt_dir_load)
        self.critic = CriticNetwork(input_dims=critic_input_dims, lr=lr, # input_dims=input_dims
                                    fc1_dims=train_cfg.C_fc1_dims_ppo, fc2_dims=train_cfg.C_fc2_dims_ppo, chkpt_dir_save=chkpt_dir_save, chkpt_dir_load=chkpt_dir_load)
        self.memory = PPOMemory(batch_size=batch_size)
       
    def remember(self, states, actions, probs, vals, rewards, done, flights, enode, observation_delay=False, cumulative_reward=False):
        # 'states' is as follows: states = [global_observations, local_observations]
        n_global_AND_local_states = len(states[0]) + len(states[1])
        '''
        Real number of states (that could be counted either considering the global (0) or the local state (1)).
        This number can be computed based either on the global or on the local observations state: this choice is
        arbitrary as the single state will be still 'single' regardless of the number of the features that it includes (indeed a single
        global state will have more features than a single local state, but the state will be 'a single one' anyway).
        '''
        n_states = len(states[0])
        n_flights = len(flights)

        current_state = []
        assert n_states==len(actions)==len(probs)==len(vals)==len(rewards), 'State-action-probs-vals-rewards dimension mistmatch!'
        # Update the current observations associated with the current elapsed time:
        for i in range(n_states):
            # Both global and local observations are stored for the same state:
            current_state = [states[0][i]] + [states[1][i]]
            self.memory.store_memory(current_state, actions[i], probs[i], vals[i], rewards[i], done) # -> DONE is the same for all the system

        # Execute a backupdate of the observations, rewards and probs only if the observation delay is being taken into account during learning:
        if observation_delay:
            for aoi, obs in enode.memory.items(): 
                '''
                The check on 'if aoi!=most_recent_aoi' is not needed since the 'elapsed_t' is updated the scope of 'step()',
                and hence when 'elapsed_t' enters in 'remember()', the current elapsed time is already at the next time. For this reason,
                at this point here, the most recent AoI available at the ENode-side is always the one right before the current elapsed time: 
                '''
                backupdated_state = (enode.memory[aoi]['norm_obs'], [f.memory[aoi]['norm_obs'] for f in flights])
                
                if not cumulative_reward:
                    backupdated_rewards = enode.rew_history[aoi]['non-cumulative']
                # Cumulative reward case:
                else:
                    backupdated_rewards = enode.rew_history[aoi]['cumulative']

                backupdated_dones = enode.memory[aoi]['dones_infos'][0] # -> '0' picks the feature 'done', and '1' picks the feature 'info'
                
                backupdapte_i = list(enode.memory).index(aoi) # -> index of the delayed observation to "backupdate" at its 'aoi' time

                self.memory.store_memory(backupdated_state, actions, probs, vals,
                                         backupdated_rewards, backupdated_dones, backupdapte_i, self.actor, self.critic, self.train_cfg, n_flights)

    def save_models(self):
        #print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observations):
        actions_list = []
        probs_list = []
        values_list = []
        
        all_observations = observations[0]
        local_observations = observations[1]
 
        for obs_idx, obs_i in enumerate(all_observations):
            # CTDE learning paradigm case:
            if self.train_cfg.ct_de_paradigm:
                actor_state = T.tensor([local_observations[obs_idx]], dtype=T.float).to(self.actor.device)
            else:
                actor_state = T.tensor([all_observations[obs_idx]], dtype=T.float).to(self.actor.device)

            critic_state = T.tensor([all_observations[obs_idx]], dtype=T.float).to(self.actor.device) # qui dovrai mettere .critic.device ????

            dist = self.actor(actor_state)
            value = self.critic(critic_state)
            
            action = dist.sample()
            probs = T.squeeze(dist.log_prob(action)).item()
            
            # Discrete action space case:
            if self.train_cfg.action_space=='discrete':
                action = T.squeeze(action).item()
            # Continuous action space case:
            else:
                action = [T.squeeze(a).item() for a in action[0]] # -> '[0]' is needed since tensor is saved as follows: tensor: [[value1, value2]] 
            value = T.squeeze(value).item()
            actions_list.append(action)
            probs_list.append(probs)
            values_list.append(value)

        return actions_list, probs_list, values_list


    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)
            
            all_batches_actor_loss = 0
            all_batches_critic_loss = 0
            all_batches_total_loss = 0
            for batch in batches:
                global_state = [s[0] for s in state_arr[batch]]
                local_state = [s[1] for s in state_arr[batch]]

                # CTDE learning paradigm case:
                if self.train_cfg.ct_de_paradigm:
                    actor_states = T.tensor(np.array(local_state), dtype=T.float).to(self.actor.device) # np.array(...) speeds up things
                else:
                    actor_states = T.tensor(np.array(global_state), dtype=T.float).to(self.actor.device) # np.array(...) speeds up things 

                critic_states = T.tensor(np.array(global_state), dtype=T.float).to(self.actor.device)
                
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch], dtype=T.float).to(self.actor.device)
                
                dist = self.actor(actor_states)
                dist_entropy = dist.entropy().mean()
                
                critic_value = self.critic(critic_states)
                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp() # prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean() # .detach()
                all_batches_actor_loss += actor_loss

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean() # .detach()
                all_batches_critic_loss += critic_loss

                # You could use also the scaled entropy term:
                total_loss = actor_loss + self.c_value*critic_loss -self.c_entropy*dist_entropy
                # total_loss.detach()
                all_batches_total_loss += total_loss

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
        
        n_batches = len(batches)
        # Compute the mean of each loss w.r.t. the number of batches:
        all_batches_actor_loss /= n_batches
        all_batches_critic_loss /= n_batches
        all_batches_total_loss /= n_batches

        self.memory.clear_memory()

        return all_batches_actor_loss, all_batches_critic_loss, all_batches_total_loss               


