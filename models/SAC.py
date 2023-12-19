import os
import numpy as np
import copy
from muavenv.global_vars import MODEL_FOLDER, LOCAL_OBS_NAMES
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import math

# Useful to debug the training process --> it stops whenever a 'weird value' (e.g., NaN, inf) is returned by a pytorch function:
#T.autograd.set_detect_anomaly(True)  

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions): # -> for the time being 'input_shape' is not used
        self.mem_size = max_size
        self.mem_cntr = 0
        
        self.states = []
        self.new_states = []
        self.actions = [] 
        self.rewards = [] 
        self.dones = []
        
    def store_transition(self, state, action, reward, state_, done, backupdate_batch=None, n_agents=None):
        n_stored_states = len(self.states)

        # Observation delay is not considered (and hence the 'backupdate' process is not needed):
        if backupdate_batch==None:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.new_states.append(state_)
            self.dones.append(done)
            
            self.mem_cntr += 1
        else:
            backupdate_batch_start = backupdate_batch*n_agents
            global_states = state[0]
            local_states = state[1]
            global_new_states = state_[0]
            local_new_states = state_[1]
            
            backupdate_batch_i = backupdate_batch_start
            for obs_idx, obs_i in enumerate(global_states):
                '''
                Check if the number of the current observations belongs to an agent whose observation
                has not been stored (due to the buffer capacity):
                '''
                if backupdate_batch_i>=n_stored_states:
                    break

                self.states[backupdate_batch_i] = copy.deepcopy([global_states[obs_idx]] + [local_states[obs_idx]])
                self.new_states[backupdate_batch_i] = copy.deepcopy([global_new_states[obs_idx]] + [local_new_states[obs_idx]])
                self.rewards[backupdate_batch_i] = reward[obs_idx]
                self.dones[backupdate_batch_i] = done # -> 'done' is the same for all the agents, thus there is only one 'done' for all the agents

                backupdate_batch_i += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        states = np.array(self.states)[batch]        
        states_ = np.array(self.new_states)[batch]
        actions = np.array(self.actions)[batch]
        rewards = np.array(self.rewards)[batch]
        dones = np.array(self.dones)[batch]

        return states, actions, rewards, states_, dones

class CriticNetwork(nn.Module):
    def __init__(self, train_cfg: "TrainingConfig", lr, input_dims, n_actions, fc1_dims=256, fc2_dims=256,
            name='critic', chkpt_dir_save=MODEL_FOLDER, chkpt_dir_load=MODEL_FOLDER): # tmp/ppo , chkpt_dir='tmp/sac'):
        super(CriticNetwork, self).__init__()
        
        self.train_cfg = train_cfg

        if chkpt_dir_save!=None:
            if not os.path.isdir(chkpt_dir_save):
                os.mkdir(chkpt_dir_save)
            self.checkpoint_file_save = os.path.join(chkpt_dir_save, name+'_torch_sac')
        if chkpt_dir_load!=None:
            self.checkpoint_file_load = os.path.join(chkpt_dir_load, name+'_torch_sac')

        # CTDE learning paradigm case:
        if self.train_cfg.action_space=='discrete':
            self.fc1 = nn.Linear(input_dims[0], fc1_dims)
        else:
            self.fc1 = nn.Linear(input_dims[0]+n_actions, fc1_dims)
        
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action=None):
        # Discrete action space case:
        if self.train_cfg.action_space=='discrete':
            x = state
        # Continuous action space case:
        else:
            assert action!=None, 'The action space is continuous, and hence the action must be passed to the Critic!'
            x = T.cat([state, action], dim=1)
        
        action_value = self.fc1(x)
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)

        return q

    def save_checkpoint(self):
        #print('... saving models ...')
        T.save(self.state_dict(), self.checkpoint_file_save)

    def load_checkpoint(self):
        print('... loading models ...')
        self.load_state_dict(T.load(self.checkpoint_file_load))

class ValueNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims=256, fc2_dims=256,
            name='value', chkpt_dir_save=MODEL_FOLDER, chkpt_dir_load=MODEL_FOLDER):
        super(ValueNetwork, self).__init__()

        if chkpt_dir_save!=None:
            if not os.path.isdir(chkpt_dir_save):
                os.mkdir(chkpt_dir_save)
            self.checkpoint_file_save = os.path.join(chkpt_dir_save, name+'_torch_sac')
        if chkpt_dir_load!=None:
            self.checkpoint_file_load = os.path.join(chkpt_dir_load, name+'_torch_sac')

        # NN Structure:
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.v = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)

        return v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file_save)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file_load))

class ActorNetwork(nn.Module):
    def __init__(self, train_cfg: "TrainingConfig", lr, input_dims, max_action, fc1_dims=256, 
            fc2_dims=256, n_actions=2, name='actor', chkpt_dir_save=MODEL_FOLDER, chkpt_dir_load=MODEL_FOLDER): #chkpt_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()

        self.train_cfg = train_cfg
        self.reparam_noise = 1e-6
        self.max_action = max_action

        if chkpt_dir_save!=None:
            if not os.path.isdir(chkpt_dir_save):
                os.mkdir(chkpt_dir_save)
            self.checkpoint_file_save = os.path.join(chkpt_dir_save, name+'_torch_sac')
        if chkpt_dir_load!=None:
            self.checkpoint_file_load = os.path.join(chkpt_dir_load, name+'_torch_sac')

        # NN Structure:
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mu = nn.Linear(fc2_dims, n_actions)
        self.softmax = nn.Softmax(dim=-1)
        self.sigma = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        if self.train_cfg.action_space=='discrete':
            prob = self.mu(prob)
            prob = self.softmax(prob)
            # Distribution:
            probabilities = Categorical(prob)
        else:
            mu = self.mu(prob)
            sigma = self.sigma(prob)
            sigma = T.clamp(sigma, min=self.reparam_noise, max=1)
            probabilities = Normal(mu, sigma)

        return probabilities, prob

    def sample_normal(self, state, reparameterize=True):
        probabilities, prob = self.forward(state)

        # Reparameterization case:
        if reparameterize:
            # Discrete action space case:
            if self.train_cfg.action_space=='discrete':
                actions = probabilities.sample() # reparameterization -> (mean + std * N(0,1)
            # Continuous action space case:
            else:
                actions = probabilities.rsample()    
        else:
            actions = probabilities.sample()
        
        # Discrete action space case:
        if self.train_cfg.action_space=='discrete':
            # Action probabilities for calculating the adapted soft-Q loss:
            action = actions
            
            z = T.as_tensor([prob[0][action[i].item()]== 0.0 for i in range(prob.size()[0])]).to(self.device)
            z = z.float()*self.reparam_noise
            
            log_probs = T.as_tensor([T.log(prob[0][i] + z[i]) for i in range(z.size()[0])]).to(self.device)            
        # Continuous action space case:
        else:
            '''
            'max_action' is used as in this way the action is going to be proportional to tanh(); the tanh() is bounded by +1 and -1 and not all
            the enivornments are bounded by +1 and -1, and hence we want to take into account this without cutting off half of the action space arbitrarily.
            In few words, 'max_action' takes care of the fact that the environment may have max actions outside the bounds +1 and -1 (which are obviously the bounds of the tanh())
            '''
            action = T.tanh(actions)*T.tensor(self.max_action).to(self.device)
            log_probs = probabilities.log_prob(actions)
            # Enforcing Action Bound:
            log_probs -= T.log(1-action.pow(2)+self.reparam_noise)
            log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self):
        #print('... saving models ...')
        # Both parameters and persistent buffers (e.g. running averages) are included:
        T.save(self.state_dict(), self.checkpoint_file_save) # -> 'state_dict()' returns a dictionary containing a whole state of the module.

    def load_checkpoint(self):
        print('...loading models ...')
        self.load_state_dict(T.load(self.checkpoint_file_load)) # -> 'load_state_dict()' copies parameters and buffers from state_dict into this module and its descendants

class Agent():
    def __init__(self, train_cfg: "TrainingConfig", chkpt_dir_save, chkpt_dir_load, max_action,
            actor_input_dims, critic_input_dims, value_input_dims, action_dim, n_actions=2, lr1=0.0003, lr2=0.0003, alpha=0.0003,
            gamma=0.99, max_size=1000000, tau=0.005,
            batch_size=256, reward_scale=2, n_epochs=10):
        self.train_cfg = train_cfg
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.action_dim = action_dim
        self.n_epochs = n_epochs
        self.scale = reward_scale
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        # 'alpha' autotuning (and reward scale assignment accordingly):
        if alpha=='auto':
            self.alpha_autotune = True
            alpha_initial = 1
            target_entropy_scale = 0.98
            
            # Discrete action space case:
            if self.train_cfg.action_space=='discrete':
                self.target_entropy = target_entropy_scale*(-np.log(1/self.n_actions))
                self.log_alpha = T.tensor(np.log(alpha_initial), requires_grad=True)
                # 'exp(_log_alpha)' can prevent alpha dropping too fast when it is close to zero (if it is 0, then scale=1/alpha=1/0=inf!):
                self.alpha = self.log_alpha.exp()
                self.alpha_optimiser = T.optim.Adam([self.log_alpha], lr=lr2) # -> use the same learning rate used by critic and value networks 
            # Continuous action space case:
            else:
                self.target_entropy = -target_entropy_scale*T.prod(T.Tensor(self.action_dim).to(self.device)).item()
                self.log_alpha = T.zeros(1, requires_grad=True, device=self.device)
                self.alpha = self.log_alpha.exp().item()
                self.alpha_optimiser = optim.Adam([self.log_alpha], lr=lr2) # -> use the same learning rate used by critic and value networks
            
            self.scale = pow(self.alpha, -1)
        # Explicit reward scale assignment:
        elif alpha==None:
            self.alpha_autotune = False
            self.scale = reward_scale
        # Implicit reward scale assignment:
        else:
            self.alpha_autotune = False
            self.alpha = alpha
            self.scale = pow(self.alpha, -1)

        self.actor = ActorNetwork(train_cfg=train_cfg, lr=lr1, input_dims=actor_input_dims, max_action=max_action,
                                  fc1_dims=train_cfg.A_fc1_dims_sac, fc2_dims=train_cfg.A_fc2_dims_sac, n_actions=n_actions,
                                  name='actor', chkpt_dir_save=chkpt_dir_save, chkpt_dir_load=chkpt_dir_load)
        
        self.critic_1 = CriticNetwork(train_cfg=train_cfg, lr=lr2, n_actions=n_actions, input_dims=critic_input_dims,
                                      fc1_dims=train_cfg.C_fc1_dims_sac, fc2_dims=train_cfg.C_fc2_dims_sac, name='critic_1',
                                      chkpt_dir_save=chkpt_dir_save, chkpt_dir_load=chkpt_dir_load)
        self.critic_2 = CriticNetwork(train_cfg=train_cfg, lr=lr2, n_actions=n_actions, input_dims=critic_input_dims,
                                      fc1_dims=train_cfg.C_fc1_dims_sac, fc2_dims=train_cfg.C_fc2_dims_sac, name='critic_2',
                                      chkpt_dir_save=chkpt_dir_save, chkpt_dir_load=chkpt_dir_load)
        
        self.value = ValueNetwork(lr=lr2, input_dims=value_input_dims,
                                  fc1_dims=train_cfg.V_fc1_dims_sac, fc2_dims=train_cfg.V_fc2_dims_sac,
                                  name='value', chkpt_dir_save=chkpt_dir_save, chkpt_dir_load=chkpt_dir_load)
        self.target_value = ValueNetwork(lr=lr2, input_dims=value_input_dims,
                                         fc1_dims=train_cfg.V_fc1_dims_sac, fc2_dims=train_cfg.V_fc2_dims_sac,
                                         name='target_value', chkpt_dir_save=chkpt_dir_save, chkpt_dir_load=chkpt_dir_load)

        self.memory = ReplayBuffer(max_size=max_size, input_shape=critic_input_dims, n_actions=n_actions) # -> the input dims of the replay buffer is set equal to that of the Critic network

        self.update_network_parameters(tau=1)

    def temperature_loss(self, log_action_probabilities, action_probs=None):
        if self.train_cfg.action_space=='discrete':
            assert action_probs!=None, 'When the action space is discrete, then the "action_probs" must be used to compute the temperature loss!'
            alpha_loss = (action_probs*(-self.log_alpha*(log_action_probabilities + self.target_entropy).detach())).mean()
        else:
            alpha_loss = -(self.log_alpha*(log_action_probabilities + self.target_entropy).detach()).mean()
        return alpha_loss

    def choose_action(self, observations):
        actions_list = []
        
        all_observations = observations[0]
        local_observations = observations[1]

        for obs_idx, obs_i in enumerate(all_observations):
            # CTDE learning paradigm case:
            if self.train_cfg.ct_de_paradigm:
                actor_state = T.tensor([local_observations[obs_idx]], dtype=T.float).to(self.actor.device)
            else:
                actor_state = T.tensor([all_observations[obs_idx]], dtype=T.float).to(self.actor.device)

            action, _ = self.actor.sample_normal(actor_state, reparameterize=False)
            action.cpu().detach().numpy()[0]
            
            # Discrete action space case:
            if self.train_cfg.action_space=='discrete':
                action = T.squeeze(action).item()
            # Continuous action space case:
            else:
                action = [T.squeeze(a).item() for a in action[0]] # -> '[0]' is needed since tensor is saved sa follows: tensor: [[value1, value2]]

            actions_list.append(action)

        return actions_list

    def remember(self, states, actions, rewards, new_states, done, flights, enode, observation_delay=False, cumulative_reward=False):
        n_global_AND_local_states = len(states[0]) + len(states[1]) # states is as follows: states = [global_observations, local_observations], where 'global_observations'
        n_states = len(states[0]) # This is the real number of states (that could be counted either considering the global (0) or the local state (1))
        n_actions = len(actions)
        n_rewards = len(rewards)
        n_new_states = len(new_states[0])
        n_flights = len(flights)

        current_state = []
        assert n_states==n_actions==n_rewards==n_new_states, 'State-action-probs-vals-rewards dimension mistmatch!'
        # Update the current observations associated with the current elapsed time:
        for i in range(n_states):
            # Both global and local observations are stored for the same state:
            current_state = [states[0][i]] + [states[1][i]]
            new_current_state = [new_states[0][i]] + [new_states[1][i]]
            
            self.memory.store_transition(current_state, actions[i], rewards[i], new_current_state, done)
        
        # Execute a backupdate of the observations, rewards and probs only if the observation delay is being taken into account during learning:
        if observation_delay:
            #print("CI SONOOOOOOOOOOOOOOOOOOOOOOOOOO")
            #breakpoint()
            for aoi, obs in enode.memory.items(): 
                '''
                The check on 'if aoi!=most_recent_aoi' is not needed since the 'elapsed_t' is updated the scope of 'step()',
                and hence when 'elapsed_t' enters in 'remember()', the current elapsed time is already at the next time. For this reason,
                at this point here, the most recent AoI available at the ENode-side is always the one right before the current elapsed time: 
                '''
                backupdated_state = (enode.memory[aoi]['norm_obs'], [f.memory[aoi]['norm_obs'] for f in flights])
                
                if not cumulative_reward:
                    backupdated_rewards = enode.rew_history[aoi]['non-cumulative']
                else:
                    backupdated_rewards = enode.rew_history[aoi]['cumulative']
                '''
                --------------------------------------------------------------------------------------------------------------
                NOTE:
                Since 'obs=obs_' after calling 'remember()' in 'train_and_sweep.py', then the actual backupdated state
                corresponds to the same backupdate but on the new_state and at the AoI before the current one:
                --------------------------------------------------------------------------------------------------------------
                '''
                sorted_aoi_list = sorted(list(enode.memory.keys()))
                previous_aoi_idx = sorted_aoi_list.index(aoi)-1
                # If there is no previous AoI (i.e., no previous observation), then use the current one:
                previous_aoi = sorted_aoi_list[previous_aoi_idx] if previous_aoi_idx>=0 else aoi
                backupdated_new_state = (enode.memory[previous_aoi]['norm_obs'], [f.memory[previous_aoi]['norm_obs'] for f in flights])

                backupdated_dones = enode.memory[aoi]['dones_infos'][0] # -> '0' picks the feature 'done' and '1' picks the feature 'info'
                
                backupdapte_i = list(enode.memory).index(aoi) # -> index of the delayed observation to "backupdate" at its 'aoi' time 
                
                self.memory.store_transition(backupdated_state, actions, backupdated_rewards, backupdated_new_state,
                                             backupdated_dones, backupdapte_i, n_flights)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                    (1-tau)*target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        #print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

        global_state = [s[0] for s in state]
        local_state = [s[1] for s in state]
        global_new_state = [s[0] for s in new_state]
        local_new_state = [s[1] for s in new_state]

        # CTDE learning paradigm case:
        if self.train_cfg.ct_de_paradigm:
            actor_state = T.tensor(local_state, dtype=T.float).to(self.actor.device)
        else:
            actor_state = T.tensor(global_state, dtype=T.float).to(self.actor.device)

        critic_state = T.tensor(global_state, dtype=T.float).to(self.actor.device)
        critic_state_ = T.tensor(global_new_state, dtype=T.float).to(self.actor.device)

        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)

        value = self.value(critic_state).view(-1)
        value_ = self.target_value(critic_state_).view(-1)
        value_[done] = 0.0

        # No reparameterization when computing the value loss:
        actions, log_probs = self.actor.sample_normal(actor_state, reparameterize=False)
        log_probs = log_probs.view(-1)
        
        # Discrete action space case:
        if self.train_cfg.action_space=='discrete':
            q1_new_policy = self.critic_1.forward(critic_state)
            q2_new_policy = self.critic_2.forward(critic_state)
        # Continuous action space case:
        else:
            q1_new_policy = self.critic_1.forward(critic_state, actions)
            q2_new_policy = self.critic_2.forward(critic_state, actions)
        
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()

        # Discrete action space case:
        if self.train_cfg.action_space=='discrete':
            value_target = actions*(critic_value - log_probs)
        # Continuous action space case:
        else:
            value_target = critic_value - log_probs
        
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        # Discrete action space case:
        if self.train_cfg.action_space=='discrete':
            reparameterize = False
        # Continous action space case:
        else:
            reparameterize = True

        # Possible reparameterization to compute the actor loss:
        actions, log_probs = self.actor.sample_normal(actor_state, reparameterize=reparameterize)
        log_probs = log_probs.view(-1)
        # Discrete action space case:
        if self.train_cfg.action_space=='discrete':
            q1_new_policy = self.critic_1.forward(critic_state)
            q2_new_policy = self.critic_2.forward(critic_state)
        # Continuous action space case:
        else:
            q1_new_policy = self.critic_1.forward(critic_state, actions)
            q2_new_policy = self.critic_2.forward(critic_state, actions)
        
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        
        # Entropy-regularized policy loss
        actor_loss = log_probs - critic_value
        # Discrete action space case:
        if self.train_cfg.action_space=='discrete':
            actor_loss = T.mean(actions*actor_loss) # un'implementazione qui mette 'sum(dim=1)' dentro le parentesi --> !!!!!!!!!!!!!
        # Continuous action space case:
        else:
            actor_loss = T.mean(actor_loss)
        
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        
        # Entropy here comes into play:
        q_hat = self.scale*reward + self.gamma*value_
        q1_old_policy = self.critic_1.forward(critic_state, action).view(-1)
        q2_old_policy = self.critic_2.forward(critic_state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # Possible alpha loss computation (and reward scale recomputation):
        if self.alpha_autotune:
            self.alpha_optimiser.zero_grad()
            # Discrete action space case:
            if self.train_cfg.action_space=='discrete':
                alpha_loss = self.temperature_loss(log_probs, actions)
            # Continous action space case:
            else:
                alpha_loss = self.temperature_loss(log_probs)
            alpha_loss.backward()
            self.alpha_optimiser.step()
            # Discrete action space case:
            if self.train_cfg.action_space=='discrete':
                self.alpha = self.log_alpha.exp()
            # Continuous action space case:
            else:
                self.alpha = self.log_alpha.exp().item()
            self.scale = pow(self.alpha, -1)
        else:
            alpha_loss = None

        self.update_network_parameters()

        return actor_loss, critic_1_loss, critic_2_loss, critic_loss, value_loss, alpha_loss