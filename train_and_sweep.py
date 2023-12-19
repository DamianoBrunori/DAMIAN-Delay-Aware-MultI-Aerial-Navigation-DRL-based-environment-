import time
from tqdm import tqdm
import numpy as np
import math
import warnings
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from parsers.scenario import ScenarioConfig
from parsers.training import TrainingConfig
from muavenv.utils import Plot, TrainAux
from models.PPO import Agent as AgentPPO
from models.SAC import Agent as AgentSAC
from muavenv.utils import MainAux
from muavenv.global_vars import *
from typing import Optional, Tuple, List, Dict, Union, Any
import pickle
import functools
import wandb
import gc

class TrainANDSweep:
    """
    Class TrainANDSweep
    """

    def __init__(self, env: "Environment",
                       args: Any,
                       scnr_cfg: ScenarioConfig,
                       train_cfg: TrainingConfig,
                       writer: SummaryWriter,
                       model_save_dir: str,
                       model_load_dir: str,
                       plot_dir: str,
                       settings_src_dir: str,
                       external_obs_dir: str,
                       debug: Optional[bool] = False,
                       visible: Optional[bool] = False,
                       render_steps_saving: Optional[int] = 20) -> None:
        self.env = env
        self.args = args
        self.scnr_cfg = scnr_cfg
        self.train_cfg = train_cfg
        self.writer = writer
        self.model_save_dir = model_save_dir
        self.model_load_dir = model_load_dir
        self.settings_src_dir = settings_src_dir
        self.external_obs_dir = external_obs_dir 
        self.plot = Plot(save_dir=plot_dir) # charts utilities
        self.train_aux = TrainAux() # train utilities
        self.debug = debug
        self.visible = visible
        self.render_steps_saving = render_steps_saving

    def secmin2hours(self, time: Union[int, float]) -> Union[int, float]:
        """
        Conversion of the time into hours based on the time unit used.

        :param time: time to convert

        :return time_h: time in hours 
        """

        if self.env.time_unit=='s':
            time_h = time*3600
        elif self.env.time_unit=='m':
            time_h = time*60
        else:
            time_h = time

        return time_h

    def time_scaling(self) -> int:
        """
        Scale factor computation according to the time unit used.

        :param: None

        :return time_scaling_factor: integer representing the time scaling factor computed
        """

        if self.env.time_unit=='s' or not self.env.scenario_action_delay: # -> no scale factor if time_unit='s' or actions are taken instantaneously
            time_scaling_factor = 1
        elif self.env.time_unit=='m':
            time_scaling_factor = 60
        else:
            time_scaling_factor = 3600

        return time_scaling_factor

    def pretraining_setting(self) -> "Agent":
        """
        Initialize the agent and load the pretrained models (if test option is enabled).

        :param: None

        :return agent: the initialized agent
        """

        # CTDE learning paradigm disabled or single-agent case enabled:
        if (not self.train_cfg.ct_de_paradigm) or (self.env.num_flights==1):
            actor_input_dims = self.env.observation_space.shape
            critic_input_dims = actor_input_dims
        # CTDE learning paradigm or multi-agent case enabled:
        else:
            actor_input_dims = self.env.local_observation_space.shape
            critic_input_dims = self.env.observation_space.shape

        # Only used if SAC algorithm is applied: 
        value_input_dims = critic_input_dims # the input dimensions of the Value network is set as the input dimensions of the Critic network

        # It is not know which are the agents associated with the observations stored in the batch when the observation delay is enabled:
        if self.env.learning_observation_delay:
            batch_size = self.train_cfg.batch_size*self.scnr_cfg.num_flights # -> avoid to lose info associated with the multi-agent case
        else:
            batch_size = self.train_cfg.batch_size

        batch_size = int(batch_size) # -> 'int' is needed as batch_size is recomputed and '20' can be interpreted as '20.0', namely as a float
        
        # PPO algorithm settings:
        if self.env.algorithm=='ppo':
            agent = AgentPPO(train_cfg=self.train_cfg, chkpt_dir_save=self.model_save_dir, chkpt_dir_load=self.model_load_dir, n_actions=self.env.n_actions,
                          actor_input_dims=actor_input_dims, critic_input_dims=critic_input_dims, gamma=self.train_cfg.gamma_ppo,
                          gae_lambda=self.train_cfg.gae_lambda_ppo, lr=self.train_cfg.lr_ppo,
                          policy_clip=self.train_cfg.policy_clip_ppo, c_value=self.train_cfg.c_value_ppo,
                          c_entropy=self.train_cfg.c_entropy_ppo, batch_size=batch_size,
                          n_epochs=self.train_cfg.N)
            
            print('\nCritic Network:')
            summary(agent.critic, critic_input_dims)

        # SAC algorithm settings:
        elif self.env.algorithm=='sac':
            if self.train_cfg.action_space=='discrete':
                max_action = None # --> it will not be used when using a discrete action space
            else:
                max_action = self.env.action_space.high
            
            agent = AgentSAC(train_cfg=self.train_cfg, chkpt_dir_save=self.model_save_dir, chkpt_dir_load=self.model_load_dir,
                          max_action=max_action,
                          actor_input_dims=actor_input_dims, critic_input_dims=critic_input_dims, value_input_dims=value_input_dims,
                          action_dim=self.env.action_dim, n_actions=self.env.n_actions, lr1=self.train_cfg.lr1_sac, lr2=self.train_cfg.lr2_sac, alpha=self.train_cfg.alpha_sac,                          
                          gamma=self.train_cfg.gamma_sac, max_size=1000000, tau=self.train_cfg.tau_sac, # 'max_size' forse puoi lasciarla così di default (ma è da capire bene a cosa si riferisca esattamente!!!!!)
                          batch_size=batch_size, reward_scale=self.train_cfg.reward_scale_sac, # da capire bene cosa si intenda in SAC con 'batch_size' e quindi valutare se lasciare lo stesso di PPO oppure no !!!!!!!!!!!!!!
                          n_epochs=self.train_cfg.N)

            print('\nCritic Network 1:')
            summary(agent.critic_1, critic_input_dims)
            print('\nCritic Network 2:')
            summary(agent.critic_2, critic_input_dims)
            print('\nValue Network:')
            summary(agent.value, value_input_dims)

        # Load the the learned policy in the available models for the 'test case':
        if self.args.test:
            agent.load_models()

        print('\nActor Network:')
        summary(agent.actor, actor_input_dims)
        time.sleep(3)

        return agent

    def train(self, agent: "Agent", metric_name: Optional[Union[str, None]] = None, sweep_cng_dict: Optional[Union[Dict, None]] = None) -> None:
        """
        Training phase (apply the hyperparameters tuning search if the sweep option is enabled).

        :param agent: the already intialized agent
        :param metric: the reference metric to use when the sweep option is enabled
        :param sweep_cng_dict: a dictionary contaning the configuration parameters to use for the hyperparameters search (when the sweep option is enabled)   
        
        :return: None
        """

        if self.env.scenario_action_delay:
            time_unit = self.env.time_unit
        # No time unit if actions take place instantaneously:
        else:
            time_unit = ''
        time_scaling_factor = self.time_scaling()
        best_score = self.env.reward_range[0]
        score_history = []
        frames = []

        learn_iters = 0 # -> number of times effectively used to learn from the store experience
        avg_score = 0
        n_steps = 0 # -> number of total steps performed through all the episodes
        n_steps_per_episode = []
        avg_mobile_distance_per_episode = []

        # Set how often to learn:
        if self.scnr_cfg.explicit_clock==False:
            N = self.train_cfg.N
        else:
            N = self.env.system_clock

        usecase_for_plot = self.args.usecase1 or self.args.usecase2

        # Hyperparameters tuning case:
        if self.args.sweep:
            # Weight&Biases initialization:
            wandb.init()
            sweep_config = wandb.config
            sweep_params = list(sweep_config.keys())
            train_cfg_params = list(self.train_cfg.__dict__.keys())
            env_params = list(self.env.__dict__.keys())
            # Compute how many hours need to elapse between a learning step and the next one: 
            min_it_time_h = self.secmin2hours(time=int((self.env.dt*N)/time_scaling_factor))
            # Invalid sweep inputs check: 
            if 'epoch_duration' in sweep_params:
                epoch_durations_to_test = sweep_cng_dict['parameters']['epoch_duration']
                can_check = False
                if 'value' in epoch_durations_to_test:
                    epoch_durations_to_test = [epoch_durations_to_test['value']]
                    can_check = True
                elif 'values' in epoch_durations_to_test:
                    epoch_durations_to_test = epoch_durations_to_test['values']
                    can_check = True
                else:
                    warnings.warn('\n\nNot fixed values have been using for "epoch duration" sweep parameter, thus it could cause an errror at run time since in this case it is not perform any check to ensure that the epoch duration [h] is not smaller than the minimum time needed between a learning step and the next one!\n\n')
                    time.sleep(3)
                    can_check = False

                if can_check:
                    for ep_dur in epoch_durations_to_test:
                        assert ep_dur>=min_it_time_h, '\n\nFound "epoch duration" {} [h] too short among the ones selected for the sweep (check them all: {})! The epoch duration [h] must be larger enough to allow to start at least 1 iteration. Now the learning process runs every {} iterations, i.e., every {} [h]: thus either the epoch duration is set in such a way to be euqual or greater than {} or the number of learning iteration N is decreased\n\n'.format(ep_dur, epoch_durations_to_test, N, min_it_time_h, min_it_time_h)
            for sweep_prm_name in sweep_params:
                assert (sweep_prm_name=='epoch_duration') or (sweep_prm_name in train_cfg_params), 'Invalid Sweep parameter {}! Select sweep paramters among the ones available in training.ini file (exept for the epoch time duration "epoch_duration [h]"): {}'.format(sweep_prm_name, train_cfg_params)
                setattr(self.train_cfg, sweep_prm_name, sweep_config[sweep_prm_name])
                if sweep_prm_name in env_params:
                    setattr(self.env, sweep_prm_name, sweep_config[sweep_prm_name])            

            if self.env.persistent_task and 'n_episodes' in sweep_params:
                assert False, '\n\nYou are trying to set a sweep set of values for the number of episodes for a persistent task: when the task is persistent the episode never ends and then you need to use the "epoch duration" instead as if it represented the episodes!\n\n'

        #it = 0 # -> this initialization here is needed only for the rendering

        # Run the episodes:
        for ep in tqdm(iterable=range(1, self.train_cfg.n_episodes+1)): # -> to resize the progress bar to 10 chars use -> bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'
            if self.args.sweep:
                # Modify here if a new algorithm to be used is added:
                if self.env.algorithm=='ppo':
                    metrics = METRICS_PPO
                elif self.env.algorithm=='sac':
                    metrics = METRICS_SAC

            it = 0 # -> number of iterations for the current episode
            frames = [] # -> store the animation's frames for the current episodes
            cur_episode_distances = [] # -> store the distances among the agents in the current episode 
            # Check for a initial possible external observation:
            MainAux.manage_possible_external_obs(env=self.env, external_obs=self.args.external_obs, external_obs_dir=self.external_obs_dir, epoch=ep, it=it)
            # Reset environment
            obs = self.env.reset()
            # Set done status to false
            done = False
            scores = [0 for i in range(self.env.num_flights)]
            # Save (also plot if debug mode is enabled) the discretized environmnet with inner and outer points:
            self.plot.whole_env(pol=self.env.airspace.cartesian_polygon, POIs=self.env.lps, outer_p=self.env.outer_p, step=self.scnr_cfg.step, debug=self.debug)
            # Save (also plot if debug mode is enabled) the source path at every new episode:
            self.plot.source_path(pol=self.env.airspace.cartesian_polygon, walk=self.env.current_source_walk, local_d=self.scnr_cfg.local_d, debug=self.debug)
            # Execute one episode
            while not done:
                # Adding the current frame for the animation to save:
                frames.append(self.env.render(visible=self.visible, debug=self.debug)) 
                time.sleep(0.05)
                
                # Case of a persistent (i.e., never ending) task: 
                if self.env.persistent_task:
                    scores = [0 for i in range(self.env.num_flights)]

                # Check for possible external observation only after the environment reset (initialization):
                if it>0:
                    MainAux.manage_possible_external_obs(env=self.env, external_obs=self.args.external_obs, external_obs_dir=self.external_obs_dir, epoch=ep, it=it)
                
                # Action selection based on PPO: 
                if self.env.algorithm=='ppo':
                    actions, probs, vals = agent.choose_action(obs)
                # Action selection based on SAC:
                elif self.env.algorithm=='sac':
                    actions = agent.choose_action(obs)
                
                # The step() function manages the case in which the observation is considered during the learning phase:
                obs_, rews, done, info = self.env.step(actions)

                for i in range(self.env.num_flights):
                    # Store the rewards for all the agents:
                    scores[i] += rews[i]
                
                it += 1
                n_steps += 1

                '''
                When the actions are taken instantaneously, then the time elapsed for each iteration is computed a posteriori and it
                does not appear in the corresponding rendered scene:
                '''
                if self.env.scenario_action_delay:
                    actual_it_time = int((self.env.dt*it)/time_scaling_factor)
                else:
                    actual_it_time = ''
                

                """
                ______________________________________________________________________________________________________________________________
                The learning part (with the loss computation, rewards storing, ...) is performed only if an explicit clock has not been set,
                otherwise it is needed that the current learning iteration corresponds to a time instant
                that can be sampled by the selected system clock.
                ______________________________________________________________________________________________________________________________
                """
                # Modify the remember() arguments' passing if a new algorithm to be used is added:
                if self.env.algorithm=='ppo':
                    agent.remember(obs, actions, probs, vals, rews, done, self.env.flights, self.env.enode, self.env.learning_observation_delay, self.env.train_cfg.cumulative_reward)
                elif self.env.algorithm=='sac':
                    agent.remember(obs, actions, rews, obs_, done, self.env.flights, self.env.enode, self.env.learning_observation_delay, self.env.train_cfg.cumulative_reward)
                
                obs = obs_

                # Choose if learning from the stored experience or not:
                if (n_steps%N==0) and (not self.args.no_learn):
                    # Training case:
                    if self.env.algorithm=='ppo':
                        actor_loss, critic_loss, total_loss = agent.learn()
                        critic1_loss = None
                        critic2_loss = None
                        value_loss = None
                        alpha_loss = None
                    elif self.env.algorithm=='sac':
                        actor_loss, critic_1_loss, critic2_loss, critic_loss, value_loss, alpha_loss = agent.learn()
                        total_loss = None
                    # Hyperparameters tuning case:
                    if self.args.sweep:
                        # Modify here if a new algorithm to be used is added:
                        if self.env.algorithm=='ppo':
                            metrics['actor_loss'] = actor_loss
                            metrics['critic_loss'] = critic_loss
                            metrics['total_loss'] = total_loss
                        elif self.env.algorithm=='sac':
                            metrics['actor_loss'] = actor_loss
                            metrics['critic1_loss'] = critic1_loss
                            metrics['critic2_loss'] = critic2_loss
                            metrics['critic_loss'] = critic_loss
                            metrics['value_loss'] = value_loss
                            if alpha_loss!=None:
                                metrics['alpha_loss'] = alpha_loss

                    learn_iters += 1

                # Case in which the rendered animation has been saving (either every 'render_steps_saving' steps or when done=True): 
                if n_steps % self.render_steps_saving==0:
                    if not self.args.sweep:
                        if done:
                            frames.append(self.env.render(visible=self.visible, debug=self.debug)) # -> add the last frame when DONE
                        self.plot.save_frames_as_gif(frames=frames, ep=ep, it=it, actual_it=actual_it_time, time_unit=self.env.time_unit, done=done)
                    frames = []
                
                avg_agent_scores = np.mean(scores)
                score_history.append(np.mean(scores))
                avg_score = np.mean(score_history[-100:])

                # Action sigma decay (only if the action space is continuous):
                if self.train_cfg.action_space=='continuous':
                    self.train_cfg.sigma = self.train_aux.decayed_action_sigma(self.train_cfg.sigma, self.train_cfg.sigma_decay_rate, self.train_cfg.sigma_min)
                if avg_score > best_score:
                    best_score = avg_score
                    if self.args.sweep:
                        metrics['best_score'] = best_score
                    else:
                        agent.save_models()

                # ______________________________________________________________________________________________________________________________

                if learn_iters>=1:
                    # If the task is persistent, then store the loss and score values based on the number of iterations:
                    if self.env.persistent_task:
                        self.train_aux.tensorboard_writers(writer=self.writer, avg_agent_scores=avg_agent_scores, actor_loss=actor_loss,
                                                           critic_loss=critic_loss, flights=self.env.flights, num_flights=self.env.num_flights,
                                                           scores=scores, ep=it, train=self.args.train,
                                                           total_loss=total_loss, critic1_loss=critic1_loss, critic2_loss=critic2_loss,
                                                           value_loss=value_loss, alpha_loss=alpha_loss, usecase=usecase_for_plot)
                    # Hyperparameters tuning case:
                    if self.args.sweep:
                        metric = {metric_name: metrics[metric_name]}
                        wandb.log(metric)
                        if 'epoch_duration' in sweep_params:
                            # Convert the actual iteration time in hours (if needed): 
                            actual_it_time_h = self.secmin2hours(time=actual_it_time)
                            # End the current epoch if the current actual iteration time is equal to the current one set in the sweep configuration file:
                            if actual_it_time_h>=sweep_config.epoch_duration: # -> '>' is needed since 'sweep_config.epoch_duration' can be also a float value
                                # metric is uploaded on wandb after and outside iteration while loop
                                break
                # Case in which the losses have not been computed yet:
                else:
                    # PPO algorithm:
                    if self.env.algorithm=='ppo':
                        critic_loss = math.nan
                        actor_loss = math.nan
                        total_loss = math.nan
                    # SAC algorithm:
                    elif self.env.algorithm=='sac':
                        critic_loss = math.nan
                        actor_loss = math.nan
                        critic1_loss = math.nan
                        critic2_loss = math.nan
                        value_loss = math.nan

                # Terminal training output for a non-persistent task:
                if not self.env.persistent_task:
                    if not done:
                        end_keyword = '\r'
                    else:
                        end_keyword = '\n'
                    if self.env.algorithm=='ppo':
                        print('|Epoch: {:d} |Cur. avg score: {:.2f} |Last 100 eps avg score: {:.2f} |Critic Loss: {:.2f} |Actor Loss: {:.2f} |Total Loss: {:.2f} |It: {:d} |Learning Steps: {:.2f} |Info: {:s}'.format(ep, avg_agent_scores, avg_score, critic_loss, actor_loss, total_loss, it, learn_iters, info), end=end_keyword, flush=True)
                    elif self.env.algorithm=='sac':
                        # Individual Critic 1 an Critic 2 are not printed to show a clearer output:
                        print('|Epoch: {:d} |Cur. avg score: {:.2f} |Last 100 eps avg score: {:.2f} |Critic Loss (1+2): {:.2f} |Actor Loss: {:.2f} |Value Loss: {:.2f} |It: {:d} |Learning Steps: {:.2f} |Info: {:s}'.format(ep, avg_agent_scores, avg_score, critic_loss, actor_loss, value_loss, it, learn_iters, info), end=end_keyword, flush=True)
                # Terminal training output for a persistent task:
                else:
                    if self.env.algorithm=='ppo':
                        print('|It: {:d} |Cur. avg score: {:.2f} |Last 100 itrs avg score: {:.2f} |Critic Loss: {:.2f} |Actor Loss: {:.2f} |Total Loss: {:.2f} |Learning Steps: {:.2f} |Info: {:s}'.format(it, avg_agent_scores, avg_score, critic_loss, actor_loss, total_loss, learn_iters, info))
                    elif self.env.algorithm=='sac':
                        print('|It: {:d} |Cur. avg score: {:.2f} |Last 100 itrs avg score: {:.2f} |Critic Loss: {:.2f} |Actor Loss: {:.2f} |Critic1 Loss: {:.2f} |Critic2 Loss: {:.2f} |Value Loss: {:.2f} |Learning Steps: {:.2f} |Info: {:s}'.format(it, avg_agent_scores, avg_score, critic_loss, actor_loss, critic1_loss, critic2_loss, value_loss, learn_iters, info))
            
                for i, agent_i in enumerate(self.env.flights):
                    for j, agent_j in enumerate(self.env.flights):
                        if i!=j:
                            cur_dist = agent_i.position.distance(agent_j.position)
                            cur_episode_distances.append(cur_dist)

            # EPOCH/EPISODE ENDED:
            if learn_iters>=1:
                # If the task is NOT persistent, then store the loss and score values based on the number of episodes:
                if not self.env.persistent_task:
                    self.train_aux.tensorboard_writers(writer=self.writer, avg_agent_scores=avg_agent_scores, actor_loss=actor_loss,
                                                       critic_loss=critic_loss, flights=self.env.flights, num_flights=self.env.num_flights,
                                                       scores=scores, ep=ep, train=self.args.train,
                                                       total_loss=total_loss, critic1_loss=critic1_loss, critic2_loss=critic2_loss,
                                                       value_loss=value_loss, alpha_loss=alpha_loss, usecase=usecase_for_plot)                
                # Store the metric and log them on W&B if hyperparameters tuning is performed:
                if self.args.sweep:
                    metric = {metric_name: metrics[metric_name]}
                    wandb.log(metric)

            if not self.args.sweep:
                self.writer.flush()
                self.writer.close()

            n_steps_per_episode.append(it)
            # It could obviously cause 'mean of empty slice' for single-agent case:
            # avg_mobile_distance_per_episode.append(np.mean(cur_episode_distances)/ep)

            # close rendering
            self.env.close()

        # Save the first metric (i.e., number of iterations per episode):
        with open('metrics1.pickle', 'wb') as f:
            pickle.dump(n_steps_per_episode, f)

        '''
        # Save the second metric (i.e., average distance among UAVs for the current episode):
        with open('metrics2.pickle', 'wb') as f:
            pickle.dump(avg_mobile_distance_per_episode, f)
        '''

        f.close()
    
    def run(self) -> None:
        """
        Run either the training or the the hyperparameters tuning according to the options selected by the user.

        :param: None

        :return: None
        """
        
        # Case in which the sweep option is enabled:
        if self.args.sweep:
            yaml_sweep_file = 'settings/sweep_config.yaml'
            # Generate a dictionary with the sweeping configuration parameters based on the .yaml file containing the desired configuration parameters:
            sweep_config_dict = MainAux.yaml2dict(file=yaml_sweep_file)
            available_metrics = list(METRICS_PPO.keys()) if self.env.algorithm=='ppo' else list(METRICS_SAC.keys()) 
            # Store the desired reference metric for the hyperparameters tuning:
            metric_name = sweep_config_dict['metric']['name']
            assert metric_name in available_metrics, 'Invalid metric selected (-> {}) in sweep file {}! Select a valid metric among the following ones available: {}'.format(metric_name, yaml_sweep_file, available_metrics)

            # Agent initialization:
            agent = self.pretraining_setting()

            # Weight&Biases set up:
            sweep_id = wandb.sweep(sweep_config_dict, project='Sweep ' + NOW_FOLDER.replace(':', '_'))
            # Run the W&B Sweeping by calling the train() function: 
            wandb.agent(sweep_id, function=functools.partial(self.train, agent=agent, metric_name=metric_name, sweep_cng_dict=sweep_config_dict))
        # Case in which the sweep option is disabled:
        else:
            # Agent initialization:
            agent = self.pretraining_setting()
            print("\n\nTraining starts . . .")
            time.sleep(2)
            # Run the training:
            self.train(agent=agent)

        print("\n\nTraining ended.")
        time.sleep(2)