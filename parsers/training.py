import os
import shutil
import warnings
import time
import sys
import ast
from typing import Optional, Tuple, List, Dict, Union
import pathlib
from configparser import ConfigParser
from muavenv.global_vars import *

class TrainingConfig():
    """
    Config class
    """
    def __init__(self, settings_src='settings/training_parameters.ini', settings_dst='settings/scenario_parameters.ini', algorithm='ppo'):
        self.settings_src = settings_src + 'training_parameters.ini'
        if settings_dst!=None:
            self.settings_dst = settings_dst + 'training_parameters.ini'
        else:
            self.settings_dst = settings_dst
        saving_folder = LOGS_FOLDER
        saving_filename ='training_run.txt'
        saving_filepath = saving_folder + saving_filename # saving_folder + '/' + saving_filename
        if not os.path.isdir(saving_folder):
            os.mkdir(saving_folder)
        config = ConfigParser()
        l = config.read(self.settings_src)
        if l==[]:
            print(l)
            print("ERROR cannot load training settings init file")
            sys.exit(1)
        
        self.str_trues = ['true', 'True']
        self.str_falses = ['false', 'False']
        # Only store the letters associate with the algorithm used:
        self.algorithm = algorithm[:-1] if not algorithm.isalpha() else algorithm

        # [System]
        system = config['System']
        self.ct_de_paradigm = system['CT_DE']
        self.action_space = system['action_space']
        self.action_delay = system['action_delay']
        self.actions_size = int(system['actions_size'])
        self.observation_delay = system['observation_delay']
        self.observations = ast.literal_eval(system['observations'])
        self.actions = ast.literal_eval(system['actions'])
        self.ct_de_paradigm = self.str_to_bool(self.ct_de_paradigm)
        self.action_delay = self.str_to_bool(self.action_delay)
        self.observation_delay = self.str_to_bool(self.observation_delay)

        # [PPO Hyperparameters]
        hyperparams_ppo = config['PPO Hyperparameters']
        self.lr_ppo = float(hyperparams_ppo['lr'])
        self.gamma_ppo = float(hyperparams_ppo['gamma'])
        self.gae_lambda_ppo = float(hyperparams_ppo['gae_lambda'])
        self.policy_clip_ppo = float(hyperparams_ppo['policy_clip'])
        self.c_value_ppo = float(hyperparams_ppo['c_value'])
        self.c_entropy_ppo = float(hyperparams_ppo['c_entropy'])
        self.A_fc1_dims_ppo = int(hyperparams_ppo['A_fc1_dims'])
        self.A_fc2_dims_ppo = int(hyperparams_ppo['A_fc2_dims'])
        self.C_fc1_dims_ppo = int(hyperparams_ppo['C_fc1_dims'])
        self.C_fc2_dims_ppo = int(hyperparams_ppo['C_fc2_dims'])

        # [SAC Hyperparameters]
        hyperparams_sac = config['SAC Hyperparameters']
        self.lr1_sac = float(hyperparams_sac['lr1'])
        self.lr2_sac = float(hyperparams_sac['lr2'])
        self.alpha_sac = self.str_to_none(hyperparams_sac['alpha'])
        self.gamma_sac = float(hyperparams_sac['gamma'])
        self.tau_sac = float(hyperparams_sac['tau'])
        self.reward_scale_sac = self.str_to_none(hyperparams_sac['reward_scale'])
        self.A_fc1_dims_sac = int(hyperparams_sac['A_fc1_dims'])
        self.A_fc2_dims_sac = int(hyperparams_sac['A_fc2_dims'])
        self.C_fc1_dims_sac = int(hyperparams_sac['C_fc1_dims'])
        self.C_fc2_dims_sac = int(hyperparams_sac['C_fc2_dims'])
        self.V_fc1_dims_sac = int(hyperparams_sac['V_fc1_dims'])
        self.V_fc2_dims_sac = int(hyperparams_sac['V_fc2_dims'])

        # [General Hyperparameters]
        hyperparams_general = config['General Hyperparameters']
        self.batch_size = float(hyperparams_general['batch_size'])
        self.sigma = float(hyperparams_general['action_sigma'])
        self.sigma_decay_rate = float(hyperparams_general['action_sigma_decay_rate'])
        self.sigma_min = float(hyperparams_general['action_sigma_min'])
        self.n_episodes = int(hyperparams_general['n_episodes'])
        self.N = int(hyperparams_general['N'])

        # [Terminal Conditions]
        terminal_conditions = config['Terminal Conditions']
        self.task_ending = terminal_conditions['task_ending']
        self.task_success = terminal_conditions['task_success']
        self.time_failure = terminal_conditions['time_failure']
        self.battery_failure = terminal_conditions['battery_failure']
        self.perc_uavs_detecting_the_source = float(terminal_conditions['perc_uavs_detecting_the_source'])
        self.perc_uavs_discharged = float(terminal_conditions['perc_uavs_discharged'])
        self.task_ending = self.str_to_bool(self.task_ending)
        self.task_success = self.str_to_bool(self.task_success) 
        self.time_failure = self.str_to_bool(self.time_failure)
        self.battery_failure = self.str_to_bool(self.battery_failure)
        self.persistent_task = not (self.task_ending or self.task_success or self.time_failure or self.battery_failure)

        # [Reward]
        reward = config['Reward']
        self.cumulative_reward = reward['cumulative_reward']
        self.cumulative_rew_size = int(reward['cumulative_rew_size'])
        self.W_agent = float(reward['W_agent'])
        self.W_system = float(reward['W_system'])
        self.Wa = float(reward['Wa'])
        self.Wb = float(reward['Wb'])
        self.Wc = float(reward['Wc'])
        self.Wd = float(reward['Wd'])
        self.We = float(reward['We'])
        self.Wf = float(reward['Wf'])
        self.Wg = float(reward['Wg'])
        self.Wh = float(reward['Wh'])
        self.Wi = float(reward['Wi'])
        self.Wl = float(reward['Wl'])
        self.Wm = float(reward['Wm'])
        self.Wn = float(reward['Wn'])

        self.cumulative_reward = self.str_to_bool(self.cumulative_reward)

        self.validate_params()
        self.summary_log(saving_filepath)

    def str_to_bool(self, v: str) -> Union[bool, None]:
        if v in self.str_trues:
            return True
        elif v in self.str_falses:
            return False
        else:
            return None

    def str_to_none(self, v: str) -> Union[str, None]:
        if v=='none' or v=='None' or v==None:
            return None
        else:
            if v!='auto':
                return float(v)
            else:
                return v

    def validate_params(self) -> None:
        local_obs_present = False
        if self.ct_de_paradigm:
            for obs in self.observations:
                if obs in LOCAL_OBS_NAMES:
                    local_obs_present = True
                    break
        assert (self.ct_de_paradigm and local_obs_present) or (not self.ct_de_paradigm), 'You are trying to execute a CT/DE paradigm without considering any local observation:' \
                                                                                        + '\nyou must select at least one local observation (among "observations" feature in "training_parameters.ini") in order to execute a CT/DE paradigm!' \
                                                                                        + '\nList of local observations available: ' + str(LOCAL_OBS_NAMES) 
        warnings_raise = False
        if self.action_space=='discrete' and (self.sigma!=0. and self.sigma_min!=0.):
            warnings.warn('You chose a discrete action space and set action_sigma and min_sigma values for action selection to non-zero values:' \
                           + '\nfor discret action space, sigma will not be used.')
            print()
            warnings_raise = True
        if self.persistent_task and self.n_episodes>1:
            warnings.warn('You are trying to use a number of episodes larger than one for a persistent task:' \
                           + '\nwhen the task is persistent the episode never ends and then it will be a single infinite episodes.')
            print()
            self.n_episodes = 1
            warnings_raise = True
        if self.cumulative_reward==True and self.cumulative_rew_size==self.batch_size:
            warnings.warn('The step to BACKupdate the cumulative reward is equal to the ENode memory size, thus' \
                           + '\nnone of the cumulative rewards stored in the ENode memory will be actually BACKupdated!')
            print()
            warnings_raise = True
        if warnings_raise:    
            time.sleep(3)

        if self.algorithm=='sac':
            assert self.alpha_sac=='auto' or self.alpha_sac==None or isinstance(self.alpha_sac, int) or isinstance(self.alpha_sac, float), 'The entropy coefficient "alpha" can only be set equal to a number or to "None" or to "auto"!' 
            assert self.reward_scale_sac==None or isinstance(self.reward_scale_sac, int) or isinstance(self.reward_scale_sac, float), 'The reward scale can only be set equal to a number or to "None"!'
            assert (self.alpha_sac=='auto') or (self.alpha_sac==None and self.reward_scale_sac!=None) or (self.alpha_sac!=None and self.reward_scale_sac==None), 'You cannot neither explicitly assign both the entropy coefficient "alpha" and the reward scale nor avoid to assign one of them!'
        assert self.action_space=='discrete' or self.action_space=='continuous', 'The action space can be either discrete or continuous!'
        assert 0.<=self.perc_uavs_detecting_the_source<=1., 'The desired percentage of UAVs detecting the source must be between 0 and 1!'
        assert 0.<=self.perc_uavs_discharged<=1., 'The maximum percentage UAVs allowed to be discharged at the same time must be between 0 and 1!'
        assert self.task_success==True or self.task_success==False, 'Task success condition must be either True or False!'
        assert self.battery_failure==True or self.battery_failure==False, 'battery failure condition must be either True or False!'
        assert self.time_failure==True or self.time_failure==False, 'Time failure condition must be either True or False!'
        for obs in self.observations:
            assert obs in OBSERVATIONS_NAMES, 'Observation ' + '"' + str(obs) + '" is not available. Select the desired observations from the following observations list:' \
                                               + '\n' + str(OBSERVATIONS_NAMES)
        for action in self.actions:
            assert action in ACTIONS_NAMES, 'Action ' + '"' + str(obs) + '" is not available. Select the desired actions from the following observations list:' \
                                               + '\n' + str(ACTIONS_NAMES) 
        
        assert 0.<=self.W_agent<=1 and 0.<=self.W_system<=1 and 0.<=self.Wa<=1. \
               and 0.<=self.Wb<=1. and 0.<=self.Wc<=1. and 0.<=self.Wd<=1. \
               and 0.<=self.We<=1. and 0.<=self.Wf<=1. and 0.<=self.Wg<=1. \
               and 0.<=self.Wh<=1. and 0.<=self.Wi<=1. and 0.<=self.Wl<=1. \
               and 0.<=self.Wm<=1. and 0.<=self.Wn<=1., 'All the weights of the rewards must be values within [0,1]!' 
        if 'distance' not in self.actions and 'angle' not in self.actions:
            raise NotImplementedError('Action space with either only "distance" or only "angle" is not implemented!')

        assert ( (self.observation_delay==False) and (self.action_delay==False) ) or (self.cumulative_reward==False) or (self.cumulative_reward==True and self.cumulative_rew_size<=self.batch_size), 'When a delay is considered during the learning phase and also the cumulative reward is used, then "cumulative_rew_size" must be lower than or equal to the batch size (and obviously greater than 0), otherwise it will not be possible to backupdate properly the Enode memory size!'
        assert ( (self.action_delay==False) or (self.action_delay==True and 1<=self.actions_size<=self.batch_size) ), 'When the action delay is considered during the learning phase, then "actions_size" must be lower than or equal to the batch size (and obviously greater than 0), otherwise it will not be possible to properly know the past actions undertaken to consider them into the observation space!'

    def printkey(self, k: str) -> None:
        print("  %s = %r" %(k,self.__dict__[k]))

    def summary_log(self, file: pathlib.Path) -> None:
        print("\n============================= Training Parameters ==============================\n")
        
        attrs = vars(self)
        section = ''
        with open(file, 'w') as f:
            for key, value in attrs.items():
                if (key=='str_trues') or (key=='str_falses') or ( ('sac' in key) and (self.algorithm=='ppo') ) or ( ('ppo' in key) and (self.algorithm=='sac') ):
                    continue
                else:
                    if key=='CT_DE':
                        section = 'System\n'
                    elif key=='lr_ppo': #'alpha_ppo'
                        section = '\nPPO Hyperparameters\n'
                    elif key=='lr1_sac': #'alpha_sac'
                        section = '\nSAC Hyperparameters\n'
                    elif key=='batch_size':
                        section = '\nGeneral Hyperparameters\n'
                    elif key=='perc_uavs_detecting_the_source':
                        section = '\nTerminal Conditions\n'
                    elif key==('cumulative_reward'):
                        section = '\nReward\n'
                    else:
                        section = ''
                    if section!='':
                        print(section)
                        f.write(section)
                    if key=='observations' or key=='actions':
                        print('\t' + key)
                        f.write('\t' + str(key) + ':') if key=='observations' else f.write('\n\t' + "".join(str(key)) + ':')
                        for v in value:
                            print('\n\t\t', v)
                            f.write('\n\t\t' + str(v))
                    else:
                        print('\t' + key, value)
                        f.write('\t' + str(key) + ': ' + str(value) + '\n')
        
        if self.settings_dst!=None:
            # Copy the src into dst only if src file is different from dst file (this happens only if a usecase has not been selected):
            if self.settings_src!=self.settings_dst:
                shutil.copyfile(self.settings_src, self.settings_dst)

        print("\n================================================================================")
        print("Training parameters settings saved in " + str(file) + "\n\n")