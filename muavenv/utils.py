import os
import shutil
import copy
import numpy as np
import imageio
import argparse
import warnings
from matplotlib import animation
import matplotlib.pyplot as plt
from PIL import Image
from shapely.geometry import Polygon, Point, LineString, box
from typing import Optional, Tuple, List, Dict, Union, Any
from parsers.scenario import ScenarioConfig
from parsers.training import TrainingConfig
from parsers.external_info import *
from muavenv.global_vars import *
import muavenv.units as u
import pandas as pd
import pathlib
import yaml
import torch.utils.tensorboard

class Scaling():
    """
    Scaling class
    """
    def __init__(self):
        pass

    @staticmethod
    def min_max_normalization(x: float, x_min: float, x_max: float) -> float:
        """
        Min-max normalization (into [0,1] interval) for observation and action spaces.
        
        :param x: value to normalize
        :param x_min: minimum 'x' value
        :param x_max: maximum 'x' value

        :return: normalized 'x' value
        """
        return (x-x_min) / (x_max-x_min)

    @staticmethod
    def mean_normalization(x: float, x_min: float, x_max: float, x_mean: float) -> float:
        """
        Mean normalization for observation and action spaces.
        
        :param x: value to normalize
        :param x_min: minimum 'x' value
        :param x_max: maximum 'x' value
        :param x_mean: mean 'x' value

        :return: normalized 'x' value
        """
        return (x-x_mean)/(x_max-x_min)

    @staticmethod
    def standardization(x, x_mu, x_sigma):
        """
        Standarization for observation and action spaces.
        
        :param x: value to standarize
        :param x_mu: mu value of 'x'
        :param x_max: sigma value of 'x'

        :return: standardized 'x' value
        """
        return (x-x_mu)/x_sigma

    @staticmethod
    def min_max_denormalization(x_norm: float, x_min: float, x_max: float) -> float:
        """
        Reverse the min-max normalization (from [0,1] interval) to get the original values for observation and action spaces.
        
        :param x_norm: normalized value
        :param x_min: minimum not-normalized value
        :param x_max: maximum not-normalized value

        :return: de-normalized 'x_norm' value
        """
        return x_norm*(x_max-x_min) + x_min

    @staticmethod
    def mean_denormalization(x_norm: float, x_min: float, x_max: float, x_mean: float) -> float:
        """
        Reverse the mean normalization to get the original values for observation and action spaces.
        
        :param x_norm: normalized value
        :param x_min: minimum not-normalized value
        :param x_max: maximum not-normalized value
        :param x_mean: mean not-normalized value

        :return: de-normalized 'x_norm' value
        """
        return x_norm*(x_max-x_min) + x_mean

    @staticmethod
    def destandardization(x_standard: float, x_mu: float, x_sigma: float) -> float:
        """
        Reverse the standarization to get the original values for observation and action spaces.
        
        :param x_standard: standardized value
        :param x_mu: mu value of 'x_standard'
        :param x_max: sigma value of 'x_standard'

        :return: de-standardized 'x_standard' value
        """
        return x_standard*x_sigma + x_mu

class Plot():
    """
    Plotting class
    """
    def __init__(self, save_dir: Optional[Union[pathlib.Path, None]]=PLOT_FOLDER):
        self.save_dir = save_dir

    def source_path(self, pol: Polygon, walk: Dict, local_d: float, debug: Optional[bool] = False) -> None:
        """
        Static visualization of the audio source with numbered intra-hotspots and inner-hotspots.

        :param pol: operative polygon to plot
        :param walk: dictionary containing the path of the audio source
        :local_d: maximum distance that can be traveled by the audio source when it moves among inner-hotspots
        :param debug: show this static audio source visualization if enabled, otherwise it will be save withouth visualizing it

        :return: None
        """
        if self.save_dir==None:
            return None
        xy_pol = pol.boundary.coords.xy
        # Outer rectangular bounding box for the considered polygon:
        outer_rec = box(min(xy_pol[0]), min(xy_pol[1]), max(xy_pol[0]), max(xy_pol[1]))
        xy_outer_rec = outer_rec.boundary.coords.xy

        fig_name = 'SourceInitialPath'
        fig = plt.figure(fig_name)
        ax = fig.gca()
        ax.axis('equal') # -> by default Pyplot have more pixels along one axis over the other, thus the circles will rendered as an ellipse. ("ax.axis('equal')" let you to use the same amount of pixels for both the axes

        ax.plot(xy_pol[0], xy_pol[1], color='blue')
        ax.plot(xy_outer_rec[0], xy_outer_rec[1], color='red')
        
        hotspot_indeces = walk.keys()
        # Source Hotspots:
        hotspots_x = [walk[n]['hotspot'].x for n in hotspot_indeces]
        hotspots_y = [walk[n]['hotspot'].y for n in hotspot_indeces]
        ax.scatter(hotspots_x, hotspots_y, s=10, color='navy')
        # Inner spots for each hotspot:
        inners_x = [[i.x for i in walk[n]['inners']] for n in hotspot_indeces]
        inners_y = [[i.y for i in walk[n]['inners']] for n in hotspot_indeces]
        for i in range(len(inners_x)):
            ax.scatter(inners_x[i], inners_y[i], s=2, color='cornflowerblue')
        # Circular surface for the maximum covered area for inner spots around the related hotspot:
        inner_circles = [plt.Circle((hotspots_x[i-1], hotspots_y[i-1]), radius=local_d, color='cornflowerblue', alpha=0.15) for i in hotspot_indeces]
        for c in inner_circles:
            ax.add_patch(c)

        for i, txt in enumerate(hotspot_indeces):
            ax.annotate(txt, (hotspots_x[i], hotspots_y[i]), fontsize=10)
            for j, txt in enumerate(inners_x[i]):
                ax.annotate(j+1, (inners_x[i][j], inners_y[i][j]), fontsize=5)

        for i, val in enumerate(hotspot_indeces):
            ax.plot([p for p in hotspots_x], [p for p in hotspots_y], color='navy')    
        for i, val in enumerate(inners_x):
            ax.plot([p for p in inners_x[i]], [p for p in inners_y[i]], color='cornflowerblue')

        if debug:
            plt.show()
        plt.savefig(self.save_dir + fig_name + '.png')
        plt.close()

    def whole_env(self, pol: Polygon, POIs: List[Point], outer_p: List[Point], step: float, debug: Optional[bool] = False) -> None:
        """
        Static visualization of the environment discretization.

        :param pol: the operative polygon defining the environment
        :param POIs: list of the Point of Interests where UAVs can land
        :outer_p: list of the points outside the operative polygon and generated during the discretization process
        :param step: radius defining the circles between a POI and its next one
        :param debug: show this static audio source visualization if enabled, otherwise it will be save withouth visualizing it  
        
        :return: None 
        """
        if self.save_dir==None:
            return None
        xy_pol = pol.boundary.coords.xy
        # Outer rectangular bounding box for the considered polygon:
        outer_rec = box(min(xy_pol[0]), min(xy_pol[1]), max(xy_pol[0]), max(xy_pol[1]))
        xy_outer_rec = outer_rec.boundary.coords.xy

        fig_name = 'EnvironmentDiscretization'
        fig = plt.figure(fig_name)
        ax = fig.gca()
        ax.axis('equal') # -> by default Pyplot have more pixels along one axis over the other, thus the circles will rendered as an ellipse. ("ax.axis('equal')" let you to use the same amount of pixels for both the axes
        
        ax.plot(xy_pol[0], xy_pol[1], color='blue')
        ax.plot(xy_outer_rec[0], xy_outer_rec[1], color='red')

        # Inner centroids and circles:
        ax.scatter([p.x for p in POIs], [p.y for p in POIs], s=2, color='cornflowerblue')
        inner_circles = [plt.Circle((p.x, p.y), radius=step, color='cornflowerblue', alpha=0.15) for p in POIs]
        for c in inner_circles:
            ax.add_patch(c)
        # Outer centroids and circles:
        ax.scatter([p.x for p in outer_p], [p.y for p in outer_p], s=2, color='orange')
        outer_circles = [plt.Circle((p.x, p.y), radius=step, color='orange', alpha=0.15) for p in outer_p] 
        for c in outer_circles:
            ax.add_patch(c)

        if debug:
            plt.show()
        plt.savefig(self.save_dir + fig_name + '.png')
        plt.close()

    def save_frames_as_gif(self, frames: List[np.array], ep: int, it: int, actual_it: int, time_unit: str, done: bool, path=PLOT_FOLDER):
        """
        Store the environment animation as a .gif.

        :param frames: list of the frames of the environment animation expressed as np.array
        :param ep: current epoch
        :param it: current iteration
        :param actual_it: real current time associated with the current iteration 'it'
        :param time_unit: current unit time measure chosen by the user
        :param done: boolean indicating whether the episode is ended (True) or not (False)
        :param path: directory path where static and dynamic images are stored

        return: None
        """
        done_str = '_done' if done==True else '_not_done' 

        ep_folder = self.save_dir + '/episode_' + str(ep) + '/'
        if not os.path.isdir(ep_folder):
            os.makedirs(ep_folder)
        frames = [Image.fromarray(frame) for frame in frames]
        gif_name = '_it ' + str(it) + ': ' + str(actual_it) + time_unit + done_str + '_.gif'
        frames[0].save(ep_folder + gif_name, save_all=True, append_images=frames[1:], duration=50, loop=0)

class TrainAux():
    """
    TrainAux class
    """

    def __init__(self):
        pass

    @staticmethod
    def decayed_action_sigma(sigma: float, sigma_decay_rate: float, min_sigma: float) -> float:
        """
        Decay the sigma value used for the actions.

        :param sigma: sigma value for the actions
        :param sigma_decay_rate: decay rate to be used to decrease 'sigma'
        :param min_sigma: minimum threshold value that can be reached by 'sigma'

        :return: new decayed value of 'sigma'
        """
        sigma -= sigma_decay_rate
        sigma = round(sigma, 4)
        if (sigma <=min_sigma):
            sigma = min_sigma
            print("setting actor output sigma to min_sigma: ", sigma)
        else:
            print("setting actor output sigma to: ", sigma)
        
        return sigma

    @staticmethod
    def tensorboard_writers(writer: "SummaryWriter", avg_agent_scores: float, actor_loss: float,
                            critic_loss: float, flights: List["Flight"],
                            num_flights: int, scores: List[float], ep: int, train: bool=True, total_loss: Optional[Union[float, None]]=None,
                            critic1_loss: Optional[Union[int, None]]=None, critic2_loss: Optional[Union[int, None]]=None,
                            value_loss: Optional[Union[int, None]]=None, alpha_loss: Optional[Union[int, None]]=None, usecase: Optional[bool] = False) -> None:
        """
        Add scalar values to visualize in Tensorboard.

        :param writer: Tensorboard "SummaryWriter" object
        :param avg_agent_scores: average scores of the agents
        :param actor_loss: actor loss function value
        :param critic_loss: critic loss function value
        :param flights: agents (i.e., UAVs)
        :param num_flights: number of the agents
        :param scores: list of the scores of each single agent
        :param ep: number of the current episode
        :param train: a boolean indicating whether the train or the test session is taking place
        :param total_loss: total loss function value
        :param critic1_loss: first critic loss function value (if any)
        :param critic2_loss: second critic loss function value (if any)
        :param value_loss: value network's loss function value (if any)
        :param alpha_loss: alpha (entropy coefficient) loss value (if any)
        :param usecase: boolean indicating if we are considering a predefined use case or not

        :return: None 
        """
        if train:
            loss_folder_name = "Train/"
        else:
            if not usecase:
                loss_folder_name = "Test/"
            else:
                loss_folder_name = "Usecase/"

        writer.add_scalar("Score/AvgAmongAllAgents", avg_agent_scores, ep)                
        writer.add_scalar(loss_folder_name + "ActorLoss", actor_loss, ep)
        writer.add_scalar(loss_folder_name + "CriticLoss", critic_loss, ep)
        if total_loss!=None:
            writer.add_scalar(loss_folder_name + "TotalLoss", total_loss, ep)
        if critic1_loss!=None:
            writer.add_scalar(loss_folder_name + "Critic1Loss", critic1_loss, ep)
        if critic2_loss!=None:
            writer.add_scalar(loss_folder_name + "Critic2Loss", critic2_loss, ep)
        if value_loss!=None:
            writer.add_scalar(loss_folder_name + "ValueLoss", value_loss, ep)
        if alpha_loss!=None:
            writer.add_scalar(loss_folder_name + "AlphaLoss", value_loss, ep)

        for i in range(num_flights):
            writer.add_scalar("Score/Agent_{:d}".format(flights[i].ID), scores[i], ep)

        #writer.add_graph()

class MainAux():
    """
    MainAux class
    """

    def __init__(self):
        self.algorithm_folder = 'ppo/'
        #pass

    def terminal_parser(self) -> Tuple:
        """
        Parse the arguments entered into the Terminal.
        
        :param: None

        :return args: a tuple containing all the arguments manually inserted by the user into the terminal
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--train", action='store_true',
                            help="Training phase enabled.")
        parser.add_argument("--test", action='store_true',
                            help="Testig phase enabled")
        parser.add_argument("--no_learn", action='store_true',
                            help="Learning phase disabled")
        parser.add_argument("--sweep", action='store_true',
                            help="Hyperparameters sweep enabled")
        parser.add_argument("--external_obs", action='store_true',
                            help="External observation feeding-in procedure enabled (position expressed in (lat,lon) coordinates).")
        parser.add_argument("-FIR_file", type=str, default=None,
                            help="Insert the path of the .csv or .geojson file containing the info of the Flight Information Region (FIR) (expressed in (lat,lon)) to use for the current run.")
        parser.add_argument("-FIR_ID", type=str, default=None,
                            help="Insert the ID of the Flight Information Region (FIR) (expressed in (lat,lon)) to select a specific FIR (if more than one is present) contained inside the .csv file in -FIR_file.")
        parser.add_argument("-model_path", type=str, default=None,
                            help="Insert the path of the (trained) model to use for the current run (use only if --test in enabled).")
        parser.add_argument("-test_scenario_params_path", type=str, default=None,
                            help="Insert the path of the file containing the parameters of the desired scenario to test based on a learned policy (to be used only if --test is enabled and if a test scenario different from that used during learning is desired). If it is not specified, then the testing scenario parameters will be automatically set equal to the ones used for the trained model.")
        parser.add_argument("--external_communication", action='store_true',
                            help="Enable a communication client/server with another machine to get external local observations of the agents: if enabled, then also --external_obs will be automatically enabled.")
        parser.add_argument("--ppo", action='store_true',
                            help="Use Proximal Optimization Policy (PPO) algorithm.")
        parser.add_argument("--sac", action='store_true',
                            help="Use Soft Actor-Critic (SAC) algorithm.")
        parser.add_argument("--usecase1", action='store_true',
                            help="Run the use case 1.")
        parser.add_argument("--usecase2", action='store_true',
                            help="Run the use case 2.")

        args = parser.parse_args()

        MainAux.validate_parsing(args=args)

        return args

    @staticmethod
    def validate_parsing(args: Tuple) -> None:
        """
        Validate the argument enetered into the terminal.
        
        :param args: arguments entered into the terminal

        :return: None
        """
        if args.external_communication:
            args.external_obs = True
            warnings.warn("\n\n--external_communication option automatically enables --external_obs option!\n\n")
            print()
            time.sleep(3)
        if args.test and args.test_scenario_params_path==None:
            warnings.warn('Testing scenario parameters have not being speficied: the same scenario parameters used during learning process will be used!')
            print()
            time.sleep(3)

        if args.ppo==False and args.sac==False:
            if (not args.usecase1) and (not args.usecase2):
                warnings.warn('No algorithm selected: PPO algorithm will be used by default!')
                print()
                time.sleep(3)

        # Modify the first assert here below if you implement a new algorithms:
        if args.test:
            if args.ppo==True or args.sac==True:
                assert False, 'You cannot select an algorithm for a test: the policy has been already learned and the algorithm used will be the same as that used during the learning'
        assert ((args.ppo==True) and (args.sac==False)) or ((args.ppo==False) and (args.sac==True)) or (((args.ppo==False) and (args.sac==False))), 'You selected more than one algorithm: only one algorithm can be used!'
        assert (args.train==True and args.test==False or args.sweep==False) or (args.train==False and args.test==True and args.sweep==False) or (args.train==False and args.test==False and args.sweep==True), 'You must select either --train or --test or --sweep, neither all nor none of them!'
        assert (args.sweep==True and args.external_obs==False) or (args.sweep==False and args.external_obs==True) or (args.sweep==False and args.external_obs==False), 'If --sweep option is enabled, then --external_obs muist be disabled and viceversa!' 
        assert (args.external_obs==True and args.FIR_file!=None) or (args.external_obs==False and args.FIR_file!=True) or (args.external_obs==False and args.FIR_file==None), 'If --external_obs is enabled, then you need to use also a -FIR_file expressed in (lat,lon) coordinates!'
        assert (args.test==True and args.model_path!=None) or (args.test==False and args.model_path==None), 'When --test is enabled, you must select also a model_path associated (and viceversa) with a pretrained model. When --test is disabled instead, you cannot set any -model_path!'
        assert ((args.train==True or args.sweep==True or args.usecase1 or args.usecase2) and (not args.no_learn)) or (args.test), '--no_learn flag is used to skip the learning phase, and thus it can be enabled only if a test (flag --test) is being performed!'
        assert (args.test==True and (args.test_scenario_params_path==None or args.test_scenario_params_path!=None) or (args.test==False and args.test_scenario_params_path==None)), 'If --test is not enabled, a file containing the scenario parameters for test case cannot be selected!'
        assert (args.FIR_file!=None and args.FIR_ID!=None) or (args.FIR_file==None and args.FIR_ID==None), 'You must set either both -FIR_file and the desired FIR -FIR_ID or none of them (the latter case will generate a random FIR in (x,y) coorinates)!'
        # (args.external_obs==True and args.FIR_file!=None and args.FIR_ID==None)
        assert ( (args.usecase1 or args.usecase2) and \
                  args.train==False and args.test==False and args.no_learn==False and args.sweep==False and \
                  args.external_obs==False and args.FIR_file==None and args.FIR_ID==None and args.model_path==None and \
                  args.test_scenario_params_path==None and args.external_communication==False and \
                  args.ppo==False and args.sac==False) or (not args.usecase1 and not args.usecase2), 'Whether a use case is selected, then the other available flag must not be selected as the use case contains a predefined setting and flag!' 

    def folders_generation(self, args) -> Tuple[str]:
        """
        Generate the directory structure for the current run based on the arguments entered into the terminal by the user.

        :param args: arguments entered into the terminal

        :return logs_dir: log directory where to mainly save TFEvents file for Tensorboard visualization)
        :return model_save_dir: directory where to save the trained model
        :return model_load_dir: directory where to load (if is this the case) the trained model
        :return plot_dir: directory where to save static and dynamic environment visualization
        :return settings_src_dir: source directory for the current scenario and training parameters settings
        :return settings_dst_dir: directory where to save the scenario and training parameters settings for the current run
        :return external_obs_dir: directory where to save info about the external observations history file (if is this the case)
        """
        def create_dirs(directory: 'str') -> None:
            if not os.path.isdir(directory):
                os.makedirs(directory)

        if args.train:
            root_dir = ROOT_FOLDER + 'train/'     
            settings_src_dir = 'settings/'
            model_load_dir = None
        elif args.test:
            root_dir = ROOT_FOLDER + 'test/'
            model_load_dir = args.model_path
            if args.test_scenario_params_path==None:
                settings_src_dir = args.model_path
            else:
                settings_src_dir = args.test_scenario_params_path
        elif args.sweep:
            root_dir = ROOT_FOLDER + 'sweep/'
            settings_src_dir = 'settings/' 
            model_load_dir = None
        # Use cases:
        else:
            # Use PPO algorithm for the provided use cases:
            args.ppo = True 
            args.sac = False
            self.algorithm_folder = 'ppo/'
            model_load_dir = None
            if args.usecase1:
                cur_usecase_folder = 'usecase1/'
                root_dir = ROOT_FOLDER + cur_usecase_folder
                settings_src_dir = 'usecases_files/' + cur_usecase_folder
            elif args.usecase2:
                cur_usecase_folder = 'usecase2/'
                root_dir = ROOT_FOLDER + cur_usecase_folder
                settings_src_dir = 'usecases_files/' + cur_usecase_folder

        if args.sac:
            self.algorithm_folder = 'sac/' 

        root_dir += self.algorithm_folder + NOW_FOLDER + '/'
        logs_dir = root_dir + LOGS_FOLDER
        
        if args.test or args.train:
            model_save_dir = root_dir + MODEL_FOLDER 
            plot_dir = root_dir + PLOT_FOLDER
            settings_dst_dir = model_save_dir
            
            create_dirs(model_save_dir)
            create_dirs(plot_dir)
        
        else:
            if args.usecase1 or args.usecase2:
                model_save_dir = root_dir + MODEL_FOLDER 
                plot_dir = root_dir + PLOT_FOLDER
                # Override the 'src' and 'dst' settings as the use cases are predefined and hence 'src' and 'dst' are the same:
                settings_dst_dir = model_save_dir #settings_src_dir 

                create_dirs(model_save_dir)
                create_dirs(plot_dir)
            else:    
                plot_dir = None
                model_save_dir = None
                settings_dst_dir = None

        if args.external_obs:
            external_obs_dir = root_dir
        else:
            external_obs_dir = None

        create_dirs(logs_dir)
        
        return logs_dir, model_save_dir, model_load_dir, plot_dir, settings_src_dir, settings_dst_dir, external_obs_dir

    @staticmethod
    def yaml2dict(file: pathlib.Path) -> Dict:
        """
        Convert a .yaml into a dictionary.

        :param file: .yaml file

        :return config_dict: dictionary made up by the parameters set in 'file' 
        """
        with open(file) as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)

            return config_dict

    @staticmethod
    def ask_for_external_obs_input(external_communication: Optional[bool] = False) -> pd.DataFrame:
        """
        Put together classes and methods to ask the user (if needed) to enter the external observation filepath.
        
        :param external_communication: boolean stating if external_communication option is enabled or not
        
        :return df: DataFrame contaning the data of the external observation file
        :return ex_info.obs_file: file containing the external obseravtion file 
        """
        ex_info = ExternalInfo(external_communication=external_communication)
        df = ex_info.obs_file_read()
        
        return df, ex_info.obs_file

    @staticmethod
    def manage_possible_external_obs(env: "Environment", external_obs: bool, external_obs_dir: pathlib.Path, epoch: int, it: int) -> None:
        """
        Manage the case in which the external oncervation option is enabled.

        :param env: the OpenAI Gym environment used
        :param external_obs: boolean stating if the external observation option is enabled or not
        :param external_obs_dir: directory where to save the external observations history
        :param epoch: current epoch
        :param it: current iteration

        :return: None
        """
        if external_obs:
            df_external_obs, external_obs_file = MainAux.ask_for_external_obs_input(external_communication=env.external_communication)
            env.external_obs = copy.deepcopy(df_external_obs)

            df_external_obs['Epoch'] = epoch
            df_external_obs['Iteration'] = it
            ext_obs_history_file = external_obs_dir + 'external_obs_history.csv'
            # Case in which the file has not been created yet: 
            os.path.exists(ext_obs_history_file)
            if not os.path.exists(ext_obs_history_file):
                shutil.copyfile(src=external_obs_file, dst=ext_obs_history_file)
                # Add columns 'Epoch' and 'Iteration' to the file containing the extenal observation history:
                df_external_obs.to_csv(path_or_buf=ext_obs_history_file, index=False)
            else:
                df_external_obs.to_csv(path_or_buf=ext_obs_history_file, mode='a', index=False, header=False)

            '''
            In case of external communication, delete the obs_file after reading it
            (for delayed actions it could be useful to overwrite it to take into account directly also some communication delays):
            '''
            if env.external_communication:
                os.remove(external_obs_file)

class DefAux:

    def __init__(self):
        pass

    @staticmethod
    def memory_update(memory: Dict, max_size: int, aoi: Union[int, float], orig_new_v: Any, new_v: Any, env: "Environment", global_obs_update: Optional[bool] = False, data_losses: Optional[List]=[], observation_loss: Optional[bool]=False) -> None:
        """
        It can be used to update the observation memory for both the ENode (global observations memory) and the UAVs (local observations memory).
        
        :param memory: the dictionary associated with the memory either of the ENode or a UAV
        :param max_size: the maximum number (i.e., the batch_size) of observation which can be stored in the selected memory
        :param aoi: the Age of Information of the 'new_v', which is always equal to the time elapsed when the 'memory' is being updated
                    (AoI is always equal to the elapsed time since either the observation is taken and stored directly on the ENode-side
                     or takend and stored directly on the UAV-side)
        :param orig_new_v: the value associated with the new ORIGINAL and not processed observation (i.e. all the obs from all the UAVs) to store
        :param new_v: the value associated with the new NORMALIZED observation to store
        :param env: the variable associated with the envirorment used
        :param global_obs_update: boolean value indicating whetaher to BACKupdate or not the global memory

        :return: None
        """

        '''
        REMARK: the local sensing of each UAV is assumed to be negligible w.r.t. the delay due to the communication, and hence
        each UAV can always get access to its updated local information in real-time
        '''

        used_memory_size = len(memory)
        all_stored_AoIs = list(memory.keys())
        if all_stored_AoIs!=[]:
            # The oldest observation time stored is the one associated with the lowest 'elapsed time'
            oldest_observation_t = min(all_stored_AoIs)
        else:
            # A default negative (i.e., not possible) value is used for the 'oldest observation time' if there are no observations stored:
            oldest_observation_t = -1
        
        def memory_backupdate(memory: Dict, aoi: Union[int, float], new_v: Any, env: "Environment", data_losses: Optional[List]=[], observation_loss: Optional[bool]=False) -> None:
            """
            Perform the backupdate process on the observations 

            The arguments of this method have the same meaning as those in the outher method 'memory_update(...)'

            :return: None 
            """

            '''
            ----------------------------------------------------------------------------------------------------------------------------------
            The original observation are stored for the first time in 'update_broadcast_obs(...)', which is called before this method, and
            hence it is impossible that 'aoi' is not in memory (as it has been previously added in 'update_broadcast_obs'). Thus, here, the
            old original observations are never added, but only used to compute the new normalized observations (if the observation delay
            must be taken into account during the learning process). Analogous reasoning can be made for the normalized observations as they
            directly derive from the original ones and thus they can be computed only after (and based on) the original observations.
            In conclusion, while the original observations related to the old stored ones are computed for the first time in
            'update_broadcast_obs(...)', the normalized observations (still related to the old observations) are computed only here based on
            the update of the original observations received in 'update_broadcast_obs(...)'.
            The CURRENT normalized observation is computed instead in 'env.py'.
            ----------------------------------------------------------------------------------------------------------------------------------
            '''
            
            def bearing(A: Point, B: Point) -> float:
                dx = A.x - B.x
                dy = A.y - B.y
                compass = math.atan2(dy, dx)
                return (compass + u.circle) % u.circle

            def drift(angle1: float, angle2: float) -> float:
                drift = angle1 - angle2

                if drift > math.pi:
                    return -(u.circle - drift)
                elif drift < -math.pi:
                    return (u.circle + drift)
                else:
                    return drift

            def enode_contains_flight_info(ID: float, memory: Dict) -> bool:
                """
                Check if the ENode contains info related to the selected 'ID'.

                :param ID: ID of the selected agent
                :param memory: memory of the ENode

                :return True/False: a boolean indicating if the the ENode contains info related to the selected agent 'ID'  
                """
                if ID in memory.keys():
                    return True
                else:
                    return False

            n_uavs = len(env.flights)

            '''
            Backupdate the original observation
            (the Enode memory associated with the current local observation has been already updated,
            thus I can assign to it the local observation to be backupdated):
            '''

            old_broadcast_obs = copy.deepcopy(memory[aoi]['orig_obs'])
            old_broadcast_obs_copy = copy.deepcopy(old_broadcast_obs)
            old_last_shot_feature = old_broadcast_obs[LAST_SHOT_KEY]
            old_source_time = old_last_shot_feature[0]
            old_source_pos = old_last_shot_feature[1]
            old_global_obs_AOI = old_broadcast_obs[AOI_KEY]
            old_current_spotting_agents = old_last_shot_feature[2]
            old_current_n_spotting_agents = len(old_current_spotting_agents)
            old_current_n_spotting_agents = Scaling.min_max_normalization(x=old_current_n_spotting_agents, x_min=0, x_max=env.n_uavs_detecting_the_source)
            
            old_elapsed_source_time = old_global_obs_AOI - old_source_time
            old_elapsed_source_time = np.clip(old_elapsed_source_time, a_min=TIME_MIN, a_max=env.max_time)
            
            old_src_dists_among_spotting_agents = []

            '''
            Here we have some redundant code as it is very similar to the one inside 'observation(...)' in 'env.py'.
            for the time being it will be kept as it is, but further on it should be better to implement only a single method to
            be used here and in 'env.py'.
            '''

            for IDi, stored_info in old_broadcast_obs.items():
                # Skip the keys not associated with any ID of a UAV:
                if (IDi==LAST_SHOT_KEY) or (IDi==AOI_KEY):
                    continue

                # Data Loss Handling:
                if observation_loss and IDi in data_losses:
                    memory_aois = list(memory.keys())
                    memory_aois.sort()
                    prev_aoi_idx = memory_aois.index(aoi)-1
                    prev_aoi = memory_aois[prev_aoi_idx]
                    for f in env.flights:
                        if f.ID == IDi:
                            cur_f = f
                            cur_f_idx = env.flights.index(cur_f)
                            break
                    prev_orig_obs_at_prev_aoi = memory_aois[prev_aoi]['orig_obs'][IDi]
                    prev_norm_obs_at_prev_aoi = memory_aois[prev_aoi]['norm_obs'][IDi]
                    prev_info_buffer_at_prev_aoi = memory_aois[prev_aoi]['source_info_buffer'][IDi]
                    memory[aoi]['orig_obs'][IDi] = prev_orig_obs_at_prev_aoi
                    memory[aoi]['norm_obs'][cur_f_idx] = prev_norm_obs_at_prev_aoi
                    memory[aoi]['source_info_buffer'][IDi] = prev_info_buffer_at_prev_aoi

                    return

                if IDi in old_current_spotting_agents:
                    old_info_i = old_broadcast_obs[IDi]
                    old_stored_pos_i = old_info_i[0][1]

                    '''
                    It is not needed to check whether 'source_dist' is NaN or not as we are computing it after checking that the current
                    flight ID is inside the list of the agents that have already spotted the source.
                    '''
                    old_source_dist = old_stored_pos_i.distance(old_source_pos)
                    old_source_dist = np.clip(old_source_dist, a_min=DIST_MIN, a_max=env.max_d)
                    old_source_dist = Scaling.min_max_normalization(x=old_source_dist, x_min=DIST_MIN, x_max=env.max_d)

                    old_src_dists_among_spotting_agents.append(old_source_dist)

            if n_uavs!=1:
                '''
                'old_avg_src_dist_among_spotting_agents' normalization is not required as the value of the distance
                of each agent w.r.t. the source has been already normalized
                '''
                if old_src_dists_among_spotting_agents!=[]:
                    old_avg_src_dist_among_spotting_agents = np.mean(old_src_dists_among_spotting_agents)
                else:
                    old_avg_src_dist_among_spotting_agents = 0. # -> default value for empty list

            for IDi, stored_info in old_broadcast_obs.items():
                # Skip the keys not associated with any ID of a UAV:
                if (IDi==LAST_SHOT_KEY) or (IDi==AOI_KEY):
                    continue

                for f in env.flights:
                    if f.ID == IDi:
                        cur_f = f
                        cur_f_idx = env.flights.index(cur_f)
                        break

                # if some package loss/delay is applied, than ENode could not contains infos related to some ID (i.e., agent):
                if enode_contains_flight_info(ID=IDi, memory=old_broadcast_obs):
                    if IDi in old_current_spotting_agents:
                        old_source_spotted = 1
                    else:
                        old_source_spotted = 0
                    old_source_spotted = Scaling.min_max_normalization(x=old_source_spotted, x_min=0, x_max=1)

                    old_coverage_area_i = env.enode.compute_coverage_area(fi=cur_f, pol=env.airspace.cartesian_polygon, aoi=aoi)
                    old_coverage_area_i = Scaling.min_max_normalization(x=old_coverage_area_i, x_min=0., x_max=math.pi*pow(cur_f.footprint_radius, 2)) # DEVI TROVARE IL MODO DI CAPIRE COME USARE IL footprint_radius dell'UAV selezionato

                    '''
                    It should be taken into account that considering the starting position could lead to get a 0
                    distance between 2 agents who are swapping their position if one of them has already arrived at
                    its target position and the other one not yet: maybe it could be changed in future.
                    '''

                    # Position and agent features processing based on the global observations available at the ENode memory:
                    old_info_i = old_broadcast_obs[IDi]
                    old_stored_pos_i = old_info_i[0][1] # -> we take the start (and not the target) position 

                    old_agent_features_i = old_info_i[2] # get the agent features (for the time being it is (self.track, self.battery))
                    old_track_agent_i = old_agent_features_i[0]
                    if env.energy_constraint:
                        old_battery_agent_i = old_agent_features_i[1]
                        old_battery_agent_i = Scaling.min_max_normalization(x=old_battery_agent_i, x_min=0., x_max=100)

                    '''
                    We can use the same info from the current local observation in 'broascast_obs' as the local one
                    is the one that has been currently processed even if it has arrived with a delay (and thus we can take local
                    observation from 'broadcast_obs'). Hence, local observations that do not depend on global ones (e.g. the battery level)
                    are obviously the same which have been already computed.
                    '''
                    if not math.isnan(old_source_time): 
                        old_source_dist = old_stored_pos_i.distance(old_source_pos)
                        old_source_bearing = bearing(old_source_pos, old_stored_pos_i)
                        old_source_drift = drift(old_source_bearing, old_track_agent_i)
                    else:
                        old_last_spotting_UAV = 0
                        old_elapsed_source_time = env.max_time
                        old_source_dist = env.max_d
                        old_source_bearing = BEARING_MAX
                        old_source_drift = DRIFT_MAX
                        pass

                    old_source_dist = np.clip(old_source_dist, a_min=DIST_MIN, a_max=env.max_d)
                    old_source_dist = Scaling.min_max_normalization(x=old_source_dist, x_min=DIST_MIN, x_max=env.max_d)
                    old_source_bearing = Scaling.min_max_normalization(x=old_source_bearing, x_min=BEARING_MIN, x_max=BEARING_MAX)
                    old_source_drift = Scaling.min_max_normalization(x=old_source_drift, x_min=DRIFT_MIN, x_max=DRIFT_MAX)
                    old_elapsed_source_time = Scaling.min_max_normalization(x=old_elapsed_source_time, x_min=TIME_MIN, x_max=env.max_time)

                    # The agent distances are processed based on the global observations available at the ENode Memory: 
                    old_agent_distances_i = []
                    old_not_spotting_agents_distances = []

                    for IDj, stored_info_j in old_broadcast_obs_copy.items():
                        # Skip the keys not associated with any ID of a UAV:
                        if (IDj==LAST_SHOT_KEY) or (IDj==AOI_KEY):
                            continue

                        if IDj!=IDi:
                            # If some package loss/delay is applied, than ENode could not contains infos related to some ID (i.e., agent):
                            if enode_contains_flight_info(ID=IDj, memory=old_broadcast_obs_copy):
                                old_info_j = old_broadcast_obs_copy[IDj]
                                old_stored_pos_j = old_info_j[0][1]

                                old_dist_ij = old_stored_pos_i.distance(old_stored_pos_j)
                                old_dist_ij = np.clip(old_dist_ij, a_min=DIST_MIN, a_max=env.max_d)
                                old_dist_ij = Scaling.min_max_normalization(x=old_dist_ij, x_min=DIST_MIN, x_max=env.max_d)
                                old_agent_distances_i.append(old_dist_ij)
                                if (IDi not in old_current_spotting_agents) and (IDj not in old_current_spotting_agents):
                                    old_not_spotting_agents_distances.append(old_dist_ij)
                    
                    if n_uavs!=1:
                        old_avg_agent_distances_i = np.mean(old_agent_distances_i)
                        '''
                        'old_avg_not_spotting_agents_distances' normalization is not required as
                        the value of the distace of each agent w.r.t. the source has been already normalized:
                        '''
                        if old_not_spotting_agents_distances!=[]:
                            old_avg_not_spotting_agents_distances = np.mean(old_not_spotting_agents_distances)
                        else:
                            old_avg_not_spotting_agents_distances = 0.

                obs = []

                # Still some redundant code will follow: for the time being it will be kept as it is.

                if 'source distance' in env.selected_observations: 
                    obs.append(old_source_dist)
                if 'source bearing' in env.selected_observations:
                    obs.append(old_source_bearing)
                if 'source drift' in env.selected_observations:
                    obs.append(old_source_drift)
                if 'source time' in env.selected_observations: 
                    obs.append(old_elapsed_source_time)
                if 'battery' in env.selected_observations:
                    if env.energy_constraint:
                        '''
                        Only the global battery observation depends on the number of the agents (it could be used
                        the avg battery level of the system to remove the dependency on the number of the agents):
                        '''
                        obs.append(old_battery_agent_i)
                        # The local battery level update is not needed since it was with no delay in the past as it is local 
                # If you want to use get access to all the agents' distances, then be aware that this is dependent on the number of the agents:
                if 'agents distances' in env.selected_observations:
                    obs += old_agent_distances_i #obs.append(old_agent_distances_i)
                if 'single-agent coverage area' in env.selected_observations:
                    obs.append(old_coverage_area_i)
                if 'AVG agent-all distance' in env.selected_observations:
                    obs.append(old_avg_agent_distances_i)
                if 'AVG agent-not_spotting_agents distance' in env.selected_observations:
                    obs.append(old_avg_not_spotting_agents_distances)
                if 'AVG spotting_agents-source distance' in env.selected_observations:
                    obs.append(old_avg_src_dist_among_spotting_agents)
                if 'current N spotting agents' in env.selected_observations:
                    obs.append(old_current_n_spotting_agents)
                if  'current agent spotted the source' in env.selected_observations:
                    # The local source spotted/not spotted "backupdate" is not needed since it was with no delay in the past as it is local:
                    obs.append(old_source_spotted)

                # 'Backupdate' the past GLOBAL normalized observation:
                memory[aoi]['norm_obs'][cur_f_idx] = obs

                '''
                BACKupdate the list of the agents with low battery level (this occurs regardless of the fact that the the global observation
                is correct or not (maybe due to some delay) as the whole system must be stopped during the task execution if the task fails
                without being informed about the nature of the failure:
                '''
                if env.energy_constraint:
                    if IDi not in env.discharged_agents_history[aoi]:
                        # You could also set a desired battery threshold instead of using 0:
                        if old_battery_agent_i<=0:
                            env.discharged_agents_history[aoi].append(IDi)
                    else:
                        if old_battery_agent_i>0:
                            env.discharged_agents_history.remove(IDi)
                
                if env.learning_action_delay:
                    if env.num_flights==1:
                        for action_idx, action in enumerate(env.flights[0].actions_memory):
                            # Do not consider in the local observation only the action before the last one (which is the one that is currently being performed):
                            if action_idx==(env.flights[0].actions_size-1): # -> 'cur_f.actions_size-1' is the index associated with the last (most recent) action stored in 'actions_memory'
                                continue
                            else:
                                cur_action_dist = action[0] # -> action[0] contains the distance
                                cur_action_head = action[1] # -> action[1] containts the heading angle
                                obs.append(cur_action_dist)
                                obs.append(cur_action_head)            
            
            backupdated_done, backupdated_info, _ = env.update_done(current_spotting_agents=old_current_spotting_agents,
                                                                    source_time=old_source_time,
                                                                    elapsed_t=old_global_obs_AOI,
                                                                    discharged_agents=env.discharged_agents_history[aoi],
                                                                    rew=None,
                                                                    backupdate=True)


            '''
            The overall 'done' and 'info' is associated with the whole system and thus it refers
            to the same values for all the agents related to the same time instant AoI. When the terminal state occurs,
            then the ENode is informed by properly setting the associated value ('dones_infos') in its memory.  
            '''
            memory[aoi]['dones_infos'] = [backupdated_done, backupdated_info]

        def set_memory(env: "Environment", memory: Dict, obs_orig: Any, obs_norm: Any, aoi: float) -> None:
            '''
            Set the memory (for the first time). The ID is used only for the normalized observations which are
            individual even if they share also a common global (pre)processed observation.

            :param env: the variable associated with the envirorment used
            :param memory: the dictionary associated with the memory either of the ENode or a UAV
            :param obs_orig: the original obseravation
            :param obs_norm: the normalized obseravation
            :param aoi: the Age of Information of the observations
            '''

            memory[aoi] = {'orig_obs': {}, 'norm_obs': [], 'source_info_buffer': {}, 'dones_infos': []}
            memory[aoi]['orig_obs'] = obs_orig
            memory[aoi]['norm_obs'] = obs_norm
            memory[aoi]['source_info_buffer'] = copy.deepcopy(env.enode.source_info_buffer)

        '''
        If the memory buffer capacity is already full (in terms of maximum number of observations which can be stored),
        then remove the oldest observation stored (only if it is older than the observation currently processed):
        '''
        if (used_memory_size>=max_size):
            # Store the information only if the AoI of the current observation is more recent than the oldest stored in memory
            # (AOI actually indicates the time at which the info was received and not really the time elapsed between the current time and when
            # it was received):
            if aoi>=oldest_observation_t:
                memory.pop(oldest_observation_t)
                set_memory(env=env, memory=memory, obs_orig=orig_new_v, obs_norm=new_v, aoi=aoi)
        # Case in which the memory buffer is not full yet:
        else:
            # Store the current observation with its AoI (which is the current elapsed time, as explained in this method arguments explanation):
            set_memory(env=env, memory=memory, obs_orig=orig_new_v, obs_norm=new_v, aoi=aoi)

        '''
        BACKupdate only the ENode memory as we are assuming that there is no sensing delay (or that it is negligible) and hence
        local observation are not delayed for considered agents):
        '''
        if global_obs_update:
            # Possible Backupdate Handling:
            if env.learning_observation_delay:
                # Global memory backupdate:
                memory_backupdate(memory=memory, aoi=aoi, new_v=new_v, env=env, data_losses=data_losses, observation_loss=observation_loss)

    @staticmethod
    def local_actions_update(agent: "Flight", action: Tuple) -> None:
        '''
        Update the stored actions at local (agent) level.

        :param agent: the desired UAV for which performing the updating
        :param action: the action to be added during the updating process

        :return None
        '''
        if len(agent.actions_memory)>=agent.actions_size: # This check is redundant since the starting size of 'actions_memory' is set equal to 'actions_size'
            agent.actions_memory.remove(agent.actions_memory[0])
        agent.actions_memory.append(action)

    @staticmethod
    def round_to_closest_integer_multiple(n: float, multiple: float) -> int:
        '''
        Return the closest integer value to 'n' which is also a multiple of 'multiple'

        :param n: the value to be rounded
        :param multiple: 'n' must be a multiple of this parameter

        :return: a rounded value of 'n' multiple of 'multiple':  
        '''
        return multiple*round(n/multiple)

