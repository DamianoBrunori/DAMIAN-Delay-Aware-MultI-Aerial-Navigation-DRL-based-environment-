import os
import math
import numpy as np
from datetime import datetime

'''
_____________________________________________________________________________________________________________________________________
If you want to change the scenario features more in detail by combining, adding and implementing new observations, actions or
functionalities (other than those already provided), you can modify the parameters defined in this file.
_____________________________________________________________________________________________________________________________________
'''

# Set the SEED equal to None to initialize the random numbers with the current time system: 
SEED = 42

# Questi poi volendo li puoi anche far settare nel file 'setting.ini' !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Motion parameters:
UAV_SIZE = 30 # -> for the UAV rendering 
M = 0.2 # -> UAV mass
# Inertias:
Ixx = 1
Iyy = 1
Izz = 1
# Proportional coefficients:
Kp_x = 1 # not used
Kp_y = 1 # not used
Kp_z = 1 # not used
Kp_roll = 25
Kp_pitch = 25
Kp_yaw = 25
# Derivative coefficients:
Kd_x = 10 # not used
Kd_y = 10 # not used
Kd_z = 1
G = 9.81 # -> gravity acceleration
TRAJ_TIPE = 5 # -> degree of the polynomial trajectory
'''
Controller interpolating the waypoints of the trajectories (if set to 1, then it is supposed to be able to always match the desired trajectory):
For the time being, no controllers are implented (other than the standard one, i.e., '1'): note that neither the controller type '1' is used: in order to use it, you must uncomment the commented lines related to 'self.controller(...)' in 'definitions.py'
'''
CONTROLLER_TYPE = 1

# __________ Audio Source __________
# Intensity sound reference:
I0 = 1e-12

# _________________ Metrics used based on the desired algorithm _________________
METRICS_PPO = {'actor_loss': math.nan,
               'critic_loss': math.nan,
               'total_loss': math.nan,
               'best_score': 0.}
METRICS_SAC = {'actor_loss': math.nan,
               'critic1_loss': math.nan,
               'critic2_loss': math.nan,
               'critic_loss': math.nan,
               'value_loss': math.nan,
               'best_score': 0.}
# ____________________________________________________________________

# _________________ Global and local observations access _________________
# Communication keys:
ID_KEY = 'ID'
LAST_SHOT_KEY = 'LAST_SHOOT_TIME'
POS_FEATURES_KEY = 'POS_FEATURES'
AGENT_FEATURES_KEY = 'AGENT_FEATURES'
AOI_KEY = 'AOI'
ALL_POS_FEATURES = 'ALL_POS_FEATURES'
# _________________________________________________________________________

# _________________ Render variables _________________
# Colors:
WHITE = [255, 255, 255]
GREEN = [0, 255, 0]
BLUE = [0, 0, 255]
LIGHT_BLUE = [173,216,230]
BLACK = [0, 0, 0]
RED = [255, 0, 0]
YELLOW = [255, 255, 0]
FUCHSIA = [255,0,255]
SAND = np.array(([173, 156, 99]))/255
# ____________________________________________________


# _________________ Min-max normalization values _________________
# Time min-max value [s]:
TIME_MIN = 0
TIME_MAX = 14400 # -> it can be arbitrarily set

# Bearing min-max values:
BEARING_MIN = 0
BEARING_MAX = 2*math.pi

# Drift min-max values:
DRIFT_MIN = -math.pi
DRIFT_MAX = math.pi

# Distance min-max values:
DIST_MIN = 0
DIST_MAX = math.inf # -> defined in 'env.py' as it depends on the dimension of each operative polygon
# ________________________________________________________________


# _________________ Polygon _________________
# Landpoints status:
FREE = True
BUSY = False
# ___________________________________________


# _________________ Observations and actions spaces _________________

# Discrete actions (used only if a discrete action space is used)

# Angles (degrees):
ANGLES = [-135, -90, -45, 0, 45, 90, 135, 180]

'''
Distances need to be chosen large enough to avoid to stay on the current position as each time that a distance is chosen,
then the closest landing point is selected (also including the one related to the current position):
'''
DISTANCES = [0, 50, 100, 150, 200]

# Discrete actions for usecase 1:
ANGLES1 = [-135, -90, -45, 0, 45, 90, 135, 180]
DISTANCES1 = [0, 50, 100, 150, 200]

# Discrete actions for usecase 2:
ANGLES2 = [-90, -45, 0, 45, 90, 180]
DISTANCES2 = [0, 50, 100, 150]

DEFAULT_PAST_ACTION_VALUE = (0., 0.) # -> to be updated every time that a macro-action is added!

# Remember to add new observation names here in case they will be added in future!:
LOCAL_OBS_NAMES = ["battery",
                   "current agent spotted the source"]

GLOBAL_OBS_NAMES = ["source distance",
                    "source bearing",
                    "source drift",
                    "source time",
                    "agents distances",
                    "single-agent coverage area",
                    "AVG agent-all distance",
                    "AVG agent-not_spotting_agents distance",
                    "AVG spotting_agents-source distance",
                    "current N spotting agents"]

# Names of the available observations:
OBSERVATIONS_NAMES = [
                        "source distance",
                        "source bearing",
                        "source drift",
                        "source time",
                        "battery",
                        "agents distances",
                        "single-agent coverage area",
                        "AVG agent-all distance",
                        "AVG agent-not_spotting_agents distance",
                        "AVG spotting_agents-source distance",
                        "current N spotting agents",
                        "current agent spotted the source"
                     ]

# Names of the available actions:
ACTIONS_NAMES = [
                    "distance",
                    "angle"
                ]
# ___________________________________________________________________

# __________ Observations Delays [s] __________
MIN_OBS_DELAY = 3.
MAX_OBS_DELAY = 7.
PROB_OBS_DELAY = 0.15
PROB_OBS_LOSS = 0.05
# _____________________________________________


# _______________ Clocks frequency (used only if 'explicit_clock' feature is enabled in 'scenario_parameters.ini'): ______________
# Unit measure: [Hz]
AGENT_CLOCK_FREQ = 0.5 # = 2 seconds
ENODE_CLOCK_FREQ = 0.25 # = 4 seconds
INITIAL_SYSTEM_CLOCK_FREQ = 0.1 # = 10 seconds
# ________________________________________________________________________________________________________________________________


# _________________ Directories _________________
# Folder for tensorboard logs:
ROOT_FOLDER = 'runs/'
now = datetime.now()
NOW_FOLDER = now.strftime("%m-%d-%Y_%H:%M:%S")  
# Algorithm name folders:
ALGORITHM_FOLDER = 'ppo/'
LOGS_FOLDER = 'logs/'
# Folder to save the NNs checkpoints based on the used algorithm:
MODEL_FOLDER = 'model/'
# Folder for environment plot (-png and .gif) based on the used algorithm:
PLOT_FOLDER = LOGS_FOLDER + 'env_images/'
# Directories where the local server (where the NN code is run) stores the observation received by the external machine:
PATH_FOR_EXT_COMMUNICATION = './MachinesCommunication/'
OBS_FILENAME_FOR_EXT_COMMUNICATION = 'external_obs.csv'
PRED_FILENAME_FOR_EXT_COMMUNICATION = 'prediction.csv'
# _______________________________________________