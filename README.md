# DAMIAN (Delay-Aware MultI-Aerial Navigation) DRL-based Environment


### Table of Contents

- [Description](#description)
- [Directory Tree](#directory-tree)
- [Setup](#setup)
- [Run](#run)
- [References](#references)
- [License](#license)
- [Author Info](#author-info)

## Description 

This is OpenAI-Gym environment representing a multi-UAV system that can optimize to learning different tasks, e.g., area coverage, spotting, tracking, energy minimization: a static (ord dynamic) audio source is present in the main scenario here provided. 
It is provided with two Deep Reinforcement Learning (DRL) algorithms, i.e., Multi-agent Proximal Policy Optimization (PPO) and Soft Actor-Critic (SAC), with theri dealyed-informed variations.

The Unmanned Aerial Vehicles (UAVs) motion take is based on their maximum speed and/or acceleration and it is performed according to a polynomial trajectory which can be of 3th, 5,th or 7th degree: the trajectory is supposed assumed to be perfectly travelled through the usage of a controller that interpolates 4 main waypoints and hence allows the UAVs to move from a landing point to another one. The audio source motion is modelled instead based on a bang-coast-bang motion, i.e., by using a trapezoidal speed profile. Note also that the UAVs are supposed to fly at different Flight Levels (FLs) in order to avoid overlapping trajectories at the same altitude during the horizontal flight phase.
Many scenario and training features can be set at wish by using the configuration files .ini (see next section), but the main features that can be set and that will change completely the training process and the animation associated with the motion of the UAVs , are the 'action_delay' and the 'observation_delay' features.
By enabling the 'action_delay', the effect of the performed actions can be seen only after a certain amount of time (thus, actions are delayed in time) depending on:

* takeoff time
* flight phase time
* landing time

If this feature is not enabled, then actions take place instantaneously.
In this latter case, the battery consumption (if enabled) will be still computed in the proper way: indeed it will consider the chosen resolution step (i.e., the smallest time step where UAVs can take an action) selected by the user and it will still depend on the three main flight phases previously listed. For what concerns the 'osberavation_delay', we still have a relevant variation during the leraning process, but the redenred animation will be the same: in particular, if 'action_delay' is not enabled, the animation will be shown in 2D, otherwise it will be shown in 3D.   

For more details about the scenario and training parameters configuration, see [Configuration](#configuration) section.

### Environment Structure

Here below it is shown a scheme representing the structure implementation of this environment. 

<p align="center">
  <img src=Images/FrameworkOverview.jpg>
</p>

The configuration files are used to generate the environment and the agents and to set up the training parameters.
PPO and SAC algorithms can be used, but any other algorithm can be implemented and/or used very easily and quickly. Only one Neural Network (NN) is used and it is feeded by the inputs coming from each agent local observation which is collected by a ENode (or a Server) aimed at collecting all the local observations and broadcasting the main info that can be derived from each of them to all the agents.

Thus, the environment is structured as a Mixed Observability Markovian Decision Process (MOMDP), made up by a global observation associated with the updated info broadcasted by the Server-node and a partial observation related to each agent local observation and constrained to the sensing range of each UAV. Thus, we could state that the used paradigm is mostly centralized, bu we need also to specify that even if there is only a single NN (i.e., centralized training), the info processed are made up by both global and local observations and refer to each different UAV (which senses the environment in a distributed and local manner): each UAV acts based on both a global info and on its own local observation. Note that the global shared info is processed in such a way to make it independent from the number of the agents, and that the user can also select a Centralized Training with Decentralized Execution (CTDE) learning paradigm: the latter case allows the agents to select a new action based on their own observations, without necessarily relying on a global one.  

### Configuration

#### Scenario Parameters Setting

Here it is reported the list of all the parameters that can be set in order to modify and vary different scenario features.
A detailed explanation of the meaning of each of the following parameters can be found in settings/scenario_parameters.ini.

```
[Time Setting]
dt = 1
unit_measure = s 
explicit_clock = false

[Operative Polygon]
min_area = 100000
max_area = 200000 
same_env = false 
n_landing_points = all
step = 30
deg = 45

[UAVs]
n_uavs = 1
min_fl = 1
max_fl = 4
sep_fl = 0.5
sensing_time = 0
same_initial_location = true
uavs_locations = [(62.85586449489398, -119.21836454696785)]
energy_constraint = false
max_speed = 5
max_acc = 2
min_db_level = 105
p_eng = 10
v_ref = 20
p_bat = 9
p_bat_charging = 4
b_efficiency = 12
b_res = 5
action_delay = false 
observation_delay = true
observation_loss = false 

[Audio Source]
static = true
constant = true 
source_location = (51.48646156683597, 32.62646123452701)
source_db_level = 140
global_max_speed = 20
global_max_acc = 2.5
local_max_speed = 1.7 
local_max_acc = 1.1
n_hotspots = 10
n_p_min = 3
n_p_max = 7
local_d = 50

[Communication]
STAR = t
```

#### Training Parameters Setting

All the parameters related to the training phase are listed here below.
A detailed explanation of the meaning of each of the following parameters can be found in settings/training_parameters.ini.

```
[System]
CT_DE = false
action_space = discrete
action_delay = false
actions_size = 3
observation_delay = false
observations = [
		"current agent spotted the source"
		] 
actions = [
	   "distance",
	   "angle"
	   ]

[PPO Hyperparameters]
lr = 0.0003
gamma = 0.9
gae_lambda = 0.85
policy_clip = 0.2
c_value = 0.5
c_entropy = 0.01
A_fc1_dims = 256
A_fc2_dims = 256
C_fc1_dims = 256
C_fc2_dims = 256

[SAC Hyperparameters]
lr1 = 0.0003
lr2 = 0.0003
alpha = auto
gamma = 0.99
tau = 0.005
reward_scale = 2
A_fc1_dims = 256
A_fc2_dims = 256
C_fc1_dims = 256
C_fc2_dims = 256
V_fc1_dims = 256
V_fc2_dims = 256

[General Hyperparameters]
batch_size = 5
action_sigma = 0.6
action_sigma_decay_rate = 0.05 
action_sigma_min = 0.1
n_episodes = 500
N = 15

[Terminal Conditions]
perc_uavs_detecting_the_source = 0.3
perc_uavs_discharged = 0.60
task_ending = false
task_success = true
time_failure = true
battery_failure = false

[Reward]
cumulative_reward = true
cumulative_rew_size = 3
W_agent = 1
W_system = 1
Wa = 1
Wb = 1
Wc = 1
Wd = 1
We = 1
Wf = 1
Wg = 1
Wh = 1
Wi = 1
Wl = 1
Wm = 1
Wn = 1
```

#### Hyperparamters Tuning Setting (Sweeping)

The list of parameters listed here below represent what all that parameters on which it is possible to performe a tuning. here it is possible to assign a set o values (to each parameter) among which to perorm the tuning.
A detailed explanation of the meaning and usage of each of the following parameters can be found in section _Visualize The Hyperparameters Tuning Charts Through Weights & Biases_ and in file settings/sweep_config.yaml.

```
program: main.py
method: grid
metric:
  name: best_score
  goal: maximize
parameters:
  lr_ppo:
    values: [0.1, 0.4]
  batch_size:
    values: [16, 32]
  epoch_duration:
    values: [20, 21]
early_terminate:
  type: hyperband
  max_iter: 2
  s: 2
```

### Environment Generation

The generated environment is a polygon, whose surface can be either manually selected (in settings/scenario_parameters.ini) or entered by using an external FIR_file (see later on). If a discrete action space is chosen, then it is discretized into N landing points (not otherwise).

The image here below represents the discretized environment, where:
* the blue surface represents the discretized polygon with all the possible landing points;
* the red square represents the bounding box of the polygon;
* the orange surface represents the area laying outside the polygon (between the polygon and its bounding box).


<p align="center">
  <img src=Images/EnvironmentDiscretization.png>
</p>

The path that is supposed to be travelled by the audio source is generated offline and interpolated at training time.
The audio source can move at different speeds (see image below):
* faster when travels the dark blue paths;
* slower when travels the light blue paths.

<p align="center">
  <img src=Images/SourceInitialPath.png>
</p>

### Environment Animation

In order to get a complete overview of this environment, here it is provided either the 2D and the 3D animation of the learning process of the UAVs looking for the audio source. 2D animation is shown when the action execution is assumed to be completed by every agent at the end of each timestep (i.e., no actions' delays applied to the considered scenario). 

<p align="center">
  <img src=Images/env_animation2D.gif width="49%">
  <img src=Images/env_animation3D.gif width="49%">
</p>

In particular, the following items are shown in the animation:
* green and small crosses inside the polygon representing the landing points;
* black cross structure with four terminal colored circles indicating the numbered UAVs with their four rotors and where the they are located;
* big empty colored circles representing the sensing range of each agent (in the image only the sensing range of the 4th UAV is enabled);
* a filled dark circle place on the polygon showing where the audio source is located (it is surrounded by an empty violet circle if it is emitting an audio frequency). 

## Directory Tree 

```
├── LICENSE
├── logs
│   ├── scenario_run.txt
│   └── training_run.txt
├── main.py
├── models
│   ├── PPO.py
├── muavenv
│   ├── definitions.py
│   ├── env.py
│   ├── global_vars.py
│   ├── __init__.py
│   ├── units.py
│   ├── utils.py
│   └── utilsWandB.py
├── parsers
│   ├── external_info.py
│   ├── scenario.py
│   ├── sweeping.py
│   └── training.py
├── README.md
├── requirements.txt
├── runs
│   ├── sweep
│   │   └── ppo
│   ├── test
│   │   └── ppo
│   ├── train
│   │   └── ppo
├── settings
│   ├── scenario_parameters.ini
│   ├── sweep_config.yaml
│   ├── sweep_parameters.ini
│   └── training_parameters.ini
├── setup.py
├── train_and_sweep.py
```

## Setup

Before proceeding with the installation (3 different possible procedures are here listed), be aware that the framework has been tested with Python version 3.9.7 on OS Ubuntu 20.04.4 LTS.
Using either a different OS and/or a different Python version could cause incompatibility and/or instability of the framework.

### Standard installation

Be sure that pip is installed ([click here](https://pip.pypa.io/en/stable/installation/#get-pip-py) for more details about pip installation). 

Navigate through the terminal to the main directory and install the needed requirements.

```
pip install -r requirements.txt
python setup.py install
```

### Anaconda-based Installation

Install Anaconda by following the [official documentation](https://docs.anaconda.com/anaconda/install/).
Once installed, open the terminal and create your own environment.

```
conda create --name <your_env_name> --file requirements.txt
```

### Docker Installation

To begin with, [install Docker Engine](https://docs.docker.com/engine/install/ubuntu/) correctly.
Now, navigate through the terminal to the main directory containing the 'Dockerfile' file and build the docker image 'delayed_mdp' with the TAG 'multiuav':

```
docker build -t delayed_mdp:multiuav .
```

All the needed libraries are now ready to be used for running the code. For more detail on how to run the built docker image and other features, see the 'docker_instructions' file.  

## Run

Different running modes are available[^1]. Note that the modes not described in [References](#references) are not guaranteed to work as intended in all their variations. 

```
usage: main.py [-h] [--train] [--test] [--no_learn] [--sweep] [--external_obs]
               [-FIR_file FIR_FILE] [-FIR_ID FIR_ID] [-model_path MODEL_PATH]
               [-test_scenario_params_path TEST_SCENARIO_PARAMS_PATH]
               [--external_communication] [--ppo] [--sac]

optional arguments:
  -h, --help            show this help message and exit
  --train               Training phase enabled
  --test                Testig phase enabled
  --no_learn            Learning phase disabled
  --sweep               Hyperparameters sweep enabled
  --external_obs        External observation feeding-in procedure enabled
                        (position expressed in (lat,lon) coordinates)
  -FIR_file FIR_FILE    Insert the path of the .csv or .geojson file containing
                        the info of the Flight Information Region (FIR) (expressed in (lat,lon))
                        to use for the current run
  -FIR_ID FIR_ID        Insert the ID of the Flight Information Region (FIR) (expressed in (lat,lon))
                        to select a specfic FIR (if more than one is present) contained inside the .csv file in -FIR_file
  -model_path MODEL_PATH
                        Insert the path of the (trained) model to use for the current run (use only if --test in enabled)
  -test_scenario_params_path TEST_SCENARIO_PARAMS_PATH
                        Insert the path of the file containing the parameters of the desired scenario to test based on
                        a learned policy (to be used only if --test is enabled and if a test scenario different from that
                        used during learning is desired). If it is not specified, then the testing scenario parameters
                        will be automatically set equal to the ones used for the trained model
  --external_communication
                        Enable a communication client/server with another machine to get external local observations of
                        the agents: if enabled, then also '--external_obs will' be automatically enabled
  --ppo                 Use Proximal Optimization Policy (PPO) algorithm
  --sac                 Use Soft Actor-Critic (SAC) algorithm
```

### Flight Information Region (FIR) .csv File Format: 

Here below it is represented the file format chosen to represent an external Flight Region of Interest following the standard format used by [Eurocontrol](https://www.eurocontrol.int/publication/flight-information-region-firuir-charts-2022). 

|Airspace ID |Min Flight Level|Max Flight Level|Sequence Number|Latitude |Longitude|
|------------|----------------|----------------|---------------|---------|---------|
|LlanbedrTest|0               |60              |1              |52.814992|-4.132176|
|LlanbedrTest|0               |60              |2              |52.816483|-4.131736|
|LlanbedrTest|0               |60              |3              |52.817467|-4.129553|
|LlanbedrTest|0               |60              |4              |52.812971|-4.124613|
|LlanbedrTest|0               |60              |5              |52.812117|-4.125406|
|LlanbedrTest|0               |60              |6              |52.811025|-4.124278|
|LlanbedrTest|0               |60              |7              |52.808889|-4.128036|
|LlanbedrTest|0               |60              |8              |52.808781|-4.130136|
|LlanbedrTest|0               |60              |9              |52.811183|-4.130969|
|LlanbedrTest|0               |60              |10             |52.814502|-4.132138|

### Flight Information Region (FIR) .geojson File Format:

Now another file format (.geojson) accepted as an external FIR is shown.
It is possible to edit a .geojson map data through [the free GeoJSON online editor](https://geojson.io/#map=15/52.5492/-3.9595).

<p align="center">
  <img src=Images/geojson_online_editor.png>
</p>

Once the online editor has been opened, follow these simple few steps:
* zoom in or zoom out in order to identify the desired area;
* click on 'draw a polygon' (exagonal icon) and frame the desired FIR;
* if you need more than one FIR, then repeat the previous step until all the desired FIRs have been framed;
* now download the file by clicking on Save -> GeoJSON (the downloaded file is ready to be used).

### External (Local) Observations File Format (.csv)

Here it is shown the format used to represent an external local observation of the agents:
* _ID_ is the identification number (or acronym) associated with the the considered UAV;
* _source time_ indicates the elapsed time since the UAV ID spotted the source;
* _lat_ is the latitude of the position of the UAV ID;
* _lon_ is the longitude of the position of the UAV ID;
* _source spotted_ is a binary value indicating if the UAV ID is currently spotting the source or not;
* _track_ is the heading angle of the UAV ID;
* _takeoff-landing time_ points out the takingoff and landing times (which are supposed to be equal based on the UAV hardware and its flight level);
* _battery_ indicates the remaining battery percentage of the UAV ID;
* _FLA_ stands for Flight Level Assignment and represents the flight altitude of the UAV ID;

|ID |source time|lat      |lon      |source spotted|track|takeoff-landing time|battery|AoI|FLA|
|---|-----------|---------|---------|--------------|-----|--------------------|-------|---|---|
|AL2|1          |52.815992|-4.131736|1             |90   |12                  |90     |4  |10 |
|OI3|1          |52.815992|-4.131736|0             |-35  |26                  |100    |4  |15 |
|PL8|3          |52.815992|-4.131736|0             |0    |20                  |85     |2  |20 |
|AS2|0          |52.815992|-4.131736|1             |90   |12                  |90     |2  |12 |
|OK3|0          |52.815992|-4.131736|0             |-35  |26                  |100    |2  |18 |
|PP8|3          |52.815992|-4.131736|0             |0    |20                  |85     |2  |22 |
|UL2|0          |52.815992|-4.131736|1             |90   |12                  |90     |3  |24 |
|RI3|1          |52.815992|-4.131736|0             |-35  |26                  |100    |4  |26 |
|FL8|1          |52.815992|-4.131736|0             |0    |20                  |85     |4  |20 |
|FL9|NA         |52.815992|-4.131736|0             |0    |20                  |85     |3  |28 |


### Running Examples

Now a non-exhaustive list of examples with different running modes will follow: the most straightforward commands, i.e., --test_scenario_params_path and --no_learn, will be not analyzed.
For the sake of simplicity, we will run all the following examples with the PPO algorithm when the training phase is required.

#### Hyperameters Tuning

```
python main.py --sweep
```

#### Training

```
python main.py --train --ppo
```

#### Testing

```
python main.py --test -model_path <your_model_path> 
```

#### External Flight Information Region (FIR) by using a properly formatted .csv file:

Remember to specify either --train or --test - model_path <your_model_path>. It is possible to add also an external observation (see next section).
The --train option has been here used just to provide a working example.

```
python main.py --train --FIR_file <your_FIR_file_path> -FIR_ID <your_FIR_ID> --ppo
```

If only one FIR is present inside the selcted .csv FIR_file and no FIR_ID is specified, then that only FIR will be automatically selected. 

#### External Flight Information Region (FIR) by using a properly formatted .geojson file:

Remember to specify either --train or --test - model_path <your_model_path>. It is possible to add also an external observation (see next section). Here the --train option is used only to provide a working example.

```
python main.py --train --FIR_file <your_FIR_file_path> --ppo
```

After running the previous command, the user will be asked to enter the number of the desired FIR to select (if more than one is present inside the FIR .geojson file).
In this case, by entering the number 'i' (i=1,...,N, where N is the number of FIR present inside the selected .geojson FIR), the chosen FIR will be first the i-th in order of occurrence inside the used .geojson FIR file.

#### External (Local) Observations[^2]

Remember to specify either --train or --test - model_path <your_model_path>. It is possible to add also a FIR_file associated with a FIR_ID (see previous section). In this case the --train option is used only to provide a working example. 

```
python main.py --train --external_observation --ppo
```

A file 'external_obs_history.csv' will be generated in the directory 'runs/(train_or_test_folder)/algorithm_name/timestamp' (e.g., if you used the flag --train and the PPO algorithm, then the folder will be 'runs/train/ppo/timestamp)'. This file will contain external observations passed to the model, and for each observation will be specificed the associated epoch and iteration.

#### External Communication[^2]

The same premises (related to the --train and --test options) made so far are valid also for this use-case example. 

``` 
python main.py --train --external_communication --ppo
```

With this option enabled, another terminal will be automatically opened to run the a listener 'server' waiting for an external observation coming from another 'client' machine. If you want to try a test on your local machine by simulating a local client/server communication, you can run also the 'client.py' file (in MachinesCommunication/ExternalMachine/client.py) in another terminal on your laptop. If different machines are used instead, remember to change the file paths specified in 'client.py' accordingly.
In this way it is possible to to run a test based on a pretrained model (but you can run a trainig as well) on a different machine than the one used to get the local observation coming from the UAVs.
Be aware that when you run a client/server local test on the same machine, the external_obs.csv file in MachinesCommunication/ExternalMachine/external_obs.csv is just a test file and it must be replace constantly by the real external observations which are supposed to be provided by the external machine running the client.py file.

*A future development idea is trying to allow the machine running the pretrained model to accept multiple 'client' connections, in such a way to be able to perform UAVs action predictions remotely for different real scenarios which are taking places on different and far areas.*    

### Visualize Training Charts Through Tensorboard

TensorBoard is used to visualize loss and reward functions associated with the training phase (click [here more info about TensorBord](https://www.tensorflow.org/tensorboard)).
You can visualize the trend of the training charts at training time and observe them varying in real time.
To do that, simply open another terminal (in addition to the one used to run the training). After that, make sure that you can use TensorBoard from this new terminal: e.g., if a conda environment with all the requirements needed to run this framework has been created, then it is needed to activate this environment also from this new terminal.
Now type the following command:

```
tensorboard --logdir <path_of_the_TFEvents_file>
```
The <path_of_the_TFEvents_file> file path, is the path where the TFEvents file (containing all the info which can be processed by TensorBoard)
is located. In our case, it is in: runs->logs->train(OR test)->ppo->"m-d-Y_H:M:S"->logs/. 

At this point, the terminal will show the the following message:

```
TensorBoard 2.9.0 at http://localhost:6006/
```

Now simply click on the local link to visualize the training charts.

Here below is shown an example of the page to which the user is redirected once the previous link has been clicked.

<p align="center">
  <img src=Images/tensorboard_example.png>
</p>

### Visualize The Hyperparameters Tuning Charts Through Weights & Biases

Weights & Biases (W&B) is here used to visualize the trend of the hyperparameters search when --sweep option is enabled.
In settings->sweep_config.yaml it is possible to set the value of the hyperparameters among those to perform the search and also the method to use for the search and many other features (for more details about the needed syntax to properly modify this configuration file, see the [W&B official documentation dedicated to the Sweep Configuration](https://docs.wandb.ai/guides/sweeps/configuration)).
Before being able to visualize the charts through W&B, you need to create a W&B account and set it up ([the official W&B documentation describes how few to set up W&B in few quick steps](https://docs.wandb.ai/quickstart)).

Once the -sweep mode has been enabled, a message from W&B will be shown on your terminal, where the user can find a clickable link to access the charts associated with the hyperparameters search. After clicking on it, the user will be automatically redirected to the W&B page allowing monitoring the trend of the hyperparameters search and many other features related to it.

The snapshot here below shows an example of some of the features that it is possible to visualize during the hyperparameters search.

<p align="center">
  <img src=Images/sweepW&B.png>
</p>

## References

[Damiano Brunori and Luca Iocchi “A Delay-aware DRL-based Environment for Cooperative Multi-UAV Systems in Multi-purpose Scenarios”, ICAART 2024 16<sup>th</sup> International Conference on Agents and Artificial Intelligence, February 2024.](https://www.insticc.org/node/TechnicalProgram/icaart/2024/presentationDetails/123479)

## License

TO BE APPLIED BEFORE OFFICIAL PUBBLICATION:

Copyright (c) 2023 Damiano Brunori

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Author Info

- Name: Damiano
- Surname: Brunori
- University: Sapienza University of Rome
- Master: [Artificial Intelligence and Robotics](https://corsidilaurea.uniroma1.it/it/corso/2019/30431/home)
- PhD: Artificial Intelligence
- university e-mail: brunori@diag.uniroma1.it
- private e-mail damiano.brunori2@gmail.com 

[^1]: Some code might be redundant and a 'code smoothing' process is still in progress
[^2]: The proper functioning of the external observations and communication is not ensured for every scenario and/or learning features combination: work in progress
