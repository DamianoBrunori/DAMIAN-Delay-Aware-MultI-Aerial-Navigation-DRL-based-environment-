"""
Environment module
"""
import gym
from gym import spaces 
from typing import Dict, List, Union, Tuple
import itertools as it
import pathlib
from muavenv.definitions import *
import muavenv.units as u
from muavenv.utils import *
from muavenv.global_vars import *
from shapely.geometry import LineString, box
import matplotlib
from math import pow, cos, sin
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.art3d as art3d
from mpl_toolkits.mplot3d import Axes3D
import io
from io import BytesIO
import copy
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

class Environment(gym.Env):
    """
    Class Environment
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self,
                 args,
                 scnr_cfg: "ScenarioConfig",
                 train_cfg: "TrainingConfig",
                 train_phase: Optional[bool] = False,
                 FIR_file: Optional[pathlib.Path] = None,
                 FIR_ID: Optional[str] = None,
                 external_obs: Optional[Union[pd.DataFrame, None]] = None,
                 external_communication: Optional[bool] = False,
                 algorithm: Optional[str] = 'ppo', # 'sac'
                 min_speed: Optional[float] = 50.,
                 max_episode_len: Optional[int] = 15,
                 min_distance: Optional[float] = 5.,
                 distance_init_buffer: Optional[float] = 5., 
                 desired_flights_points: Optional[List] = [], # --> desired fixed points for generating the flights
                 desired_airspace_verteces: Optional[List] = [], # --> desired fixed points for generating the airspace
                 **kwargs):

        self.args = args

        # Set the angles and distances for the actions associated with the 3 use cases provided:
        if self.args.usecase1:
            self.angles = ANGLES1 #[-135, -90, -45, 0, 45, 90, 135, 180]
            self.distances = DISTANCES1 #[0, 50, 100, 150, 200]
        elif self.args.usecase2:
            self.angles = ANGLES2 #[-90, -45, 0, 45, 90, 180]
            self.distances = DISTANCES2 #[0, 50, 100, 150]
        # Case in which a use case has not been running:
        else:
            # Remember that when you are performing a --test, then you must be sure that you are using the same actions used in 
            # trained model that you are using to perform the test:
            self.angles = ANGLES
            self.distances = DISTANCES

        # Scenario parametes:
        self.scnr_cfg = scnr_cfg
        self.FIR_file = FIR_file
        self.FIR_ID = FIR_ID
        self.train_phase = train_phase
        self.external_obs = external_obs
        self.external_communication = external_communication
        self.algorithm = algorithm[:-1] if not algorithm.isalpha() else algorithm # only store the letters associate with the algorithm used
        self.scenario_action_delay = self.scnr_cfg.action_delay
        self.scenario_observation_delay = self.scnr_cfg.observation_delay
        self.observation_loss = self.scnr_cfg.observation_loss
        self.num_flights = self.scnr_cfg.num_flights
        self.max_area = self.scnr_cfg.max_area
        self.min_area = self.scnr_cfg.min_area
        self.min_speed = min_speed
        self.min_distance = min_distance
        self.max_episode_len = max_episode_len
        self.distance_init_buffer = distance_init_buffer
        self.desired_flights_points = desired_flights_points
        self.desired_airspace_verteces = desired_airspace_verteces
        self.time_unit = self.scnr_cfg.time_unit
        self.dt = self.scnr_cfg.dt
        self.max_time = TIME_MAX
        self.time_scaling
        self.STAR_antennas = self.scnr_cfg.STAR_antennas
        # flight tolerance to consider that the target has been reached (in meters) --> We are assuming that the UAVAs are homogeneous with the same speed
        self.tol = self.scnr_cfg.max_speed * 1.05 * self.dt
        self.multi_agent_timer_failure = 0.
        self.min_avg_all_distances = 0.
        # source tolerance to consider that the target has been reached (in meters)
        #self.src_tol = self.src_local_max_speed * 1.05 * self.dt # --> TO SET ??? --> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.viewer = None
        self.ax = None
        self.flights_colors = []
        self.source_color = 'black'
        self.not_firing_source_radius_render = 0.
        self.firing_source_radius_render = 0.
        self.airspace = None
        self.source = None
        self.enode = None
        self.current_source_walk = None
        self.flights = [] # -> list of flights
        self.conflicts = set()  # -> set of flights that are in conflict
        self.discharges = set()  # -> set of flights with low battery 
        self.discharged_agents_history = {}
        self.i = None
        self.POIs = None # --> It will store a dictionary of POIs their associated IDs
        self.POIs2 = None # --> It will store a dictionary of POIs their associated points
        self.lps = None # --> it will store the actual POIs (i.e., landing points) without their incremental ID
        self.POIs_states = None
        self.max_d = math.inf
        self.flight_altitudes = self.scnr_cfg.flight_altitudes
        self.energy_constraint = self.scnr_cfg.energy_constraint
        self.same_initial_location = self.scnr_cfg.same_initial_location

        self.communication = None
        self.elapsed_t = 0.
        self.flight_IDs = []

        '''
        Automatically set the explicit clock to False if an external observation is used (indeed, it will be
        received and picked externally!). Set to False also the delay associated with observations and actions
        in the scenario (they are obviously already included, if any, in the external observation!):
        '''
        if self.external_obs:
            self.scnr_cfg.explicit_clock = False
            self.scenario_action_delay = False
            self.scnr_cfgs.scenario_action_delay = False
            self.scenario_observation_delay = False
            self.observation_loss = False
            self.scnr_cfgs.observation_loss = False
            warnings.warn('Explicit clock is automatically set to False, as the "external_obs" flag is enabled!')
            time.sleep(3)

        if self.scnr_cfg.explicit_clock=='fixed':
            self.system_clock = self.scnr_cfg.initial_system_clock
        elif self.scnr_cfg.explicit_clock=='deterministic':
            clocks = [self.dt, self.scnr_cfg.enode_clock, self.scnr_cfg.agent_clock]
            self.deterministic_system_clock_min = min(clocks)
            self.deterministic_system_clock_max = 3*max(clocks)
            self.epsilon_system_clock = min(clocks)
            self.system_clock = (self.deterministic_system_clock_min+self.deterministic_system_clock_max)/2
            if self.system_clock%self.dt!=0:
                self.system_clock = DefAux.round_to_closest_integer_multiple(n=self.system_clock, multiple=self.dt)

        # Training parameters
        self.train_cfg = train_cfg
        self.ct_de_paradigm = self.train_cfg.ct_de_paradigm
        self.learning_action_delay = self.train_cfg.action_delay
        self.learning_observation_delay = self.train_cfg.observation_delay
        self.max_allowed_discharged_uavs = int(self.train_cfg.perc_uavs_discharged*self.num_flights)
        self.n_uavs_detecting_the_source = int(round(self.train_cfg.perc_uavs_detecting_the_source*self.num_flights)) # round to the closest integer
        self.n_uavs_detecting_the_source = np.clip(self.n_uavs_detecting_the_source, a_min=1, a_max=self.num_flights) # the minimum must be obviously at least 1: # IN RELTÀ IL MINIMO NON DOVREBBE ESSERE 0?????
        self.persistent_task = self.train_cfg.persistent_task
        self.task_ending = self.train_cfg.task_ending
        self.task_success_enabled = self.train_cfg.task_success
        self.time_failure_enabled = self.train_cfg.time_failure
        self.battery_failure_enabled = self.train_cfg.battery_failure
        self.selected_observations = self.train_cfg.observations
        self.selected_actions = self.train_cfg.actions
        self.n_airspaces = 0
        self.generate_airspace # -> call to a property method
        self.observation_and_action_spaces_generation # call to a property method
        # Set of points ('outer_p') generated outside the operative polygon during its discretization (used only for plotting reasones and only if desired):
        self.outer_p = []
        self.all_move_per_spot = {}

        assert len(self.desired_flights_points)==self.num_flights or len(self.desired_flights_points)==0, "The number of fixed desired point must empty or equal to the number of flights!"
        assert (self.scenario_observation_delay==True) or (self.scenario_observation_delay==False and self.learning_observation_delay==False), 'If the delay is not apply to the observations in the scenario settings, then it cannot be obviously considered during the learning process!'

    @property
    def observation_and_action_spaces_generation(self) -> None:
        """
        Create the observation and action spaces.

        :param: None
        :return: None
        """
        if self.train_cfg.action_space=='discrete':
            rad_angles = [math.radians(ang) if ang>=0 else (math.radians(ang)+u.circle)%u.circle for ang in self.angles] 
            self.actions_list = list(it.product(self.distances, rad_angles))
            self.action_space = spaces.Discrete(len(self.actions_list))
            self.n_actions = self.action_space.n
            self.max_angle = max(rad_angles)
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
            self.n_actions = self.action_space.shape[0]
            self.max_angle = 1
        self.action_dim = self.action_space.shape
        obs_lows, obs_highs, obs_lows_local, obs_highs_local = self.observations_bounds
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(obs_lows),), dtype=np.float32)
        self.local_observation_space = spaces.Box(low=0, high=1, shape=(len(obs_lows_local),), dtype=np.float32)

    # Used only to define the lenght of the observation space (min and max values are 0 and 1, respectively, due to the normalization):
    @property
    def observations_bounds(self) -> List:
        """
        Return the normalized (lower and upper) bounds for observation space
        """

        # Modify here below if you add new observations, either globals or locals!
        if not self.energy_constraint and 'battery' in self.selected_observations:
            self.selected_observations.remove('battery')
        n_observations = len(self.selected_observations)
        
        lows = [0 for name in range(n_observations)]
        highs = [0 for name in range(n_observations)]
        lows_local = []
        highs_local = []

        for name_idx in range(n_observations):
            lows[name_idx] = 0.
            highs[name_idx] = 1.

        """
        N.B.:
        The following min-max values are not used (all of them ranges between 0 and 1):
        the following code is used only to compute the exact dimension of the action space
        (above all in case in which the single distances between the current agent the all the other ones is used). 
        """
        for name_idx in range(n_observations):
            obs_name = self.selected_observations[name_idx]
            if obs_name=='source distance':
                lows[name_idx] = DIST_MIN
                highs[name_idx] = self.max_d
            elif obs_name=='source bearing':
                lows[name_idx] = BEARING_MIN
                highs[name_idx] = BEARING_MAX
            elif obs_name=='source drift':
                lows[name_idx] = DRIFT_MIN
                highs[name_idx] = DRIFT_MAX
            elif obs_name=='source time':
                lows[name_idx] = TIME_MIN
                highs[name_idx] = self.max_time
            elif obs_name=='battery':
                lows[name_idx] = 0
                highs[name_idx] = 100
                lows_local.append(0.)
                highs_local.append(100)
            elif obs_name=='agents distances':
                lows.remove(lows[len(lows)-1])
                highs.remove(highs[len(highs)-1])
                # '-1' is due to the fact that agents do not take into account distance w.r.t. themselves:
                agents_distances = [i for i in range(self.num_flights-1)]
                for i in range(self.num_flights-1):
                    lows.append(DIST_MIN)
                    highs.append(self.max_d)
            elif obs_name=='AVG agent-all distance':
                lows[name_idx] = DIST_MIN
                highs[name_idx] = self.max_d
            elif obs_name=='AVG agent-not_spotting_agents distance':
                lows[name_idx] = DIST_MIN
                highs[name_idx] = self.max_d
            elif obs_name=='AVG spotting_agents-source distance':
                lows[name_idx] = DIST_MIN
                highs[name_idx] = self.max_d
            elif obs_name=='current N spotting agents':
                lows[name_idx] = 0
                highs[name_idx] = self.num_flights #1
            elif obs_name=='current agent spotted the source':
                lows[name_idx] = 0
                highs[name_idx] = 1
                lows_local.append(0.)
                highs_local.append(1.)

        if self.learning_action_delay:
            n_decomposed_actions_per_agent = len(DEFAULT_PAST_ACTION_VALUE)
            n_decomposed_history_actions_per_agent = self.train_cfg.actions_size*n_decomposed_actions_per_agent

            for decomposed_a in range(n_decomposed_history_actions_per_agent):
                lows_local.append(0.)
                highs_local.append(1.)
                # Critic and actor are obviously the same for the single-agent case:
                if self.num_flights==1:
                    lows.append(0.)
                    highs.append(1.)
        
        global_lows = np.array(lows, dtype=np.float32)
        global_highs = np.array(highs, dtype=np.float32)
        local_lows = np.array(lows_local, dtype=np.float32)
        local_highs = np.array(highs_local, dtype=np.float32)
        
        return global_lows, global_highs, local_lows, local_highs

    @property
    def max_dist_assignment(self) -> None:
        """
        Update the maximum 2D point-to-point distance that can be traveled inside the considered polygon.
        
        :param: None

        :return: None
        """
        pol = self.airspace.cartesian_polygon
        pol_coords = list(pol.exterior.coords)
        #pol_coords = pol.exterior.coords.xy

        max_d = 0
        for p1 in pol_coords:
            for p2 in pol_coords:
                p1 = Point(p1)
                p2 = Point(p2)
                if p1!=p2:
                    d = p1.distance(p2)
                    if d>max_d:
                        max_d = d

        self.max_d = max_d

    @property
    def time_scaling(self) -> None:
        """
        Scale the current time step (in seconds) according to the time unit measure selected by the user.

        :param: None

        :return None
        """
        if self.time_unit=='s':
            self.dt = self.scnr_cfg.dt
            self.max_time = TIME_MAX
        elif self.time_unit=='m':
            self.dt *= 60
            self.max_time *= 60
        else:
            self.dt *= 3600
            self.max_time *= 3600    

    def update_deterministic_sys_clock(self, current_spotting_agents: List) -> None:
        """
        Update the the system clock whether the 'deterministic' explicit clock is enabled: it is a event-based deterministic clock since
        it is updated every time that an agent is spotting the source at the current time instant.

        :param current_spotting_agents: list of the agents which are currently spotting the source
        
        :return None
        """
        if self.scnr_cfg.explicit_clock=='deterministic':
            # Case in which there is at least one agent spotting the source at the current time instant:
            if current_spotting_agents!=[]:
                self.system_clock = self.deterministic_system_clock_min
            # Case in which there are no agents spotting the source at the current time instant:
            else:
                if self.system_clock>=self.deterministic_system_clock_max:
                    self.system_clock = self.deterministic_system_clock_max
                else:
                    self.system_clock += self.deterministic_system_clock_min

    @staticmethod
    def discretize_pol(pol: Polygon, step: Optional[float] = 150,
                       deg: Optional[float] = 45, n_points: Optional[Union[int, str]] = 30,
                       step_angle: Optional[int] = 1) -> Tuple[Dict, List[Point]]:
        """
        Discretize the operative polygon by generating specific landing points. If the desired number of landing points is
        larger than the maximum number of points generated based on the chosen settings, then the number of landing points is set
        equal to the latter case.
        
        :param pol: operative polygon
        :param step: desired distance between a discrete point and the next one in the generation process
        :param deg: desired turning angle to start to generate a new row of landing points once exiting the operative polygon 
        :param n_points: desired number of landing points
        :param step_angle: classification angle [degree] to order the landing points (according to a fictitious incremental ID)
            
        :return ordered_POIs1: dictionary of the POIs generated inside the polygon accessible by point
        :return ordered_POIs2: dictionary of the POIs generated inside the polygon accessible by IDs
        :return outer_p: lists of the points generated outside the polygon  
        """
        xy_pol = pol.boundary.coords.xy
        minx = min(xy_pol[0])
        miny = min(xy_pol[1])
        maxx = max(xy_pol[0])
        maxy = max(xy_pol[1])
        assert step<(maxx-minx), 'Step must be smaller than the maximum distance (along the horizontal axis) between polygon verteces!'
        
        max_c_pol_dist = max([pol.centroid.distance(Point(p[0], p[1])) for p in xy_pol])
        # Set the distance w.r.t. centroid of the polygon to assign an incremental ID to the landing points that will be generated:
        dist_perc = 0.7
        classification_dist = max_c_pol_dist*dist_perc 

        POIs = []
        outer_p = []
        rad = math.radians(deg)
        y = miny
        x_start = minx
        x_end = maxx
        # Points exactly on the borders are considered to be outside the polygon:
        while y<maxy:
            for x in np.arange(x_start, x_end+step, step):
                current_point = Point(x, y)
                # Points inside the considered polygon:
                if pol.contains(current_point):
                    POIs.append(current_point)
                # Points outside the considered polygon:
                else:
                    outer_p.append(current_point)
            x_start = x + 2*step*math.cos(rad) # 2*step --> In this way I am sufficiently outside the polygon
            step = -step
            x_end = minx if step<0 else maxx
            y += abs(step)*math.sin(rad) # --> Y value always increases; only X value goes once to the right and once to the left
        
        def bearing(A: Point, B: Point) -> float:
            delta_x = B.x - A.x
            delta_y = B.y - A.y
            compass = math.atan2(delta_y, delta_x)
            compass = (compass + u.circle) % u.circle       
            
            return compass

        n_generated_points = len(POIs)
        '''
        If the number of desired discrete points is equal or larger than the number of the genererated points,
        then set the number of desired discrete points equal to the number of generated points:
        '''
        if n_points!='all':
            if n_points>=n_generated_points:
                n_points = n_generated_points
        else:
            n_points = n_generated_points

        polygon_centroid = pol.centroid
        # Consider only the 'n_points' closest to the centroid of the polygon: 
        c_dist = [polygon_centroid.distance(p) for p in POIs]
        POIs_indeces_sorted = np.argsort(c_dist)[:n_points]
        POIs = [POIs[idx] for idx in POIs_indeces_sorted]

        # Compute (for the second time as 'POIs' have been modified) all the available POIs distances from the centroid of the considered polygon:
        c_dist = [polygon_centroid.distance(p) for p in POIs]
        # Compute the bearings from all the available POIs to the centroid of the considered polygon:
        c_to_POIs_bearing = [bearing(p, polygon_centroid) for p in POIs]
        # Put togheter all the available POIs distances and bearings to the centroid of the considered polygon:
        c_features = [(c_dist[i], c_to_POIs_bearing[i]) for i in range(n_points)] 

        # Conversion factor from degree to radians:
        rad_to_deg = 0.0175
        # POIs ordered based on their distances w.r.t. the centroid of the polygon and scrolled with an angle going from 0° to 360° (in radiands):
        ordered_POIs1 = {} # -> dictionary following the structure --> {ID: [point, status], ...}
        ordered_POIs2 = {} # -> dictionary following the structure --> {point: [ID, status], ...}
        # POI ID:
        poi_id = 1
        # Compute the 'classification step' in radians:
        classification_angle = step_angle*rad_to_deg
        # Angle check [radians]:
        for rad in np.arange(0, 2*math.pi, classification_angle):
            # Current angle lower bound:
            lb_deg = rad
            # Current angle upper bound
            ub_deg = lb_deg + classification_angle
            # Temporary list to store the points belonging to a specific angle interval:
            a_in_cur_angle_range = []
            for cf in c_features:
                # Angle value associated to the current feature:
                cur_angle = cf[1] 
                # Current point is within the current angle threshold w.r.t. the polygon's centroid:
                if cur_angle>=lb_deg and cur_angle<ub_deg:
                    # Distance check:
                    for dist in np.arange(0, max_c_pol_dist, classification_dist):
                        # Current distance lower bound:
                        lb_dist = dist
                        # Current distance upper bound:
                        ub_dist = lb_dist + classification_dist
                        # Distance value associated to the current feature:
                        cur_dist = cf[0]
                        # Current point is within the current distance threshold w.r.t. the polygon's centroid:
                        if cur_dist>=lb_dist and cur_dist<ub_dist: 
                            # Add all the point belonging to the current angle interval in a 'temporary list':
                            a_in_cur_angle_range.append(cur_angle)
                            break
                
            # Increasing sort of all the points belonging to the current angle interval:
            a_in_cur_angle_range.sort()
            for d in a_in_cur_angle_range:
                # Select the POIs (among the available ones) based on the desired order:
                POIs_idx = c_to_POIs_bearing.index(d)
                selected_p = POIs[POIs_idx]
                # Store both the current POI, its status and the ID of the UAV in it (if any):
                ordered_POIs1[poi_id] = [selected_p, FREE, 0]
                # Store both the current POI, its status and the ID of the UAV in it (if any)
                ordered_POIs2[(selected_p.x, selected_p.y)] = [poi_id, FREE, 0]
                poi_id += 1

        actual_points_on_polygon = [(cf[0]*math.sin(cf[1]), cf[0]*math.cos(cf[1])) for cf in c_features]
        
        return ordered_POIs1, ordered_POIs2, outer_p

    def resolution(self, actions: List) -> None:
        """
        Apply the resolution actions.
        
        :param actions: list of resolution actions assigned to each flight
        
        :return: None
        """

        '''
        -----------------------------------------------------------------------------------------------------------------------------------
        The actions are executed by the correct agent since each action is saved by following the same order as the osbervations (i.e., as
        the same order in which the agents are stored in 'self.flights'). Also the case of external observation is correctly managed
        (in 'self.enode.results_for_external_obs') as each action is assign to the proper agent and, only after that,
        the ID of that UAV is communicated to the 'external source'.
        -----------------------------------------------------------------------------------------------------------------------------------
        '''
        for i, f in enumerate(self.flights):
            # If the sensing range is enabled, then the current UAV can choose a new action as it is 'listening' and 'waiting for act':
            if f.sensing_range_enabled and f.cur_t_sensing>=f.sensing_time: 
                if f.ID not in self.discharges:
                    # Target discretization (i.e., select the landing point closest to the selected target):
                    cur_achievable_landing_points = f.achievable_pos
                    if cur_achievable_landing_points!=[]: 
                        # Discrete action space case:
                        if self.train_cfg.action_space=='discrete':
                            cur_action = self.actions_list[actions[i]]
                            dist = cur_action[0]
                            heading = cur_action[1]
                        # Continuous action space case:
                        else: 
                            gauss_distance = actions[i][0]
                            dist = Scaling.min_max_denormalization(x_norm=gauss_distance, x_min=-1, x_max=1)
                            gauss_bearing = actions[i][1]
                            heading = Scaling.min_max_denormalization(x_norm=gauss_bearing, x_min=-1, x_max=1)
                            cur_action = (dist, heading)

                        # Update the number of motions (actions chosen). A motion could be also 'keeping the position' and hence 'do not fly':
                        f.n_motions_before_spotting += 1

                        # Normalize the actions for the case in which they have to be used as observations:
                        dist_for_obs = Scaling.min_max_normalization(x=dist, x_min=DIST_MIN, x_max=self.max_d)
                        angle_for_obs = Scaling.min_max_normalization(x=heading, x_min=0, x_max=self.max_angle) # -> the minimum angle will be always 0, as it is expresssed in radians according to the formula using u.circle (that returns 0 for the biggest negative angle, i.e. '-360')
                        cur_action_for_obs = (dist_for_obs, angle_for_obs)

                        # Update the actions history stored LOCALLY in the current UAV (every UAV knows exactly its actions undertaken so far):
                        DefAux.local_actions_update(agent=f, action=cur_action_for_obs)

                        x_target = f.position.x + dist*math.sin(heading)
                        y_target = f.position.y + dist*math.cos(heading)
                        new_target = Point(x_target, y_target)

                        # If the agent is outside the operative polygon, then hold the current position:
                        if not self.airspace.cartesian_polygon.contains(new_target):
                            new_target = Point(f.position.x, f.position.y)    
                        f.target = new_target
                        # if target!=start, then allows to move towards target by resetting the current moving time:
                        if f.target!=f.start:
                            f.cur_moving_time = 0

                        f.target = f.landing_point(cur_achievable_landing_points) # -> select the landing point closest to the target position
                        # If an external observation has been using, then log the chosen action and the next positions associated to it:
                        if isinstance(self.external_obs, pd.DataFrame):
                            first_current_log = True if i==0 else False 
                            self.enode.results_for_external_obs(ref_p=self.airspace.geodesics_pol_centroid, flight=f, chosen_heading=heading, chosen_dist=dist, first_current_log=first_current_log)
                        
                        '''
                        Implicitly reset the times currently taken for the 3 flight phases and compute the trajectory coefficients
                        based on the new target selected according to the action taken:
                        '''
                        f.traj_coefficient_update()
                        f.x_traj_history = []
                        f.y_traj_history = []
                        f.z_traj_history = []
                    else:
                        #print("No achievable points for UAV:" + str(i) + " with battery level:" + str(f.battery))
                        pass
                else: # -> If the current UAV is discharged, then simply skip it:
                    continue

            # If the sensing time of the current UAV is ended, then icrease it without allowing the UAV to take any action:
            elif f.sensing_range_enabled and f.cur_t_sensing<f.sensing_time:
                f.cur_t_sensing += self.dt

        if isinstance(self.external_obs, pd.DataFrame):
            if self.external_communication:
                self.enode.generate_predictions_file

        self.all_move_per_spot[self.elapsed_t+1] = {f.ID:f.n_motions_before_spotting for f in self.flights}

        return None

    def reward(self, observations: List) -> List[float]:
        """
        Return the rewards assigned to each agent.
        
        :param observations: list of observations associated with all the agents
        
        :return: list of rewards assigned to each agent
        """

        # Initialize the rewards:
        rews = np.zeros(self.num_flights) # -> it will be assigned inside 'single_rewards(...)'
        # Rewards associated with the single agents:
        rews_agent = np.zeros(self.num_flights)
        # Rewards associated with the agents swarm:
        rews_system = np.zeros(self.num_flights)

        # Single agent observation weight:
        W_agent = self.train_cfg.W_agent
        # System observation weight:
        W_system = self.train_cfg.W_system

        # Source-agent distance weight:
        Wa = self.train_cfg.Wa
        # Source-agent bearing weight:
        Wb = self.train_cfg.Wb
        # Source-agent drift weight:
        Wc = self.train_cfg.Wc
        # Source-agent time weight:
        Wd = self.train_cfg.Wd
        # Battery agent weight:
        We = self.train_cfg.We
        # Agent-all distances weight:
        Wf = self.train_cfg.Wf
        # Area coverage weight:
        Wg = self.train_cfg.Wg
        # AVG agent-all distance weight:
        Wh = self.train_cfg.Wh
        # AVG agent-not_spotting_agents distance weight:
        Wi = self.train_cfg.Wi
        # AVG agent-spotting_agents distance weight:
        Wl = self.train_cfg.Wl
        # current N spotting agents weight:
        Wm = self.train_cfg.Wm
        # current agent source spotting weight:
        Wn = self.train_cfg.Wn

        global_observations = observations[0]
        local_observations = observations[1]

        obs_for_rew = copy.deepcopy(global_observations)
        obs_for_old_rew = copy.deepcopy(self.enode.memory)

        def single_rewards(obs_for_rew: List, rews_agent: List, rews_system: List, aoi: Optional[Union[None, float]]=None) -> List:
            '''
            Return a list of rewards for the current observations, and performs a SIDE-EFFECT on local and global rewards.
            
            :param obs_for_rew: current global observations (shared+local)
            :param rews_agent: update the local rewards
            :param rews_system: update the global rewards
            :param aoi: Age of information of the current observation

            :return rews: list of rewards associated with the current observations
            '''
            if aoi==0.0:
                self.all_move_per_spot[0.0] = {f.ID:f.n_motions_before_spotting for f in self.flights}

            rews = np.zeros(self.num_flights)

            agents_positions = []
            n_desired_agents_spotting = False
            for obs_set_idx, obs in enumerate(obs_for_rew): # -> the observation order is kept
                agents_positions.append(self.flights[obs_set_idx].position)
                for single_obs_idx, obs_name in enumerate(self.selected_observations):
                    cur_obs = obs[single_obs_idx]
                    cur_f_id = self.flights[obs_set_idx].ID
                    
                    if aoi==None:
                        cur_n_motion_before_spotting = self.flights[obs_set_idx].n_motions_before_spotting #*self.dt
                    else:
                        cur_n_motion_before_spotting = self.all_move_per_spot[aoi][cur_f_id]
                    
                    if obs_name=='source distance':
                        if cur_n_motion_before_spotting!=0:
                            cur_rew = Wa*(1-cur_obs)/cur_n_motion_before_spotting
                        else:
                            cur_rew = Wa*(1-cur_obs)
                        #cur_rew = Wa*(1-cur_obs) 
                        rews_system[obs_set_idx] += cur_rew
                    elif obs_name=='source bearing':
                        cur_rew = Wb*(1-cur_obs)
                        rews_system[obs_set_idx] += cur_rew
                    elif obs_name=='source drift':
                        drift_ref = 0.5
                        if 0>=cur_obs<=drift_ref:
                            cur_rew = (1/drift_ref)*cur_obs # -> when drift_ref==0.5 we have 2*cur_obs
                        else: # -> when drift_ref>cur_obs<=1:
                            cur_rew = 1-cur_obs
                        cur_rew *= Wc
                        rews_system[obs_set_idx] += cur_rew
                    elif obs_name=='source time':
                        cur_rew = Wd*(1-cur_obs)
                        rews_system[obs_set_idx] += cur_rew
                    elif obs_name=='battery':
                        cur_rew = We*(cur_obs)
                        rews_agent[obs_set_idx] += cur_rew
                    elif obs_name=='agents distances':
                        '''
                        Take all the observations related to all the distances between the current
                        agent and all the others (it will make the global observations dependent on the number of the agents):
                        '''
                        for dist_ij in obs_for_rew[obs_set_idx:(obs_set_idx+self.num_flights)]:
                            cur_rew = Wf*dist_ij
                            rews_agent[obs_set_idx] += cur_rew
                            obs_for_rew.remove(dist_ij) # -> removing all the agents distances, allow the foor loop to scroll on the next observation during the next loop cycle
                    elif obs_name=='single-agent coverage area':
                        cur_rew = Wg*cur_obs
                        rews_agent[obs_set_idx] += cur_rew
                    elif obs_name=='AVG agent-all distance':
                        if cur_n_motion_before_spotting!=0:
                            cur_rew = Wh*(1-abs(self.min_avg_all_distances-cur_obs))/cur_n_motion_before_spotting #*self.dt
                        else:
                            cur_rew = Wh*(1-abs(self.min_avg_all_distances-cur_obs))
                        #cur_rew = Wh*(1-abs(self.min_avg_all_distances-cur_obs))
                        rews_system[obs_set_idx] += cur_rew
                    elif obs_name=='AVG agent-not_spotting_agents distance':
                        if cur_n_motion_before_spotting!=0:
                            cur_rew = Wh*(1-abs(self.min_avg_nsa_distances-cur_obs))/cur_n_motion_before_spotting #*self.dt
                        else:
                            cur_rew = Wh*(1-abs(self.min_avg_nsa_distances-cur_obs))
                        #cur_rew = Wi*(1-(self.min_avg_nsa_distances-cur_obs))
                        rews_system[obs_set_idx] += cur_rew
                    elif obs_name=='AVG spotting_agents-source distance':
                        if cur_n_motion_before_spotting!=0:
                            cur_rew = Wh*(1-abs(self.min_avg_sa_distances-cur_obs))/cur_n_motion_before_spotting #*self.dt
                        else:
                            cur_rew = Wh*(1-abs(self.min_avg_sa_distances-cur_obs))
                        #cur_rew = Wl*(1-(self.min_avg_sa_distances-cur_obs))
                        rews_system[obs_set_idx] += cur_rew
                    elif obs_name=='current N spotting agents':
                        # 'cur_obs' is always normalized in [0,1], even in this case where it represents the number of spotting agents:
                        cur_rew = Wm*(1-abs(self.train_cfg.perc_uavs_detecting_the_source-cur_obs))
                        np.clip(cur_rew, a_min=0, a_max=1)
                        rews_system[obs_set_idx] += cur_rew
                        #n_desired_agents_spotting = 1 if cur_obs==1 else False
                    elif obs_name=='current agent spotted the source':
                        if cur_n_motion_before_spotting!=0:
                            cur_rew = (Wn*(cur_obs/cur_n_motion_before_spotting)) 
                        # Avoid division by 0 (if the agent is not spotting the source, then we have 'Wn*cur_obs' instead of 'Wn*(cur_obs/0)'):
                        else:
                            cur_rew = Wn*cur_obs
                        #cur_rew = Wn*(cur_obs)
                        rews_agent[obs_set_idx] += cur_rew 
                        # Reset the number of 'motions' if the source has been spotting by the current UAV:                            
                        if cur_obs: # -> cur_obs in this sub-case is a boolean representing whether the source has been spotting (True) or not (False): 
                            self.flights[obs_set_idx].n_motions_before_spotting = 0 # -> this is reset only if this obs is used!
                            if aoi==None:
                                self.flights[obs_set_idx].n_motions_before_spotting = 0 # -> this is reset only if this obs is used!
                            else:
                                self.all_move_per_spot[aoi][cur_f_id] = 0
                    else:
                        raise NotImplementedError('Reward for observation ' + str(obs_name) + ' is not defined!')

                '''
                In the previous cases, the reward associated with the local part obviously could not be taken from the single agents,
                but it must be taken from its related info included in the global observations (as the reward refers to the centralized part).
                '''

                # Collect all the rewards for the current observation:
                rews[obs_set_idx] = W_agent*rews_agent[obs_set_idx] + W_system*rews_system[obs_set_idx]

            '''
            # Punish every position overlap between the agents:
            agents_positions = [(p.x, p.y) for p in agents_positions]
            pos_overlap_per_agent = [agents_positions.count(p) for p in agents_positions]
            for idx, n_overlap in enumerate(pos_overlap_per_agent):
                rews[idx] -= 0.2*n_overlap
            '''

            return rews

        # Rewards associated with the current ones, i.e., the ones included in the latest global observation:
        rews = single_rewards(obs_for_rew=obs_for_rew, rews_agent=rews_agent, rews_system=rews_system)

        # Non cumulative reward computation:
        all_stored_aoi = list(obs_for_old_rew.keys())
        all_stored_aoi.sort() # -> it is needed to sort all the stored AoIs since if observation delays are enabled they could not be progressively ordered 
        for aoi in all_stored_aoi:
            cur_aoi_time_instant = all_stored_aoi.index(aoi)+1
            # Remove the oldest info associated with the reward if the reward history achieved its maximum storing capacity: 
            if len(self.enode.rew_history)>=self.enode.memory_size:
                oldest_rew_aoi = min(self.enode.rew_history.keys())
                if aoi>=oldest_rew_aoi:
                    if aoi not in self.enode.rew_history:
                        self.enode.rew_history.pop(oldest_rew_aoi)
                        self.enode.rew_history[aoi] = {'non-cumulative': [], 'cumulative': []}
                    else:
                        self.enode.rew_history[aoi]['non-cumulative'] = []
            else:
                if cur_aoi_time_instant>=self.enode.cumulative_rew_size:
                    self.enode.rew_history[aoi] = {'non-cumulative': [], 'cumulative': []}
                else:
                    '''
                    If it has been scrolling the time instant that cannot BACKupdate for the cumulative reward, then
                    keep the old cumulative reward as it is, and reset only the non-cumulative reward (that will be updated later on):
                    '''
                    if aoi in self.enode.rew_history:
                        self.enode.rew_history[aoi]['non-cumulative'] = []
                    else:
                        self.enode.rew_history[aoi] = {'non-cumulative': [], 'cumulative': []}

            old_rews_system = np.zeros(self.num_flights)
            old_rews_agent = np.zeros(self.num_flights)
            old_obs_for_old_rew = obs_for_old_rew[aoi]['norm_obs']

            '''
            BACKupdated rewards recomputed based on the observations in 'enode.memory' storing the old observations,
            and also updated with the delayed and arrived observations:
            '''
            backupdated_rews = single_rewards(obs_for_rew=old_obs_for_old_rew, rews_agent=old_rews_agent, rews_system=old_rews_system, aoi=aoi)

            '''
            Update (or add to) the reward history associated with the currently scrolled AoI (it can never exceed the memory capacity size
            since this check has been already done before entering the scope of this current for-loop):
            '''
            self.enode.rew_history[aoi]['non-cumulative'] = copy.deepcopy(backupdated_rews)

            # Possible Cumulative Reward computation:
            if self.train_cfg.cumulative_reward:
                n_stored_times = len(self.enode.rew_history)

                '''
                Case in which the size of the time steps to look at (for the cumulative reward BAKCupdate) is
                LARGER than the number of the rewards stored in memory (it is set equal to the non-cumulative reward by default):
                ''' 
                if self.enode.cumulative_rew_size>n_stored_times:
                    cur_cumulative_rews = self.enode.rew_history[aoi]['non-cumulative']
                    self.enode.rew_history[aoi]['cumulative'] = copy.deepcopy(cur_cumulative_rews)
                # Case in which the BACKupdate of the cumulative rewards can be performed properly:
                else:
                    '''
                    BACKupdate only if possible, i.e., only if the current aoi is older than the oldest time step
                    that can be looked at when performing the BACKupdating.
                    '''
                    if cur_aoi_time_instant>=self.enode.cumulative_rew_size:
                        # AoIs to look at for BACKupdating the cumulative reward:
                        aoi_to_look_at = all_stored_aoi[cur_aoi_time_instant-self.enode.cumulative_rew_size:cur_aoi_time_instant]
                        cur_stored_non_cumulative_rews = [self.enode.rew_history[t]['non-cumulative'] for t in aoi_to_look_at]
                        # Assign an integer progressive number (from 1 to N=self.enode.memory_size) to the aoi associated with the current observation:
                        ti = len(aoi_to_look_at)+1
                        ti_range = range(1, ti)
                        # Compute the cumulative reward by discounting it by a factor 'gamma':
                        gamma = 0.5
                        assert 0<=gamma<=1, 'The discount factor associated with the cumulative reward must be within [0,1]!'
                        gammas = np.array([pow(gamma, (ti-t)) for t in ti_range])

                        # The number of the rewards must be equal to the number of the UAVs (if not, a padding value to avoid issues in case of data loss must be used!):
                        n_rewards = self.num_flights
                        cur_cumulative_rews = np.zeros(n_rewards)
                        for i in range(n_rewards):
                            # Rewards of the SAME UAV associated with different time instants:
                            rews_i = np.array([rew[i] for rew in cur_stored_non_cumulative_rews])
                            # Cumulative reward formula:
                            cur_cumulative_rews[i] = sum(gammas*rews_i)

                        self.enode.rew_history[aoi]['cumulative'] = copy.deepcopy(cur_cumulative_rews)

            '''
            Set the 'cumulative reward' equal to the 'non-cumulative' one by defualt: it will be actually assigned to the
            cumulative reward value only if it will be possible based on the 'cumulative_reward_size' (i.e., the number of time steps
            to look at to perform the BACKupdate of the cumulative reward):
            '''
            if self.enode.rew_history[aoi]['cumulative']==[]:
                self.enode.rew_history[aoi]['cumulative'] = copy.deepcopy(backupdated_rews)

        if self.train_cfg.cumulative_reward:
            latest_aoi = max(self.enode.rew_history.keys())
            rews = self.enode.rew_history[latest_aoi]['cumulative']

        return rews

    def observation(self) -> List:
        """
        Return the observation of each agent and 'backupdate' the old observations (if enabled and needed).
        NOTE that, for the time being, some redundant code my be present.

        :param: None

        :return: observation of each agent at the current elapsed time: is a list made up by 2 lists, where the first
                 one is the global observation, while the second one is the local observation.
        """
        observations = []
        local_observations = []
        enode_obs = {}

        '''
        The following for-loop is needed in order to get all the LOCAL observations due to the
        movements of all the agents considered at the same time (and not sequentially):
        '''
        for i, f in enumerate(self.flights):
            '''
            Remind that communication between the BS and each UAV occurs every time is possible (i.e., RX=True and Sx=True);
            Also, the BS send back to each UAVs the global info (which is made up by all the info from all the UAVs).
            ''' 
            f.update_local_obs(source=self.source, enode=self.enode, t=self.elapsed_t, external_obs=self.external_obs, geodesics_pol_centroid=self.airspace.geodesics_pol_centroid)
            # Update the local observation buffer (write on it):
            self.communication.obs_buffer_update(flight=f, t=self.elapsed_t, spawn_time=None, obs_delay=self.scenario_observation_delay, obs_loss=self.observation_loss, external_obs=self.external_obs)

        # Consider only the observation which is spawning at the current time instant (delayed observations spwan only if its time to spwan):
        if self.elapsed_t in self.communication.delayed_obs_buffer:
            spawn_observations = self.communication.delayed_obs_buffer[self.elapsed_t]
            n_observations_to_be_spawn = len(spawn_observations)
            '''
            Obviously if the current time elapsed is stored in 'delayed_obs_buffer',
            then it means that the current time elapsed is the spwaning time for the currently considered observation:
            '''
            spawn_time = self.elapsed_t
            '''
            The following double for-loop is used to get all the local observations in the buffer at the current time 
            (they must be different in number and/or coming from the same agent if there is some observation delay),
            and to resend the global observation derived from these local observations received by the ENode:  
            '''
            for i, f in enumerate(self.flights):
                # Obviously there is more than one observation received at the current time, then we need to scroll all of them, if any:
                for spawn_obs in spawn_observations: 
                    # Observations are exchanged only if the spawning time is really the same as the current time instant (it is just a double check made in the scope of 'info_exchange()'):
                    self.communication.info_exchange(enode=self.enode, agent=f, cur_t=self.elapsed_t,
                                                     local_flight_obs=spawn_obs, spawn_obs_time=spawn_time,
                                                     n_observations_to_be_spawn=n_observations_to_be_spawn, energy_constraint=self.energy_constraint)
            '''
            Remove the current spawing time (obviously equal to the current elapsed time) from the buffer containing all the
            observations (delayed, if the 'delay' feature is applied). Do not take care about the arguments except for 'spawn_time',
            which is the only one used in case of removal of an observation from the 'delayed_obs_buffer':
            '''
            self.communication.obs_buffer_update(flight=f, t=self.elapsed_t, spawn_time=self.elapsed_t, obs_delay=self.scenario_observation_delay, obs_loss=self.observation_loss, external_obs=self.external_obs)

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

        def enode_contains_flight_info(ID: float) -> bool:
            """
            Check if the ENode contains info related to the selected 'ID'.

            :param ID: ID of the selected agent

            :return True/False: a boolean indicating if the the ENOde contains info related to the selected agent 'ID'  
            """
            if ID in self.enode.broadcast_obs.keys():
                return True
            else:
                return False

        last_shot_feature = self.enode.broadcast_obs[LAST_SHOT_KEY]
        source_time = last_shot_feature[0]
        source_pos = last_shot_feature[1]
        global_obs_AOI = self.enode.broadcast_obs[AOI_KEY]
        current_spotting_agents = last_shot_feature[2]
        current_n_spotting_agents = len(current_spotting_agents)
        current_n_spotting_agents = Scaling.min_max_normalization(x=current_n_spotting_agents, x_min=0, x_max=self.n_uavs_detecting_the_source)

        elapsed_source_time = self.elapsed_t - source_time
        elapsed_source_time = np.clip(elapsed_source_time, a_min=TIME_MIN, a_max=self.max_time)
        
        # Store the distances of the current agents detecting the current source:
        src_dists_among_spotting_agents = []

        batteries_agents = []
        n_uavs = len(self.flights)
        all_aois = list(self.enode.memory.keys()) 
        if self.elapsed_t in all_aois:
            all_aois.remove(self.elapsed_t)
        if self.elapsed_t!=0.:
            last_t_aoi = max(all_aois)
        else:
            last_t_aoi = self.elapsed_t
        
        for f in self.flights:
            info_i = self.enode.broadcast_obs[f.ID]
            
            if enode_contains_flight_info(ID=f.ID):
                if self.energy_constraint:
                    agent_features_i = info_i[2]
                    battery_agent_i = agent_features_i[1]
                    battery_agent_i = Scaling.min_max_normalization(x=battery_agent_i, x_min=0., x_max=100)
                    batteries_agents.append(battery_agent_i)
            
            if f.ID in current_spotting_agents:
                # Start info:
                old_pos_i = info_i[0][1] # -> info_i[0] gets access to the start features, and info_i[0][1] gets access to the position of the start features

                '''
                It is not needed to check whether 'source_dist' is NaN or not as it is computed after checking that the current
                flight ID is inside the list of the agent that have already spotted the source:
                '''
                source_dist = old_pos_i.distance(source_pos)
                source_dist = np.clip(source_dist, a_min=DIST_MIN, a_max=self.max_d)
                source_dist = Scaling.min_max_normalization(x=source_dist, x_min=DIST_MIN, x_max=self.max_d)
                src_dists_among_spotting_agents.append(source_dist)
        
        if n_uavs!=1:
            if src_dists_among_spotting_agents!=[]:
                avg_src_dist_among_spotting_agents = np.mean(src_dists_among_spotting_agents)
            else:
                avg_src_dist_among_spotting_agents = 0.
        
        if self.energy_constraint:
            avg_batteries = np.mean(batteries_agents)
         
        '''
        ____________________________________________________________ REMINDER: ____________________________________________________________
        IDi_pos_features = self.enode.broadcast_obs[IDi][0][0] # take the START position of the local obs associated with UAV IDi
        IDi_pos_features = self.enode.broadcast_obs[IDi][1][1] # take the TARGET position of the local obs associated with UAV IDi
        IDi_agent_features = self.enode.broadcast_obs[IDi][2] # take the AGENT_FEATURES_KEY, i.e., (track, battery), of the local obs associated with UAV IDi
        IDi_cur_agent_stored_pos = self.enode.broadcast_obs[IDi][3] # take the list 'last_stored_pos' of the local obs associated with UAV IDi
        IDi_AOI = self.enode.broadcast_obs[IDi][4] # take the AoI of the local obs associated with UAV IDi
        
        'avg_src_dist_among_spotting_agents' normalization is not required since the value of the distace of each agent w.r.t. the source has been already normalized
        ____________________________________________________________________________________________________________________________________
        '''
        
        # The agents are scrolled rergardless of the order of their IDs (they are scrolled by following the order in which they are stored):
        obs = []
        obs_local = []
        loss_obs_indeces = []
        obs_to_replace_for_data_loss = [] # -> this will contain all the OLD observations used in placed of the ones that would have been arrived (but actually thei did not, since a data loss occured)  
        for f in self.flights:
            IDi = f.ID
            cur_f = f
            cur_f_idx = self.flights.index(cur_f)
            # if some package loss/delay is applied, than ENode could not contains infos related to some ID (i.e., agent):
            if enode_contains_flight_info(ID=IDi):
                # Position and agent features processing based on the global observations available at the ENode-side:
                '''
                # Possible way to handle data loss:
                if self.elapsed_t!=0. and f.ID in self.communication.data_losses:
                    info_i = self.enode.memory[last_t_aoi]['norm_obs'][f.ID] # -> assign the previous global obs
                else:
                    # compute info_i as already defined here below
                '''
                info_i = self.enode.broadcast_obs[IDi]
                IDi_AOI = info_i[4] # -> take the AoI of the local obs associated with UAV IDi

                # The coverage area of flight 'i' is related to the global observation available at the ENode side (see also the scope of 'compute_coverage_area'):
                coverage_area_i = self.enode.compute_coverage_area(fi=cur_f, pol=self.airspace.cartesian_polygon) # -> possibly BACKupdated in 'utils.py'
                coverage_area_i = Scaling.min_max_normalization(x=coverage_area_i, x_min=0., x_max=math.pi*pow(cur_f.footprint_radius, 2))

                if cur_f.ID in current_spotting_agents:
                    source_spotted = 1
                else:
                    source_spotted = 0

                source_spotted = Scaling.min_max_normalization(x=source_spotted, x_min=0, x_max=1)

                # Start info:
                old_pos_i = info_i[0][1] # -> # info_i[0] gets access to the start features, and info_i[0][1] gets access to the position of the start features

                '''
                It should be taken into account that considering the starting position could lead to get a 0
                distance between 2 agents who are swapping their position if one of them has already arrived at
                its target position and the other one not yet: maybe it could be changed in future.
                '''

                agent_features_i = info_i[2] # -> get the agent features (for the time being it is (self.track, self.battery))
                track_agent_i = agent_features_i[0]
                '''
                It is not needed to check for 'not math.isnan(source_pos) and not math.isnan(last_spotting_UAV)'
                as they depend on each other 'in cascade':
                '''
                if not math.isnan(source_time): 
                    source_dist = old_pos_i.distance(source_pos)
                    source_bearing = bearing(source_pos, old_pos_i)
                    source_drift = drift(source_bearing, track_agent_i)
                else:
                    last_spotting_UAV = 0
                    elapsed_source_time = self.max_time #math.nan
                    source_dist = self.max_d #math.nan 
                    source_bearing = BEARING_MAX #math.nan
                    source_drift = DRIFT_MAX #math.nan
                    pass

                source_dist = np.clip(source_dist, a_min=DIST_MIN, a_max=self.max_d)
                source_dist = Scaling.min_max_normalization(x=source_dist, x_min=DIST_MIN, x_max=self.max_d)
                source_bearing = Scaling.min_max_normalization(x=source_bearing, x_min=BEARING_MIN, x_max=BEARING_MAX) 
                source_drift = Scaling.min_max_normalization(x=source_drift, x_min=DRIFT_MIN, x_max=DRIFT_MAX)
                elapsed_source_time = Scaling.min_max_normalization(x=elapsed_source_time, x_min=TIME_MIN, x_max=self.max_time)
                if self.energy_constraint:
                    battery_agent_i = Scaling.min_max_normalization(x=battery_agent_i, x_min=0., x_max=100)

                # The agent distances are processed based on the global observations available at the ENode-side: 
                agent_distances_i = []
                not_spotting_agents_distances = []
                for IDj in self.flight_IDs:
                    if IDj!=IDi:
                        # If some package loss/delay is applied, than ENode could not contains infos related to some ID (i.e., agent):
                        if enode_contains_flight_info(ID=IDj):
                            info_j = self.enode.broadcast_obs[IDj]

                            # Start info:
                            old_pos_j = info_j[0][1] # -> info_j[0] gets access to the start features, and info_j[0][1] gets access to the position of the start features

                            dist_ij = old_pos_i.distance(old_pos_j)
                            dist_ij = np.clip(dist_ij, a_min=DIST_MIN, a_max=self.max_d)
                            dist_ij = Scaling.min_max_normalization(x=dist_ij, x_min=DIST_MIN, x_max=self.max_d)
                            agent_distances_i.append(dist_ij)
                            if (IDi not in current_spotting_agents) and (IDj not in current_spotting_agents):
                                not_spotting_agents_distances.append(dist_ij) 
                
                if n_uavs!=1:
                    avg_agent_distances_i = np.mean(agent_distances_i)  
                    # 'avg_not_spotting_agents_distances' normalization is not required since the value of the distace of each agent w.r.t. the source has been already normalized
                    if not_spotting_agents_distances!=[]:
                        avg_not_spotting_agents_distances = np.mean(not_spotting_agents_distances)
                    else:
                        avg_not_spotting_agents_distances = 0.

            # List storing the global and local observations associated with the current elapsed time:
            obs = []
            obs_local = []

            if 'source distance' in self.selected_observations: 
                obs.append(source_dist)
            if 'source bearing' in self.selected_observations:
                obs.append(source_bearing)
            if 'source drift' in self.selected_observations:
                obs.append(source_drift)
            if 'source time' in self.selected_observations: 
                obs.append(elapsed_source_time)
            if 'battery' in self.selected_observations:
                if self.energy_constraint:
                    '''
                    In the following line the global observation based on the battery will depend on the number of the agents.
                    To avoid this it can be also considered to not include the battery in the global obseravation and keep it only
                    at local level.
                    '''
                    obs.append(avg_batteries) 
                    # The local battery level must be taken directly from the local agent battery level:
                    local_battery_i = local_obs[AGENT_FEATURES_KEY][1]
                    obs_local.append(local_battery_i)
            if 'agents distances' in self.selected_observations:
                obs += agent_distances_i # -> it is the avg distance among all the agents
            if 'single-agent coverage area' in self.selected_observations:
                obs.append(coverage_area_i)
            if 'AVG agent-all distance' in self.selected_observations:
                obs.append(avg_agent_distances_i) # -> when using this, consider that it make the global obseravtion dependent on the number of the agents
            if 'AVG agent-not_spotting_agents distance' in self.selected_observations:
                obs.append(avg_not_spotting_agents_distances)
            if 'AVG spotting_agents-source distance' in self.selected_observations:
                obs.append(avg_src_dist_among_spotting_agents)
            if 'current N spotting agents' in self.selected_observations:
                obs.append(current_n_spotting_agents)
            if  'current agent spotted the source' in self.selected_observations:
                obs.append(source_spotted)
                # The local osbervation related to the feature spotted/not spotted must be taken directly from the local agent observation:
                source_spotted_local_i = cur_f.local_obs[LAST_SHOT_KEY][2] # -> only take the local observation realted to "spotted/not spotted"
                obs_local.append(source_spotted_local_i)

            if self.learning_action_delay:
                '''
                The history of the actions locally performed has been already updated and it obviously remains unchanged regardless of either
                the global observations and the AoIs. We store in the local observation used for the larning process the past actions as single values
                and not as tuples (dist, heading).
                '''
                for action_idx, action in enumerate(cur_f.actions_memory):
                    # Do not consider in the local observation only the action before the last one (which is the one that is currently being performed):
                    if action_idx==(cur_f.actions_size-1): # -> 'cur_f.actions_size-1' is the index associated with the last (most recent) action stored in 'actions_memory'
                        continue
                    else:
                        cur_action_dist = action[0] # -> action[0] contains the distance
                        cur_action_head = action[1] # -> action[1] containts the heading angle
                        obs_local.append(cur_action_dist)
                        obs_local.append(cur_action_head)
                        
                        if self.num_flights==1:
                            obs.append(cur_action_dist)
                            obs.append(cur_action_head)

                '''
                The only case in which the AoI of the current observation is already in memory is the one associated with the initial case:
                indeed every global AoI will be obviously present at local level since we are assuming no delay at local level (every
                observation at local level is stored and then available instantaneously).
                In this case it is not needed to append any past actions to the current observation, since the observation is old and then
                actions have been already selected and performed at the local level.
                '''

            # Local observation (and thus, since it is local, without any delay) to be used for the learning phase:
            local_observations.append(obs_local)
            # Global observation (and thus, since it is global, made up by possible delayed local observations) to be used for the learning phase:
            observations.append(obs)

            # Possible data Loss Handling:
            if self.observation_loss and IDi in self.communication.data_losses:
                enode_memory_aois = list(self.enode.memory.keys())
                enode_memory_aois.sort()
                last_enode_memory_aoi = enode_memory_aois[-1]
                old_obs_for_data_loss = self.enode.memory[last_enode_memory_aoi]['norm_obs'][cur_f_idx]
                obs_to_replace_for_data_loss.append(old_obs_for_data_loss)
                obs_loss_idx = observations.index(obs)
                loss_obs_indeces.append(obs_loss_idx)
                self.communication.data_losses.remove(IDi)

            '''
            Update the UAV observation memory history based on the current local info (and thus always
            updated based on the current elapsed time). Also, at local level you do not need to take into account data loss:
            '''
            DefAux.memory_update(memory=cur_f.memory, max_size=cur_f.memory_size, aoi=global_obs_AOI, orig_new_v=copy.deepcopy(f.local_obs), new_v=obs_local, env=self) # -> it has been already checked that 'cur_f' performs a side effect on 'f' contained in 'self.flights'

        self.discharged_agents_history[global_obs_AOI] = [copy.deepcopy(self.discharges)]

        '''
        Update the ENODE observation memory history based on the latest 'global_obs_AOI'
        which is obviously equal to the elapsed time (at global level you also need to take into account data loss):
        '''
        DefAux.memory_update(memory=self.enode.memory, max_size=self.enode.memory_size, orig_new_v=copy.deepcopy(self.enode.broadcast_obs), aoi=global_obs_AOI, new_v=observations, env=self, global_obs_update=True, data_losses=self.communication.data_losses, observation_loss=self.observation_loss)

        self.update_deterministic_sys_clock(current_spotting_agents=current_spotting_agents)

        '''
        Possible data Loss Handling:
        you can replace the old observations only here, i.e., when the memory has been updated (also based on some possible data loss).
        The local observation are not modified, since we are assuming that we do not have sensing issues, and hence a data loss can cause
        'problems' only at the global level:
        '''
        for count_idx, loss_idx in enumerate(loss_obs_indeces):
            observations[loss_idx] = obs_to_replace_for_data_loss[count_idx]

        return observations, local_observations 

    # Conflicts are not taken into account for the time being: 
    def update_conflicts(self) -> None:
        """
        Updates the set of flights that are in conflict
        
        :param : None

        :return: None
        """
        # Reset the conflicts' set:
        self.conflicts = set()

        for i in range(self.num_flights - 1):
            if i not in self.discharges:
                for j in range(i + 1, self.num_flights):
                    if j not in self.discharges:
                        distance = self.flights[i].position.distance(self.flights[j].position)
                        if distance < self.min_distance:
                            self.conflicts.update((i, j))

    def update_discharged_agents(self) -> None:
        """
        Updates the set of flights with low battery level.
        
        :param: None

        :return: None
        """

        for i, f in enumerate(self.flights):
            if f.ID not in self.discharges:
                if f.observable_battery_level<=0: # -> you could also choose to use a 'battery_threshold' instead of 0
                    self.discharges.add(f.ID)
            else:
                if f.observable_battery_level>0: # -> you could also choose to use a 'battery_threshold' instead of 0
                    self.discharges.remove(f.ID)

    def update_done(self, current_spotting_agents: List, source_time: float, elapsed_t: float, discharged_agents: List, rew: Optional[Union[List[float], None]] = None, backupdate: Optional[bool]=False) -> Tuple[bool, str]:
        '''
        Check termination status.
        It happens when:
            - Ending:
                (1) The audio source has stopped its motion
            - Success:
                (1) at least 'n' agents detect the source and at the same time there are at least 'm' not discharged agents    
            - Failure:
                (1) time failure
                (2) battery failure
        
        :param current_spotting_agents: number of the agent currently spotting the source
        :param source_time: time since last detection of the source
        :param elapsed_t: current elapsed time since the beginning of the episode
        :param discharged_agents: percentage of discharged UAVs
        :param rew: current reward
        :param backupdate: boolean indicating if BACKupdated must performed or not

        :return done: boolean indicating if thge current episode is terminated or not
        :return info: string containing information about episode
        :return rew: possibly updated reward based on the occuring of a terminal condition for the current episode  
        '''

        if self.energy_constraint: # not backupdate
            self.update_discharged_agents()
        
        if not self.persistent_task:
            done = False
            info = 'NOT A TERMINAL STATE'
            last_n_spotting_agents = len(current_spotting_agents)
            '''
            If the suource is not static then the task ending corresponds to the source stopping,
            otherwise it corresponds to when the time elapsed is equal to 'self.max_len_episode':
            '''
            if self.args.usecase2:
                self.max_episode_len=15
                t_th = 35
            else:
                t_th = 1

            if not self.task_ending:
                ending = False
            else:
                # the info associated with the path travelled by the source obviously does not depend on any feature:
                ending = (self.source.path_traveled or self.elapsed_t==self.max_episode_len) if not self.scnr_cfg.source_static else self.elapsed_t==self.max_episode_len

            success = last_n_spotting_agents>=self.n_uavs_detecting_the_source 
            battery_failure = len(discharged_agents)>=self.max_allowed_discharged_uavs
            elapsed_t_since_last_shot = (elapsed_t-source_time) if not math.isnan(source_time) else elapsed_t
            # Single-agent failure case
            if len(self.flights)==1:
                time_failure = elapsed_t_since_last_shot>=self.max_episode_len
            # Multi-agent failure case:
            else:
                if not backupdate:
                    self.multi_agent_timer_failure += 1 # -> it actually represents the overall time elapsed since the last failure or success
                if self.args.usecase2:
                    #time_failure = self.multi_agent_timer_failure==self.max_episode_len #or elapsed_t_since_last_shot>=t_th
                    time_failure = elapsed_t_since_last_shot>=t_th
                else:
                    # TIME FAILURE 1 -> given by the maximum allowed elapsed time since the "beginning of the episode":
                    #time_failure = self.multi_agent_timer_failure==self.max_episode_len
                    # TIME FAILURE 2 -> given by the elapsed time since the last spotting being less than "t_th":
                    #t_th = 1
                    time_failure = elapsed_t_since_last_shot>=t_th

            #time_over = self.i==self.max_episode_len
            if ending:
                if self.task_ending: # -> it is actually redundant since ending is always set to False when 'self.task_ending=False' !!!!!!!!!!
                    done = True
                    if not self.scnr_cfg.source_static:
                        info = 'SOURCE STOPPED'
                    else:
                        info = 'MAXIMUM EPISODE TIME EXPIRED'
                '''
                else:
                    done = False
                    info = 'NEW SOURCE PATH GENERATED'
                '''
            elif success:
                if self.task_success_enabled:
                    done = True
                    info = 'TASK SUCCESS'
                    if len(self.flights)==1:
                        success_rew = 0.5
                        if type(rew)==np.ndarray:
                            rew[0] += success_rew
                        # backupdate case (=None):
                        else:
                            #rew = success_rew
                            pass
                    else:
                        if not backupdate:
                            success_rew = 2/(self.multi_agent_timer_failure*self.dt) #/self.multi_agent_timer_failure
                        # NOT backupdate case:
                        if type(rew)==np.ndarray:
                            #print('ooooo', self.multi_agent_timer_failure, success_rew)
                            for ri, r in enumerate(rew):
                                rew[ri] += success_rew #0.5
                        # backupdate case (=None)
                        else:
                            #rew = [success_rew for f in self.flights]
                            pass
            elif time_failure:
                if self.time_failure_enabled:
                    done = True
                    info = 'TIME FAILURE'
                    # NOT backupdate case:
                    if len(self.flights)>1:
                        failure_rew = 15 #+(self.multi_agent_timer_failure/8)
                        if type(rew)==np.ndarray:
                            for ri, r in enumerate(rew):
                                rew[ri] -= failure_rew
                        # backupdate case (=None)
                        else:
                            '''
                            for f in len(self.flights):
                                rew[ri] -= failure_rew
                            '''
                            pass

            elif battery_failure:
                if self.battery_failure_enabled:
                    done = True
                    info = 'BATTERY FAILURE'
            else:
                done = False
                info = 'NOT A TERMINAL STATE'
        else:
            done = False
            info = 'PERSISTENT TASK: NOT EXISTING TERMINAL STATE'

        if done and not backupdate:
            self.multi_agent_timer_failure = 0.

        return done, info, rew

    def update_positions(self) -> None:
        """
        Update the position of the agents and the source
        
        :param: None
        
        :return: None
        """

        def update_item_pos(item: Union[Flight, Source]) -> None:
            """
            Update the position either of the UAVs or the audio source

            :param item: Either an UAV or the Audio Sourcce

            :return: None 
            """
            # get current position
            position = item.position

            # Instantaneous actions' case:
            if not self.scenario_action_delay:
                if type(item)==Flight:
                    item.sensing_mode(self.communication)
                    
                    if isinstance(self.external_obs, pd.DataFrame):
                        self.local_obs[AGENT_FEATURES_KEY][1]
                        flight.battery = np.clip(flight.battery, a_min=0, a_max=100)

                    supposed_speed = item.bang_coast_bang(estimate=True, ref_point=item.target)
                    item.airspeed = supposed_speed
                    if self.energy_constraint:
                        update_battery(f)
                    item.airspeed = 0.
                new_x, new_y = (item.target.x, item.target.y)
                new_z = 0.
            # Delayed actions' case:
            else:
                if type(item)==Flight:
                    # Update all the components of the current UAV:
                    new_x, new_y, new_z = item.new_components(communication=self.communication, d=item.traj_type, dt=self.dt)
                else:
                    # Update all the components of the audio source:
                    dx, dy = item.bang_coast_bang(dt=self.dt)
                    new_x, new_y = position.x + dx*self.dt, position.y + dy*self.dt

            # Get new position and advance one time step
            item.position = Point(new_x, new_y)
            item.x_traj_history.append(item.position.x)
            item.y_traj_history.append(item.position.y)
            if type(item)==Flight:
                item.altitude = new_z
                item.z_traj_history.append(item.altitude)
            # The track is automatically oriented towards the current target:
            item.track = item.bearing

        def update_battery(flight: Flight) -> None:
            """
            Update the battery level of the selected UAV.

            :param flight: UAV

            :return: None
            """
            # No delayed actions (i.e., instantaneous):
            if not self.scenario_action_delay:
                '''
                When actions ares instantaneous, then both energy consumption and charging are computed at the same time instant
                (use an estimation to know the battery consumption after instantaneous actions): 
                '''
                energy_consumed, energy_consumption_time = flight.energy_consumption(actual_speed=flight.airspeed, estimate=True, takeoff_land_time=flight.takingoff_time)
                flight.battery -= energy_consumed
                # Actual charging time:
                charging_time = self.dt - energy_consumption_time
                flight.battery += flight.energy_charged(dt=charging_time)
            #  Delayed actions:
            else:
                # Not charging case
                if not flight.is_charging:
                    energy_consumed, _ = flight.energy_consumption(actual_speed=flight.airspeed, estimate=True, takeoff_land_time=flight.takingoff_time)
                    flight.battery -= energy_consumed1
                # Charging case
                else:
                    flight.battery += flight.energy_charged(dt=self.dt)

            # Constrain the battery level within the minimum (i.e., 0) and maximum (i.e., 100) boundary levels:
            flight.battery = np.clip(flight.battery, a_min=0, a_max=100)
            # Then, the observable battery level will be saved autonomously by each agent in its local observation
        
        def update_source_path(src: Source) -> None:
            """
            Update the path of the source.

            :param src: source

            :return: None 
            """
            # Removed the visited spots and their related stopping times:
            src.flat_walk.pop(0)
            src.times.pop(0)
            src.spot_types.pop(0)
            # Case in which the source has travelled all its path:
            if src.flat_walk==[]:
                src.path_traveled = True
            # Case in which the source has not yet travelled all its path:
            else:
                # set the new (first element after the prevoius removal) source start, target and time:
                src.start = copy.deepcopy(src.target)
                src.target = src.flat_walk[0]
                local_time = src.times[0]
                src.track = src.bearing
                src.cur_moving_time = 0

        # SOURCE UPDATE:
        src = self.source
        # Dynamic source case:
        if not src.static:
            # Case in which the source has not yet travelled all its path:
            if not src.path_traveled:
                motion_time = src.bcb_T if src.bcb_satisfied else src.travel_phase_distance/src.groundspeed
                # Instantaneous actions' case:
                if not self.scenario_action_delay:
                    update_item_pos(src)
                    src.shooting_decision
                    update_source_path(src=src)
                # Delayed actions' case:
                else:
                    # Case in which the source has NOT reached its current target yet:
                    if src.cur_moving_time < motion_time:
                        # Update the source position:
                        update_item_pos(src)
                    # Case in which the source has reached its current target:
                    else:
                        # Decrease the source time for its current stop on a spot:
                        src.local_time -= self.dt
                        src.shooting_decision
                        # Case in which the source time on a spot is ended (i.e., it is time to move):
                        if src.local_time <= 0.:
                            update_source_path(src=src)
            # Case in which the source has travelled all its path:
            else:
                src.x_traj_history = []
                src.y_traj_history = []
                # If the audio source is not supposed to stop (i.e., it is not a terminal state condition), then it keeps going to move:
                if not self.task_ending:
                    if self.scnr_cfg.same_env and self.scnr_cfg.source_location!=None:
                        first_hotspot = self.scnr_cfg.source_location
                    else:
                        first_hotspot = src.position
                    self.current_source_walk = Source.source_walk(self.airspace.cartesian_polygon, n_hotspots=self.scnr_cfg.n_hotspots, n_p_min=self.scnr_cfg.n_p_min, n_p_max=self.scnr_cfg.n_p_max, local_d=self.scnr_cfg.local_d, first_hotspot=first_hotspot)
                    self.source = Source.flat_source_walk(scnr_cfg=self.scnr_cfg, walk=self.current_source_walk, dt=self.dt)
                else:
                    src.is_shooting = 0.

        # Static source case:
        else:
            src.shooting_decision

        # FLIGHTS UPDATES:
        for i, f in enumerate(self.flights): 
            # Instantaneous actions' case:
            if not self.scenario_action_delay:
                f.sensing_mode
                update_item_pos(f)
                f.start = copy.deepcopy(f.target)
                f.position = copy.deepcopy(f.start)
                #f.target = self.lps[random.randint(0, len(self.lps)-1)] # --> Just a never ending random selection of different landing point
            # Delayed actions' case:
            else:
                if f.ID not in self.discharges:
                    '''
                    Update the position only if the starting point is different from the target point:
                    this will avoid to produce any unwanted (but also not realistic) energy consumption
                    due to the take-off and landing phase on the same point (obviously only for the case 'start=target'),
                    above all if action delays are taken into account:
                    '''
                    if f.start!=f.target:
                        '''
                        Everything is handled by 'new_components' (that is called inside 'update_item_pos()') and in
                        'traj_coefficient_update' that is called inside 'resolution()'.
                        '''
                        update_item_pos(f)
                        if self.energy_constraint:
                            update_battery(f)

    def step(self, actions: List) -> Tuple[List, List, bool, Dict]:       
        """
        Performs a simulation step.

        :param actions: list of resolution actions assigned to each flight
        
        :return: observation, reward, done status and other information
        """

        # Apply resolution actions:
        self.resolution(actions)

        '''
        Update the current positions before the next observations (i.e., based on the execution of the actions previously selected)
        only if the observations are NOT provided EXTERNALLY. Indeed, if an external observation is provided, it means that the actions
        selected have been suggested to the UAVs, but you do not really know how they actually move until you do not get their next external
        observation. If an external observation is not provided instead you can obviously simulate their motion based on the actions
        selected.
        '''
        if not isinstance(self.external_obs, pd.DataFrame):
            # Update positions
            self.update_positions()

        # Update conflicts (conflicts are not considered for the time being):
        # self.update_conflicts()

        '''
        Case in which either the explicit clock is not enabled OR
        the very first iteration is occurring OR
        the explicit clock is set and the agent clock can actually 'sample the environment' at the current time instant:
        '''
        if (self.scnr_cfg.explicit_clock==False) or (self.elapsed_t==0.) or (self.scnr_cfg.explicit_clock!=False and self.elapsed_t%self.scnr_cfg.agent_clock==0):
            # Compute observation
            obs = self.observation()
        # Case in which the explicit clock is enabled and the agent cannot 'sample the environment' (due to its clock) at the current time instant:
        else:
            '''
            There always exists an old available observation (except for the case of a data loss) as the
            very first time instant is considered in the previous 'if condition'.
            '''
            all_aois = list(self.enode.memory.keys()) 
            last_t_aoi = max(all_aois)
            observations = self.enode.memory[last_t_aoi]['norm_obs']
            local_observations = [f.memory[last_t_aoi]['norm_obs'] for f in self.flights]
            obs = [observations, local_observations]

        '''
        The only updates needed to make the rendering consistent with what is happenring in the EXTERNAL environment,
        are the ones associated with the position and the track angles of the UAVs (all the other info related to the EXTERNAL observations
        have been already updated for the computation of the action selection to communicate to the EXTERNAL UAVs):
        '''
        if isinstance(self.external_obs, pd.DataFrame):
            for i, f in enumerate(self.flights):
                # [0] select the start features, while [1] select the position's feature associated with the start features:
                f.position = f.local_obs[POS_FEATURES_KEY][0][1] # ONLY FOR THE RENDERING -> both start and target are set equal to the communicated EXTERNAL position, thus you can choose either 'start' ([0]) or 'target' ([1]) to update the current EXTERNAL position
                f.altitude = 0. # -> for the time being, it is always set to 0, as the UAVs are assume to communicate only when landed
                f.track = f.local_obs[AGENT_FEATURES_KEY][0] # ONLY FOR THE RENDERING -> Update the track according to the one coomunicated by the EXTERNAL observation
                f.x_traj_history.append(f.position.x)
                f.y_traj_history.append(f.position.y)
                f.z_traj_history.append(f.altitude)

        # Compute reward
        rew = self.reward(obs)

        # Increase absolute time counter
        self.elapsed_t += self.dt
        # Increase steps counter
        self.i += 1

        # Update done set
        done, info, rew = self.update_done(current_spotting_agents=self.enode.broadcast_obs[LAST_SHOT_KEY][2],
                                           source_time=self.enode.broadcast_obs[LAST_SHOT_KEY][0],
                                           elapsed_t=self.elapsed_t,
                                           discharged_agents=self.discharges,
                                           rew=rew)

        if len(self.all_move_per_spot)>self.enode.memory_size*MAX_OBS_DELAY:
            oldest_obs = min(self.all_move_per_spot.keys())
            self.all_move_per_spot.pop(oldest_obs)

        return obs, rew, done, info

    def reset(self) -> List:
        """
        Reset the environment and returns the initial observation.
        
        :param: None
        
        :return: initial observation
        """
        if self.n_airspaces>1 and not self.scnr_cfg.same_env:
            self.generate_airspace

        # Create a set of Point of Interests for the considered airspace (i.e., polygon discretization):
        self.POIs, self.POIs2, self.outer_p = Environment.discretize_pol(pol=self.airspace.cartesian_polygon, step=self.scnr_cfg.step,
                                                                         deg=self.scnr_cfg.deg, n_points= self.scnr_cfg.n_points,
                                                                         step_angle=1)
        self.lps = [p[0] for p in self.POIs.values()] # -> Actual Point of Interests (i.e., landing points 'lps')
        self.POIs_status = [0 for p in self.POIs]
        if self.n_airspaces>1 and not self.scnr_cfg.same_env:
            self.flight_IDs = []

        # Create a source:
        if self.scnr_cfg.source_location!=None:
            first_hotspot = self.scnr_cfg.source_location
        else:
            first_hotspot = None
        self.current_source_walk = Source.source_walk(self.airspace.cartesian_polygon, n_hotspots=self.scnr_cfg.n_hotspots, n_p_min=self.scnr_cfg.n_p_min, n_p_max=self.scnr_cfg.n_p_max, local_d=self.scnr_cfg.local_d, first_hotspot=first_hotspot)
        self.source = Source.flat_source_walk(scnr_cfg=self.scnr_cfg, walk=self.current_source_walk, dt=self.dt)
        if self.n_airspaces==1:
            self.first_init_source = copy.deepcopy(self.source)
        
        # Update the communiction from ENode side at every possible time instant if an explicit clock is not being used: 
        always_update_t = True if self.scnr_cfg.explicit_clock==False else False 
        if self.scnr_cfg.same_env:
            self.source = copy.deepcopy(self.first_init_source)
        
        # Create a communication module:
        self.communication = CommunicationModule(source=self.source, STAR=self.STAR_antennas, always_update_t=always_update_t, dt=self.dt)

        # Create random flights or predefined flights:
        self.flights = []
        tol = self.distance_init_buffer * self.tol
        min_distance = self.distance_init_buffer * self.min_distance
        current_fla = 1
        POIs_left = copy.deepcopy(self.lps)

        # Case in which an external observation is used:
        if isinstance(self.external_obs, pd.DataFrame):
            external_initial_obs = list(zip(self.external_obs['lat'], self.external_obs['lon']))
            self.desired_flights_points = self.airspace.latlon2xy(latlon_ref=self.airspace.geodesics_pol_centroid,
                                                                  latlon_points=external_initial_obs)

            '''
            ________________________________________________________________________________________________
            For the time being, it is always assumed to know the initial observations of all the UAVs during
            the starting phase of the training process  
            ________________________________________________________________________________________________
            '''
            assert (self.train_phase and self.num_flights==len(self.desired_flights_points)) or (self.train_phase), 'The external initial observations must be complete (for all the drones set in the scenario_parameters.ini file!)'

            if all(point==self.desired_flights_points[0] for point in self.desired_flights_points):
                self.same_initial_location = True
            else:
                self.same_initial_location = False
        
        # Case in which an external observation is NOT used:
        else:
            # A random location is selected among all the available landing points:
            if self.same_initial_location:
                same_d_point_for_all = p = random.sample(POIs_left, 1)[0]

        if self.n_airspaces>1 and self.scnr_cfg.same_env:
            pass
        
        # Recompute the flights only if there is more than 1 airspace and you do NOT want to use the same environment:
        else:
            while len(self.flights) < self.num_flights:
                valid = True
                # Case in which an external observation is NOT used:
                if not isinstance(self.external_obs, pd.DataFrame):
                    if not self.same_initial_location:
                        if self.desired_flights_points!=[]:
                            d_point = self.desired_flights_points[len(self.flights)]
                        else:
                            d_point = None
                    else:
                        d_point = same_d_point_for_all

                    flight_id = len(self.flights)+1
                    cur_flight_altitude = self.flight_altitudes[flight_id]
                # Case in which an external observation is used:
                else:
                    d_point = Point(self.desired_flights_points[len(self.flights)][0], self.desired_flights_points[len(self.flights)][1])
                    flight_id = self.external_obs['ID'].iat[len(self.flights)]
                    cur_flight_altitude = self.external_obs['FLA'].iat[len(self.flights)]
                # Sub-case in which the uavs locations have been manually selected:
                if self.scnr_cfg.uavs_locations!=[]:
                    cur_uav_idx_location = len(self.flights)-1 if len(self.flights)>0 else 0
                    d_point = self.scnr_cfg.uavs_locations[cur_uav_idx_location]

                candidate = Flight.generation(scnr_cfg=self.scnr_cfg, train_cfg=self.train_cfg, airspace=self.airspace,
                                              fla=current_fla, source_power=self.source.power,
                                              POIs=self.lps, POIs_left=POIs_left, flight_ID=flight_id,
                                              flight_altitude=cur_flight_altitude, 
                                              desired_point=d_point, tol=tol)
                
                '''
                # Ensure that candidate is not in conflict (conflicts are not considered for the time being):
                for f in self.flights:
                    if candidate.position.distance(f.position) < min_distance:
                        valid = False
                        break
                '''

                if valid:
                    self.flights.append(candidate)
                    current_fla += 1
                    self.flight_IDs.append(flight_id)

                if self.n_airspaces<=1:
                    self.first_flights_init = copy.deepcopy(self.flights)

        if self.scnr_cfg.same_env:
            self.flights = copy.deepcopy(self.first_flights_init)

        self.all_move_per_spot = {}
        self.all_move_per_spot[0.0] = {f.ID:0 for f in self.flights}

        '''
        Set the minimum desired distance between all the agents equal to the normalized footprint radius of the first UAVs.
        UAVs are considered to be homogenerous, and hence we can simply use the footprint radius of one of them:
        '''
        self.min_avg_all_distances = self.flights[0].footprint_radius*6.5 # -> minimum avg distance for all agents
        self.min_avg_nsa_distances = self.flights[0].footprint_radius*6.5 # -> minimum avg distance for all non-spotting agents
        self.min_avg_sa_distances = self.flights[0].footprint_radius*6.5 # -> minimum avg distance for all spotting agents
        
        self.min_avg_all_distances = Scaling.min_max_normalization(x=self.min_avg_all_distances, x_min=DIST_MIN, x_max=self.max_d)
        self.min_avg_nsa_distances = Scaling.min_max_normalization(x=self.min_avg_nsa_distances, x_min=DIST_MIN, x_max=self.max_d)
        self.min_avg_sa_distances = Scaling.min_max_normalization(x=self.min_avg_sa_distances, x_min=DIST_MIN, x_max=self.max_d)

        # Create e ENode:
        self.enode = ENode(train_cfg=self.train_cfg, scnr_cfg=self.scnr_cfg, POIs=self.POIs, POIs2=self.POIs2, lsp=self.lps,flights=self.flights, delayed_obs_buffer=self.communication.delayed_obs_buffer, Rx=True, Sx=True)

        for f in self.flights:
            '''
            Initialize the observation received by each UAV from the ENode right after deploying all the UAVs
            on specific initial landable locations. In this way, also the achievable positions are initialized:
            '''
            f.update_received_obs(enode=self.enode)
            # update the POI status based on the initial deployment of the UAVs:
            self.enode.update_POI_status(status=BUSY, t=0., elem=f.position, flight_ID=f.ID)

        # Initialise steps counter
        self.i = 0
        # Initialize the 'absolute' reference time: 
        self.elapsed_t = 0.

        # Clean conflicts and done sets (remind that, for the time being, the conflicts are not taken into account):
        self.conflicts = set()
        self.discharges = set()

        self.n_airspaces += 1

        # Return initial observation
        return self.observation()

    @property
    def generate_airspace(self) -> None:
        """
        Generate a new arispace (i.e., operative polygon) and assign accordingly a new value
        to the maximum travelling distance associated to it.

        :param: None

        :return: None
        """
        self.n_airspaces += 1
        # Create random or predefined airspace (from external_FIR .csv file):
        if self.scnr_cfg.polygon_verteces==[]:
            self.airspace = ENode.airspace_computation_and_loading(FIR_file=self.FIR_file, FIR_ID=self.FIR_ID,
                                                                   min_area=self.min_area, max_area=self.max_area)
        # Create a manually set airspace:
        else:
            self.airspace = ENode.airspace_computation_and_loading(FIR_file=self.FIR_file, FIR_ID=self.FIR_ID,
                                                                   polygon_verteces=self.scnr_cfg.polygon_verteces)
        # Update the maximum distance 'self.max_d' that can be traveled inside the considered polygon:
        self.max_dist_assignment

    def render(self, mode=None, visible=True, debug=False) -> Union[None, np.ndarray]:
        """
        Renders the environment.

        :param mode: rendering mode
        :param debug: enable or disable the debug mode

        :return cur_im: array representing the environment (in such a way to be always able to save it)
        """

        def draw_text(label) -> "DrawText":
            """
            Aux method to insert text during the render.

            :param label: label to use for the text

            return text: text redy to be drawn in the final render
            """
            class DrawText:
                def __init__(self, label:pyglet.text.Label):
                    self.label=label
                def render(self):
                    self.label.draw()

            return DrawText(label)

        def transformation_matrix(flight: 'Flight'):
            """
            Compute the transformation matric for the currently considered UAV based on it parameters (positions and angles).

            :param flight: the currently considered flight (UAV)

            :return array: array containing the tranformed matrix related to 'flight'
            """
            x = flight.position.x
            y = flight.position.y
            z = flight.altitude
            roll = flight.roll
            pitch = flight.pitch
            yaw = flight.yaw
            return np.array(
                [[cos(yaw) * cos(pitch), -sin(yaw) * cos(roll) + cos(yaw) * sin(pitch) * sin(roll), sin(yaw) * sin(roll) + cos(yaw) * sin(pitch) * cos(roll), x],
                 [sin(yaw) * cos(pitch), cos(yaw) * cos(roll) + sin(yaw) * sin(pitch)
                  * sin(roll), -cos(yaw) * sin(roll) + sin(yaw) * sin(pitch) * cos(roll), y],
                 [-sin(pitch), cos(pitch) * sin(roll), cos(pitch) * cos(roll), z]
                 ])

        if self.viewer is None:

            self.viewer = plt.figure(figsize=(6, 6))
            self.ax = self.viewer.add_subplot(111, projection='3d')
            # Enable the 2D camera view if the action delay is not enabled: 
            if not self.scenario_action_delay:
                self.ax.view_init(azim=-90, elev=90)
                self.ax.set_zticks([])

            # The followiong random colour is used for the UAVs' rotors and their related drawn trajectories:
            for f in self.flights:
                r = lambda: random.randint(0,255) 
                color = '#%02X%02X%02X' % (r(),r(),r())
                self.flights_colors.append(color)

            pol_perimeter = self.airspace.cartesian_polygon.length

            pol_Xcoords, pol_Ycoords = self.airspace.cartesian_polygon.exterior.coords.xy
            pol_Xcoords, pol_Ycoords = np.array(pol_Xcoords[:-1]), np.array(pol_Ycoords[:-1])
            pol_Zcoords = np.array([0. for coord in pol_Xcoords])

            verts = [list(zip(pol_Xcoords, pol_Ycoords, pol_Zcoords))]
            poly = Poly3DCollection(np.array(verts), alpha=0.6, color=SAND)

            minx, miny, maxx, maxy = self.airspace.cartesian_polygon.buffer(30).bounds

            self.ax.set_xlim(minx, maxx)
            self.ax.set_ylim(miny, maxy)
            self.ax.set_zlim(0, max(self.flight_altitudes.values())+1)

            self.ax.add_collection3d(poly)

            # Radius sizes of circles representing the landing points and the source: 
            landing_point_radius = pol_perimeter*0.0125 #*0.00625
            self.not_firing_source_radius_render = pol_perimeter*0.0125
            self.firing_source_radius_render = self.not_firing_source_radius_render + 10*self.not_firing_source_radius_render

            # Landing points (related to the polygon discretization) rendering:
            self.ax.scatter(xs=[p.x for p in self.enode.lps], ys=[p.y for p in self.enode.lps], zs=[0 for p in self.enode.lps], s=landing_point_radius, c='green', alpha=0.8, marker='x')


            # ------------------------------------------------------------------------------------------------
            #                                   FOR DEBUGGING ONLY
            # ------------------------------------------------------------------------------------------------
            if debug:
                # Polygon's centroid position renderering (only for debugging)
                polygon_centroid = self.airspace.cartesian_pol_centroid
                self.ax.scatter(xs=polygon_centroid.x, ys=polygon_centroid.y, zs=0., c='red', alpha=1., marker='o')
                for i, p in enumerate(self.enode.lps):
                    # Check the interpolation of the polygon's centroid with the landpoints:
                    centroid_plan = LineString([self.airspace.cartesian_pol_centroid, p])
                    x_points = np.linspace(polygon_centroid.x, p.x, 10)
                    y_points = np.linspace(polygon_centroid.y, p.y, 10)
                    self.ax.plot(x_points, y_points, color='red')
                    # Landpoints labels (IDs) redering: -> it slows down the rendering
                    self.ax.text(p.x, p.y, 0., s=i+1, c='red', size=10) # (255, 165, 0, 255)
            # Lock rendering at the beginning for debugging:
            # ------------------------------------------------------------------------------------------------

        # SOURCE RENDERING (to be changed if you want to add the possibility to handle multiple sources):
        source_path_render = []
        source_render = []
        source_firing = []
        # Draw the souce path (if the source motion is enabled):
        if self.source.is_shooting==1:
            source_firing = self.ax.scatter(xs=self.source.position.x, ys=self.source.position.y, zs=0., s=self.firing_source_radius_render, alpha=0.7, facecolors='none', edgecolors='violet', linewidths=3) # -> to enable the NOT filled option, then use "facecolors='none'"                
        source_render = self.ax.scatter(xs=self.source.position.x, ys=self.source.position.y, zs=0., c=self.source_color, s=self.not_firing_source_radius_render, alpha=0.8) # to enable the NOT filled option, then use "facecolors='none', edgecolors='red'"
        source_path_render, = self.ax.plot(self.source.x_traj_history, self.source.y_traj_history, 0., color=self.source_color, linestyle=':')

        # UAVS RENDERING:
        n_uavs = len(self.flights)
        uavs_rotors_render = [None for i in range(n_uavs)]
        uavs_structure1_render = [None for i in range(n_uavs)]
        uavs_structure2_render = [None for i in range(n_uavs)]
        uavs_ID_render = [None for i in range(n_uavs)]
        uavs_paths_render = [None for i in range(n_uavs)]
        uavs_footprint_render = [None for i in range(n_uavs)]
        for i, f in enumerate(self.flights):
            # Flights rendering:
            T = transformation_matrix(flight=f)

            p1_t = np.matmul(T, f.p1)
            p2_t = np.matmul(T, f.p2)
            p3_t = np.matmul(T, f.p3)
            p4_t = np.matmul(T, f.p4)

            '''
            ------------------------------------------------------------------------------------------
            # This draw a squared surface (over the X-structure) similar to the ones of the P-drones:
            x = [p1_t[0], p2_t[0], p3_t[0], p4_t[0]]
            y = [p1_t[1], p2_t[1], p3_t[1], p4_t[1]]
            z = [p1_t[2], p2_t[2], p3_t[2], p4_t[2]]
            
            uavs_pdrone_surface_render[i] = self.ax.plot_trisurf(x, y, z, color='#04508c')
            ------------------------------------------------------------------------------------------
            '''

            # Draw the UAV rotors:
            uavs_rotors_render[i], = self.ax.plot([p1_t[0], p2_t[0], p3_t[0], p4_t[0]],
                                          [p1_t[1], p2_t[1], p3_t[1], p4_t[1]],
                                          [p1_t[2], p2_t[2], p3_t[2], p4_t[2]], 'o', markersize='6.5', fillstyle='none', color=self.flights_colors[i])
            
            # Draw the UAV 'X-structure':
            uavs_structure1_render[i], = self.ax.plot([p1_t[0], p2_t[0]], [p1_t[1], p2_t[1]],
                                                 [p1_t[2], p2_t[2]], marker='|', color='#333333') # 'rod' 1
            uavs_structure2_render[i], = self.ax.plot([p3_t[0], p4_t[0]], [p3_t[1], p4_t[1]],
                                                      [p3_t[2], p4_t[2]], marker='|', color='#333333') # 'rod' 2

            # Draw the UAV footprint:
            footprint_area = math.pi*f.footprint_radius
            theta = np.linspace(0, 2*np.pi , 150)
            if f.sensing_range_enabled:
                x_foot = f.position.x + f.footprint_radius*np.cos(theta)
                y_foot = f.position.y + f.footprint_radius*np.sin(theta)
                uavs_footprint_render[i], = self.ax.plot(x_foot, y_foot, 0., color=self.flights_colors[i], alpha=0.9)

            # Draw the UAV path (trajectory):
            uavs_paths_render[i], = self.ax.plot(f.x_traj_history, f.y_traj_history, f.z_traj_history, color=self.flights_colors[i], linestyle=':')

            # Draw the UAV ID:
            uavs_ID_render[i] = self.ax.text(f.position.x, f.position.y, f.altitude+0.15, s=f.ID, c=self.flights_colors[i], size=13) # -> style='italic', size=13
        
        if visible:
            plt.pause(0.0001)

        self.viewer.canvas.draw()

        cur_im = np.array(self.viewer.canvas.renderer._renderer)
        # To be changed if you want to handle multiple sources:
        source_render.remove()
        source_path_render.remove()
        if source_firing!=[]:
            source_firing.remove()
        for i in range(n_uavs):
            uavs_rotors_render[i].remove()
            uavs_structure1_render[i].remove()
            uavs_structure2_render[i].remove()
            uavs_paths_render[i].remove()
            uavs_ID_render[i].remove()
            if uavs_footprint_render[i]!=None:
                uavs_footprint_render[i].remove() 
        
        # Always return an array representing the environment in such a way to be always able to save it:
        return cur_im

    def close(self) -> None:
        """
        Close the viewer

        :param: None
        
        :return: None
        """
        if self.viewer is not None:
            plt.close(self.viewer)
            self.viewer = None
            self.ax = None
