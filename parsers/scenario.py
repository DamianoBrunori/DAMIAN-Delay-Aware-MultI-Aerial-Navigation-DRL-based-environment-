import os
import shutil
import sys
from typing import Optional, Tuple, List, Dict, Union
import numpy as np
import pathlib
import random
import ast
from shapely.geometry import Point
from configparser import ConfigParser
from muavenv.global_vars import LOGS_FOLDER, AGENT_CLOCK_FREQ, ENODE_CLOCK_FREQ, INITIAL_SYSTEM_CLOCK_FREQ

class ScenarioConfig():
    """
    Config class
    """
    def __init__(self, settings_src='settings/', settings_dst='settings/'):
        self.settings_src = settings_src + 'scenario_parameters.ini'
        if settings_dst!=None:
            self.settings_dst = settings_dst +'scenario_parameters.ini'
        else:
            self.settings_dst = settings_dst
        saving_folder = LOGS_FOLDER
        saving_filename ='scenario_run.txt'
        saving_filepath = saving_folder + saving_filename # saving_folder + '/' + saving_filename
        if not os.path.isdir(saving_folder):
            os.mkdir(saving_folder)
        config = ConfigParser()
        l = config.read(self.settings_src)
        if l==[]:
            print("ERROR cannot load scenario settings init file")
            sys.exit(1)
        
        self.str_trues = ['true', 'True']
        self.str_falses = ['false', 'False']
        self.explicit_clock_values = [False, 'fixed', 'deterministic']

        # [Time Setting]
        time = config['Time Setting']
        self.dt = float(time['dt'])
        self.time_unit = time['unit_measure']
        self.explicit_clock = self.str_to_bool(time['explicit_clock'])
        """
        ________________________________________________________________________________________________________________
        The clock frequency is always converted into seconds regardless of the unit measure used for the 'time',
        hence the variable storing the time will be expressed in seconds
        (i.e., 'dt=2' with 'unit_measure=h' --> dt=7200). In this way there are no conversion operations to perform
        during the execution of the bang-coast-bang motion):
        ________________________________________________________________________________________________________________
        """
        self.agent_clock = self.freq2time(freq_v=AGENT_CLOCK_FREQ, conv2sec=True)
        self.enode_clock = self.freq2time(freq_v=ENODE_CLOCK_FREQ, conv2sec=True)
        self.initial_system_clock = self.freq2time(freq_v=INITIAL_SYSTEM_CLOCK_FREQ, conv2sec=True)   

        # [Operative Polygon]
        polygon = config['Operative Polygon']
        self.min_area = int(polygon['min_area'])
        self.max_area = int(polygon['max_area'])
        self.polygon_verteces = ast.literal_eval(polygon['polygon_verteces']) 
        self.same_env = polygon['same_env']
        self.same_env = self.str_to_bool(self.same_env)
        self.n_points = polygon['n_landing_points']
        self.step = float(polygon['step'])
        self.deg = int(polygon['deg'])

        # [Audio Source]
        audio = config['Audio Source']
        self.source_static = audio['static']
        self.source_constant = audio['constant']
        self.source_location = ast.literal_eval(audio['source_location'])
        self.db_level = float(audio['source_db_level'])
        self.src_global_max_speed = float(audio['global_max_speed'])
        self.src_global_max_acc = float(audio['global_max_acc'])
        self.src_local_max_speed = float(audio['local_max_speed'])
        self.src_local_max_acc = float(audio['local_max_acc'])
        self.n_hotspots = int(audio['n_hotspots'])
        self.n_p_min = int(audio['n_p_min'])
        self.n_p_max = int(audio['n_p_max'])
        self.local_d = float(audio['local_d'])

        self.source_static = self.str_to_bool(self.source_static)
        self.source_constant = self.str_to_bool(self.source_constant)

        # [Communication]
        communication = config['Communication']
        self.STAR_antennas = communication['STAR']
        self.STAR_antennas = self.str_to_bool(self.STAR_antennas)
        
        # [UAVs]
        uavs = config['UAVs']
        self.num_flights = int(uavs['n_uavs'])
        self.min_fl = float(uavs['min_fl'])
        self.max_fl = float(uavs['max_fl'])
        self.sep_fl = float(uavs['sep_fl'])
        self.sensing_time = float(uavs['sensing_time'])
        self.fls_available = np.arange(self.min_fl, self.max_fl, self.sep_fl)
        self.n_fls_available = len(self.fls_available)
        self.same_initial_location = uavs['same_initial_location']
        self.same_initial_location = self.str_to_bool(self.same_initial_location)
        self.uavs_locations = ast.literal_eval(uavs['uavs_locations'])
        self.energy_constraint = uavs['energy_constraint']
        self.energy_constraint = self.str_to_bool(self.energy_constraint)
        self.max_speed = float(uavs['max_speed'])
        self.max_acc = float(uavs['max_acc'])
        self.min_db_level = float(uavs['min_db_level'])
        self.p_eng = float(uavs['p_eng'])
        self.v_ref = float(uavs['v_ref'])
        self.p_bat = float(uavs['p_bat'])
        self.p_bat_charging = float(uavs['p_bat_charging'])
        self.b_efficiency = float(uavs['b_efficiency'])
        self.b_res = int(uavs['b_res'])
        self.action_delay = uavs['action_delay']
        self.action_delay = self.str_to_bool(self.action_delay)
        self.observation_delay = uavs['observation_delay']
        self.observation_delay = self.str_to_bool(self.observation_delay)
        self.observation_loss = uavs['observation_loss']
        self.observation_loss = self.str_to_bool(self.observation_loss)

        self.validate_params()
        self.summary_log(saving_filepath)

        # Flight levels assignment (FLs are randomly sampled among the available ones avoiding overlapping UAVs on the same FL):
        fls_assigned = random.sample(list(self.fls_available), self.num_flights)
        # Flight altitudes for each flight (FLA)
        self.flight_altitudes = {i:fls_assigned[i-1] for i in range(1, self.num_flights+1)}  
        #self.flight_altitudes = self.flight_altitudes[0]

    def freq2time(self, freq_v, conv2sec: Optional[bool] = False) -> float:
        """
        Convert the frequency 'freq_v' into a sampling time based on the unit measure (seconds, minutes or hours)
        selected in 'scenario_parameters.ini'.

        :param  freq_v: value expressed in [Hz]
        :param  conv2sec: a boolean value specifying if performing the conversion directly into seconds regardless of what has been used for 'self.time_unit'

        :return time: the sampling time (expressed according to the selected unit measure) equivalent to 'freq_v'
        """

        if conv2sec:
            time = 1/freq_v
        else:
            if self.time_unit=='s':
                time = 1/freq_v
            elif self.time_unit=='m':
                time = 1/(freq_v*60)
            else:
                time = 1/(freq_v*3600)

        return round(time, 1)

    def str_to_bool(self, v: str) -> Union[bool, str]:
        if v in self.str_trues:
            return True
        elif v in self.str_falses:
            return False
        else:
            return v

    def validate_params(self) -> None: 
        assert 0<=self.min_area<=self.max_area, 'Airspace area must be consistent in such a way that 0<=min_area<=max_area!' 
        assert self.time_unit=='s' or self.time_unit=='m' or self.time_unit=='h', 'Time unit must be either "s" (seconds) or  "m" (minutes) or "h" (hours)!'
        n_points_type = self.n_points
        try:
            self.n_points = int(self.n_points)
        except:
            assert self.n_points=='all', 'You can set either a number of discrete landing points or "all" to assume that all the polygon surface is completely landable!'
        assert self.STAR_antennas==True or self.STAR_antennas==False, 'STAR communication type must be set either to True or False!'
        assert self.source_static==True or self.source_static==False, 'Source static motion must be set either to True (not moving) or False (dynamic)!'
        assert self.source_constant==True or self.source_constant==False, 'Source static motion must be set either to True (not moving) or False (dynamic)!'
        assert self.min_fl>0 and self.max_fl>self.min_fl and self.n_fls_available>=self.num_flights, 'Impossible to assign properly each UAV to a different FL based on the flight levels features selected!'
        assert self.sensing_time>=0, 'The sensing time of each UAV must be equal or greater than 0!'
        assert self.same_initial_location==True or self.same_initial_location==False, 'The initial location can be equal for all the agent: it must be set either to True or False!'
        assert self.energy_constraint==True or self.energy_constraint==False, 'Energy constraint must be set either to True or False!'
        assert self.explicit_clock==False or self.explicit_clock=='fixed' or self.explicit_clock=='deterministic', 'The scenario feature explicit_clock is set equal to: ' + str(self.explicit_clock) \
                                                                                                        + '\nIt can be set only equal to one of the following parameters: ' + str(self.explicit_clock_values)
        assert (self.explicit_clock==False) or (self.explicit_clock!=False and self.agent_clock%self.dt==0 and self.enode_clock%self.dt==0 and self.initial_system_clock%self.dt==0), 'The initial system clock (and thus the system clock in general), the agent clock and the enode clock must be multiple of the simulation step "dt"! We assume that the simulation step is always small enough to allow you to "see everything" and thus that there is no delay between the simulation step and system clock.'
        assert self.action_delay==False or (self.action_delay==True and self.explicit_clock==False), 'If actions are considered to be "instantaneous" (i.e., action_delay=True), then it is not possible to set an explicit clock since actions can take place every time step regardless of the system clock!' 
        assert type(self.polygon_verteces)==list or self.polygon_verteces==[], 'The polygon verteces must be either a list of 2D tuple points or an empty list!'
        # There is no check on the UAVs locations, namely whether they are or not inside the generated polygon:
        assert self.uavs_locations==[] or (type(self.uavs_locations)==list and len(self.uavs_locations)==self.num_flights), 'The uavs locations must be either a list of 2D tuple points or an empty list: if not an empty list, then the number of the points must matcth the number of the selected agents!' 
        self.uavs_locations = [Point(uav_loc) for uav_loc in self.uavs_locations]
        if self.source_location=='none' or self.source_location=='None' or self.source_location==None:
            self.source_location = None
        else:
            self.source_location = Point(self.source_location)
        # There is no check on the audio source location, namely whether it is or not inside the generated polygon:
        assert self.source_location==None or type(self.source_location)==Point, 'The location for the audio source must be None or a float value!' 

    def printkey(self, k: str) -> None:
        print("  %s = %r" %(k,self.__dict__[k]))

    def summary_log(self, file: pathlib.Path) -> None:
        print("\n============================= Scenario Parameters =============================\n")

        attrs = vars(self)
        section = ''
        with open(file, 'w') as f:
            for key, value in attrs.items():
                if key=='str_trues' or key=='str_falses':
                    continue
                else:
                    if key=='dt':
                        section = 'Time Setting\n'
                    elif key=='min_area':
                        section = '\nOperative Polygon\n'
                    elif key==('src_global_max_speed'):
                        section = '\nAudio Source\n'
                    elif key=='STAR_antennas':
                        section = '\nCommunication\n'
                    elif key=='num_flights':
                        section = '\nUAVs\n'
                    else:
                        section = ''
                    if section!='':
                        print(section)
                        f.write(section)
                    print('\t' + key, value)
                    f.write('\t' + str(key) + ': ' + str(value) + '\n')

        if self.settings_dst!=None:
            # Copy the src into dst only if src file is different from dst file (this happens only if a usecase has not been selected):
            if self.settings_src!=self.settings_dst:
                shutil.copyfile(self.settings_src, self.settings_dst)
        
        print("\n================================================================================")
        print("Scenario parameters settings saved in " + str(file) + "\n\n")
