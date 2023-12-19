"""
Definitions module
"""
import warnings
import time
import numpy as np
import pathlib
import csv
from haversine import haversine, inverse_haversine, Unit
from shapely.geometry import Point, Polygon, box
from sklearn.cluster import KMeans
from dataclasses import dataclass, field
import muavenv.units as u
from muavenv.utils import *
from muavenv.global_vars import *
from parsers.training import *
from parsers.external_info import *
import math
from math import cos, sin
import random
from typing import Optional, Tuple, List, Dict, Union
import copy
from csv import writer
import abc

# For the time being, some redundant code or unused methods could be present: to be removed in future work.

@dataclass
class Airspace:
    """
    Airspace class
    """
    cartesian_polygon: Polygon
    cartesian_pol_centroid: Point
    geodesics_pol_centroid: Union[Tuple[float, float], None]

    @classmethod
    def manual(cls, polygon_verteces: List[Tuple]) -> Polygon:
        """
        Manually creates an airspace sector set by using specific verteces.

        :param polygon_verteces: a list containing all the desired polygon verteces

        :return: a manually defined airspace
        """
        polygon = Polygon(polygon_verteces).convex_hull

        xy_polygon_centroid = polygon.centroid

        return cls(cartesian_polygon=polygon, cartesian_pol_centroid=xy_polygon_centroid, geodesics_pol_centroid=None)

    @classmethod
    def random(cls, min_area: float, max_area: float) -> Polygon:
        """
        Creates a random airspace sector with min_area<area<=max_area.

        :param max_area: maximum area of the sector (in nm^2)
        :param min_area: minimum area of the sector (in nm^2)
        
        :return: random airspace
        """
        R = math.sqrt(max_area / math.pi)

        def random_point_in_circle(radius: float) -> Point:
            alpha = 2 * math.pi * random.uniform(0., 1.)
            r = radius * math.sqrt(random.uniform(0., 1.))
            x = r * math.cos(alpha)
            y = r * math.sin(alpha)
            return Point(x, y)

        p = [random_point_in_circle(R) for _ in range(3)]
        polygon = Polygon(p).convex_hull

        while polygon.area < min_area:
            p.append(random_point_in_circle(R))
            polygon = Polygon(p).convex_hull

        xy_polygon_centroid = polygon.centroid

        return cls(cartesian_polygon=polygon, cartesian_pol_centroid=xy_polygon_centroid, geodesics_pol_centroid=None)

    @staticmethod
    def latlon2dst_bear(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
            """
            Compute the distance and the bearing between p1 and p2 (both expressed in (lat,lon) coordinates).

            :param p1: first point expressed in (lat,lon)
            :param p2: second point expressed in (lat,lon)

            :return distance, bearing: distance [m] and bearing [rad] between p1 and p2
            """
            lat1 = p1[0]
            lon1 = p1[1]
            lat2 = p2[0]
            lon2 = p2[1]

            # phi (latitudes), lambda (longitudes) from degrees to radians:
            phi1 = math.radians(lat1)
            phi2 = math.radians(lat2)

            lambda1 = math.radians(lon1)
            lambda2 = math.radians(lon2)

            delta_phi = phi2-phi1
            delta_lambda = lambda2-lambda1

            def distance() -> float:
                """
                Compute the distance between 2 refrence points with coordinates expressed in (lat,lon).
                """
                R = 6371e3 # metres
                # Use the haversine formula to find the distance between two points:
                d = haversine(p1, p2, unit=Unit.METERS)

                return d
            
            def bearing() -> float:
                """
                Compute the bearing between 2 refrence points with coordinates expressed in (lat,lon).
                """
                y = math.sin(lambda2-lambda1)*math.cos(phi2)
                x = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(lambda2-lambda1);
                theta = math.atan2(y, x) # it is already in radians
                compass = (theta + u.circle) % u.circle

                return compass

            return distance(), bearing()

    @staticmethod
    def latlon2xy(latlon_ref: List[float], latlon_points: List[float]) -> List[float]:
        """
        Coordinates transformation from (lat,lon) into (X,Y) of a desired set of points.

        :param latlon_ref: (lat,lon) reference point to compute the coordinates transofrmation
        :param latlon_points: (lat,lon) points list to convert into (X,Y) coordinates

        :return xy_points: list of points expressed in (X,Y) coordinates
        """
        xy_points = []
        for latlon in latlon_points:
            # Compute bearing and distance between the FIR centroid current and the FIR boundary point, both expressed in (lat,lon):
            dst, theta = Airspace.latlon2dst_bear(p1=latlon_ref, p2=latlon)
            # Compute the (X,Y) coordinates associated with the current FIR boundary point, both expressed in (lat,lon):
            x = dst*math.sin(theta)
            y = dst*math.cos(theta)
            xy_points.append((x, y))

        return xy_points

    @classmethod
    def external_airspace(cls, FIR_file: pathlib.Path, FIR_ID: str) -> Polygon:
        """
        Create an airspace (based on (X,Y) coordinates) from a Flight Information Region file
        containing the main airpsace features (based on (X,Y) coordinates).

        :param FIR_file: FIR .csv file
        :param FIR_ID: IF of the FIR to extract from FIR_file

        :return xy_FIR: airpsace based on (X,Y) coordinates
        """

        def get_latlon_FIR(df: pd.DataFrame, FIR_ID: str = '') -> List[Tuple[float, float]]:
            """
            Extract (lat,lon) coordinates of FIR_ID from DataFrame df.

            :param df: DataFrame containing the desired FIR
            :param FIR_ID: ID of the desired FIR

            :return FIR_latlon: (lat,lon) coordinates of the FIR with ID=FIR_ID
            """
            def get_latlon(df: pd.DataFrame, FIR_ID: str) -> List[Tuple[float, float]]:
                """
                Collect all the (lat,lon) coordinates of all the boundaries of FIR_ID
                from df.

                :param df: DataFrame containing the desired FIR
                :param FIR_ID: ID of the desired FIR

                :return FIR_latlon: list containing all the boundaries (lat,lon) coordinates of the FIR with ID=FIR_ID
                """
                FIR_IDs = df['Airspace ID'].tolist()
                lats = df['Latitude'].tolist()
                lons = df['Longitude'].tolist()
                latslons = []
                for i, ID in enumerate(FIR_IDs):
                    if FIR_ID!='':
                        if ID==FIR_ID:
                            cur_latlon = (lats[i], lons[i])
                            latslons.append(cur_latlon)
                    else:
                        latslons.append(cur_latlons)

                return latslons

            FIR_latlon = get_latlon(df=df, FIR_ID=FIR_ID)

            return FIR_latlon

        def pol_latlon2xy(latlon_FIR: List[Tuple[float, float]]) -> Tuple[Polygon, Point, Tuple[float, float]]:
            """
            Coordinates transformation from (lat,lon) into (X,Y) and centroids computation
            for both the cartesian airspace (X,Y) and the original FIR (lat,lon).

            :param latlon_FIR: List of the boundary point of the FIR expressed in (lat,lon) coordinates

            :return xy_airspace: Polygon containing all the boundary points of the FIR expressed in (X,Y) coordinates 
            """ 
            air_pol = Polygon(latlon_FIR)
            latlon_air_centroid = air_pol.centroid
            latlon_air_centroid_list = [latlon_air_centroid.x, latlon_air_centroid.y]
            
            # Compute the (X,Y) coordinates of the current airspace by transforming the (lat,lon) coordinates of the FIR:
            xy_airspace = Airspace.latlon2xy(latlon_ref=latlon_air_centroid_list, latlon_points=latlon_FIR)
            xy_airspace = Polygon(xy_airspace)
            xy_air_centroid = xy_airspace.centroid

            return xy_airspace, xy_air_centroid, latlon_air_centroid_list

        # Extract the DataFrame associated with FIR_ID contained inside the FIR file:
        ex_info = ExternalInfo(FIR_file=FIR_file, FIR_ID=FIR_ID)
        # Depending on the extension of the selectede FIR file, it can return either a Dataframe or directly the latlon FIR list: 
        df = ex_info.FIR_file_read
        if isinstance (df, pd.DataFrame):
            # Extract the boundary points of FIR_ID expressed in (lat,lon) coordinates:
            FIR_lat_lons = get_latlon_FIR(df=df, FIR_ID=FIR_ID)
        else:
            FIR_lat_lons = df
        # Get the boundary points of FIR_ID expressed in (X,Y) coordinates:
        xy_FIR, xy_FIR_centroid, latlon_FIR_centroid = pol_latlon2xy(latlon_FIR=FIR_lat_lons)

        return cls(cartesian_polygon=xy_FIR, cartesian_pol_centroid=xy_FIR_centroid, geodesics_pol_centroid=latlon_FIR_centroid)

class ENode:
    """
    ENode class
    """
    def __init__(self, train_cfg: "TrainingConfig", scnr_cfg: "ScenarioConfig", POIs: List, POIs2: List, lsp: List, flights: List, delayed_obs_buffer: Dict, Rx: Optional[bool] = False, Sx: Optional[bool] = False, cur_t: Optional[float] = 0., n_stored_pos: int = 2) -> None:
        self.train_cfg = train_cfg
        self.scnr_cfg = scnr_cfg
        self.POIs = POIs # POIs by ID
        self.POIs2 = POIs2 # POIs by point
        self.lps = lsp
        self.flights = flights
        self.Rx = Rx
        self.Sx = Sx
        self.cur_t = cur_t
        self.n_stored_pos = n_stored_pos
        self.broadcast_obs = {} # -> observation built based on all the observations collected from all the UAVs
        self.broadcast_obs[LAST_SHOT_KEY] = (math.nan, math.nan, [])
        self.queue_info = [] # -> queue storing the future needed POIs updates
        # Number of communications received (at most as many as the number of the agents): it starts from scratch at each new communication time instant:
        self.n_obs_received = 0
        self.current_spotting_agents = [] # -> list of the agents that have spotted the source at the current time instant
        self.source_info_buffer = {f.ID:(math.nan, math.nan, 0) for f in self.flights}
        self.next_actions = []
        self.clock = self.scnr_cfg.enode_clock
        self.memory = {}
        self.memory_size = self.train_cfg.batch_size
        self.delayed_obs_buffer = delayed_obs_buffer
        self.rew_history = {}
        self.cumulative_rew_size = self.train_cfg.cumulative_rew_size

        # Initial global info (collected from all the local UAVs' observations) on the ENode-side: 
        for i, f in enumerate(self.flights):
            local_obs = f.local_obs # -> each local observation is uniquely identified by the agent ID and the AoI associated to that observation
            if local_obs not in delayed_obs_buffer.values():
                local_ID = local_obs[ID_KEY]
                self.broadcast_obs[local_ID] = copy.deepcopy(local_obs[POS_FEATURES_KEY])
                self.broadcast_obs[local_ID].append(copy.deepcopy(local_obs[AGENT_FEATURES_KEY]))
                cur_agent_stored_pos = [copy.deepcopy(f.start) for p in range(self.n_stored_pos)]
                self.broadcast_obs[local_ID].append(cur_agent_stored_pos)
                # AoI of the local observation received:
                self.broadcast_obs[local_ID].append(copy.deepcopy(local_obs[AOI_KEY]))
        
        '''
        On ENode-side we only look at the time (but still indicated with AoI) at which we generate (i.e., receive)
        the (cumulative/global) info, and not at when the single local observations included in it have been generated:
        '''
        self.broadcast_obs[AOI_KEY] = self.cur_t

    def update_source_buffer(self, local_ID: Union[int, str], shot_info: Tuple):   
        self.source_info_buffer[local_ID] = shot_info

    def compute_coverage_area(self, fi: "Flight", pol: Airspace, aoi: Optional[Union[float, None]] = None) -> float:
        """
        Return the actual coverage area of the agent 'fi' w.r.t. the other agents and the operative polygon during the
        latest 'n_stored_pos' positions. The info associated with 'fi' are the ones availabled at the ENode-side in 'self.broadcast_obs'.

        :param fi: i-th flight
        :param pol: operative polygon
        :param aoi: Age of Information associated with the currently analyzed 'coverage area' info
    
        :return last_n_pos_Ai: total coverage area by of the current agent 'fi' based on its sensing range during the
                               latest 'n_stored_pos' positions.
        """

        def agents_overlapping_area(fi: Point, fi_footprint_radius: float, fj: Point, fj_footprint_radius: float) -> float:

            """
            Compute the overlapping area between two agents footprints.

            :param fi: Center of the sensing range of the agent 'i'
            :param fi_footprint_radius: radius of the footprint of the agent 'i'
            :param fj: Center of the sensing range of the agent 'j'
            :param fj_footprint_radius: radius of the footprint of the agent 'j'

            :return aij: overlapping area between two agents footprints
            """
            aij = 0. # -> overlapping area (if any) of the i-th agent with all the other agents
            
            # Setting the smaller and the larger sensing radius among i-th and j-th agents: 
            if fi_footprint_radius<fj_footprint_radius:
                r = fi_footprint_radius
                R = fj_footprint_radius
            else:
                r = fi_footprint_radius
                R = fj_footprint_radius
            
            dij = fi.distance(fj) # -> distance between i-th agent and j-th agent
            
            # Overlapping case:
            if dij<(r+R):
                # Fully overlapping case:
                if dij==0:
                    aij = Ai
                # Partially overlapping case: 
                else:
                    k1 = (pow(dij, 2) + pow(r, 2) - pow(R, 2))/(2*dij*r)
                    k2 = (pow(dij, 2) + pow(R, 2) - pow(r, 2))/(2*dij*r)
                    k3 = math.sqrt((-dij+r+R)*(dij+r-R)*(dij-r+R)*(dij+r+R))
                    # Compute the area of overlapping between the i-th and the j-th agents:
                    aij = pow(r, 2)*math.acos(k1) + pow(R, 2)*math.acos(k2) -k3/2
            # Not overlapping case:
            else:
                aij = 0

            return aij

        # Latest 'n_stored_pos' of agent 'fi':
        last_stored_fi_pos = self.broadcast_obs[fi.ID][3] if aoi==None else self.memory[aoi]['orig_obs'][fi.ID][3]
        # Area covered by agent 'fi' during the latest 'n_stored_pos' positions:
        last_n_pos_Ai = 0.
        # Current observation associated with the agent 'fi' on the ENode side:
        obs_fi = self.broadcast_obs[fi.ID] if aoi==None else self.memory[aoi]['orig_obs'][fi.ID]
        # Latest 'n_stored_pos' positions of the agent 'fi' stored on the ENode side:
        last_pos_fi = obs_fi[3] #[fi.position]
        # Check the coverage area for every latest stored position of agent 'fi':
        for last_pos_i_idx, last_pos_i in enumerate(last_pos_fi):
            # Polygon related to the current pos stored in latest 'n_stored_pos' position of the agent 'fi':
            cur_pol_footprint_i = last_pos_i.buffer(fi.footprint_radius)
            # Polygon intersection between the operative polygon and the sensing range of the current agent 'fi':
            free_pol = cur_pol_footprint_i.intersection(pol) # -> it will store the polygon of the 'cur_pol_footprint_i' not in overlap
            # Compute some possible overlapping only if the current sensing range is (either partially or fully) inside the operative polygon:
            if free_pol.area>0:
                '''
                Check for some possible overlappings between the current last stored position of the agent 'fi'
                and the corresponding (i.e., at the same time instant) latest stored position of the agent 'fj':
                '''
                for fj in self.flights: # -> the first flight is skipped since info related to the first one have been stored yet
                    # Do not consider comparison with itself at the same time instant:
                    if fi!=fj:
                        # Current observation associated with the agent 'fj' on the ENode side:
                        obs_fj = self.broadcast_obs[fj.ID] if aoi==None else self.memory[aoi]['orig_obs'][fj.ID]
                        # Latest 'n_stored_pos' positions of the agent 'fj' stored on the ENode side:
                        last_pos_fj = obs_fj[3]
                        # Position of the agent 'fj' 'temporally' associated with the positon of the agent 'fi' (both stored on the ENode side):
                        last_pos_j = last_pos_fj[last_pos_i_idx] #fj.position
                        # Polygon related to the current pos stored in latest 'n_stored_pos' position of the agent 'fj':
                        cur_pol_footprint_j = last_pos_j.buffer(fj.footprint_radius)
                        '''
                        'shapely' module could produce sometimes invalid geometry even from two valid shapes (for some approximation reasons).
                        To avoid to throw an error we skip this particular case (that could anyway affect the coverage area computation):
                        '''
                        try:
                            # Compute the polygon NOT in overlap between agents (if any):
                            free_pol = free_pol.difference(cur_pol_footprint_j)
                            # Compute the area of 'free_pol':
                            aij = free_pol.area
                        except:
                            print('Invalid shapely geometry for possible approximation error: the area coverage info is not computed for this case!')
                        # Stop computing all the remaining (if any) overlapping areas, since the current one is already fully overlapped:
                        if aij==0:
                            break
                    # Do not count the overlap with itself (fi=fj) here (it will be computed later):
                    else:
                        aij = 0

                    """
                    ________________________________________________________________________________________________
                    CONSIDERING SELF-OVERLAPS COULD LEAD THE AGENTS TO KEEP GOING TO MOVE CONTINUOUSLY:
                    YOU COULD TRY TO ENABLE/DISABLE THIS CONSIDERATION ACCORDING TO THE ENVIRONMENT FEEDBACK.
                    ________________________________________________________________________________________________
                    """
                    # Check some possible overlappings of the agent 'fi' with "itself" during the latest 'n_stored_pos' positions:
                    if last_pos_i_idx>0:
                        previous_position_i = last_pos_fi[last_pos_i_idx]
                        # Polygon related to the previous pos stored in latest 'n_stored_pos' position of the current agent 'fi':
                        previous_pos_footprint_i = previous_position_i.buffer(fi.footprint_radius)
                        # Compute the polygon of NOT in overlap of the current agent 'fi' with itself during its latest 'n_stored_pos' positions:
                        '''
                        'shapely' module could produce sometimes invalid geometry even from two valid shapes (for some approximation reasons).
                        To avoid to throw an error we skip this particular case (that could anyway affect the coverage area computation):
                        '''
                        try:
                            free_pol = free_pol.difference(cur_pol_footprint_i)
                            # Compute the area of 'free_pol':
                            ai_i_minus1 = free_pol.area
                        except:
                            print('Invalid shapely geometry for possible approximation error: the area coverage info is not computed for this case!')
                        # Stop computing all the remaining (if any) overlapping areas, since the current one is already fully overlapped:
                        if aij==0:
                            break

            free_ai = free_pol.area
            # Update the actual coverage area (by summation) during the latest N position of the current agent 'fi':
            last_n_pos_Ai += free_ai #Ai

        # Compute the mean of the coverage area of the latest N position of the current agent 'fi':
        last_n_pos_Ai /= self.n_stored_pos

        return last_n_pos_Ai

    @staticmethod
    def airspace_computation_and_loading(FIR_file: Optional[Union[pathlib.Path, None]] = None,
                                         FIR_ID: Optional[Union[str, None]] = None,
                                         min_area: Optional[Union[float, None]] = None,
                                         max_area: Optional[Union[float, None]] = None,
                                         polygon_verteces: Optional[List] = []) -> Polygon:
        """
        Load and compute (in the simulated env) the airspace provided as input.

        :param FIR_file: FIR file describing the polygon to be used
        :param FIR_ID: FIR ID of the polygon related to the FIR to be used
        :param min_area: minimum area value for the polygon to be generated
        :param max_area: maximum area value for the polygon to be generated
        :param polygon_verteces: verteces of the polygon to be used
        """  

        # If both FIR_file and FIR_ID are specified, then an external_FIR is used:
        external_FIR = FIR_file!=None and FIR_ID

        def transorm_external_airspace():
            airspace = Airspace.external_airspace(FIR_file=FIR_file, FIR_ID=FIR_ID)

            return airspace  

        def random_airspace():
            airspace = Airspace.random(min_area=min_area, max_area=max_area)

            return airspace

        def manual_airspace():
            airspace = Airspace.manual(polygon_verteces=polygon_verteces)

            return airspace

        # Process the provided external FIR for the airspace generation:
        if external_FIR:
            warnings.warn("You are using an external FIR file: if you are also using a discrete action space, then remember to set properly the available actions in 'global_vars' according to the operative space used in the FIR file!")
            time.sleep(3)
            airspace = transorm_external_airspace()
        else:
            # Random airspace generation:
            if polygon_verteces==[]:
                airspace = random_airspace()
            # Manual airspace generation:
            else:
                airspace = manual_airspace()

        return airspace

    @staticmethod
    def xy2latlon(cartesian_p: Point, latlon_ref: Tuple[float, float]) -> Tuple[float, float]:
        """
        Compute (X,Y) coordinates transformation from (lat, lot).

        :param cartesian_p: (X,Y) point
        :param latlon_ref: (lat, lon) reference point to perform the coordinate transformation

        :return latlon_des: tuple containing the (X,Y) coordinates of the desired point 
        """
        x = cartesian_p.x
        y = cartesian_p.y
        theta = math.atan2(x, y)
        compass = (theta + u.circle) % u.circle
        dst = x/math.sin(compass) # OR y/math.cos(theta)

        latlon_des = inverse_haversine(latlon_ref, dst, compass, unit=Unit.METERS)
        
        return latlon_des

    def results_for_external_obs(self, ref_p: Tuple[float, float], flight: "Flight", chosen_heading: float, chosen_dist: float, first_current_log: bool):
        """
        Print (show) the action selected based on external observation and store the action selected in the related actions' buffer.

        :param ref_p: reference point expressed in (lat,lon)
        :param flight: UAV considered
        :param chosen_heading: heading angle currenlty selected
        :param chosen_dist: distance currently selected
        :param first_current_log: boolean indicating if it is the first log or not

        :return: None
        """
        flight_latlon = ENode.xy2latlon(cartesian_p=flight.target, latlon_ref=ref_p)
        chosen_heading_deg = math.degrees(chosen_heading)
        if first_current_log:
            print('\n\nACTIONS SELECTED AND NEXT POSITIONS ASSOCIATED WITH THEM:\n')
        print('--------------------------------------------------------------------------------------------------------')
        print('UAV ID: {}'.format(flight.ID))
        print('Distance: {} [m]'.format(chosen_dist))
        print('Heading: {} [m]'.format(chosen_heading_deg))
        print('Next (available) position associated with the selected actions: {}'.format(flight_latlon))

        # Add the next action for the current agent:
        cur_flight_prediction = {'ID': flight.ID,
                                 'distance': chosen_dist,
                                 'angle': chosen_heading_deg,
                                 'next lat': flight_latlon[0],
                                 'next lon': flight_latlon[1]}
        self.next_actions.append(cur_flight_prediction)

    @property
    def generate_predictions_file(self):
        """
        Generate a predictions' file to be used only when external communication is enabled.

        :param: None

        :return: None
        """
        file_name = PATH_FOR_EXT_COMMUNICATION + PRED_FILENAME_FOR_EXT_COMMUNICATION
        headers = ['ID', 'distance', 'angle', 'next lat', 'next lon']
        rows = self.next_actions

        with open(file_name, 'w', encoding='UTF8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

        # Reset the next actions for all the agents:
        self.next_actions = []

    def sensing(self, STAR: Optional[bool] = True, sense: Optional[str] = 'ready_to_listen') -> None:
        """
        Update the ENode sensing mode.

        :param STAR: enalble or disable a STAR communication
        :param sense: set the sensing status ('ready_to_listen' or 'ready_to_broadcast'): it is used only if STAR=False
        
        :return None
        """
        assert STAR or (not STAR and (sense=='ready_to_listen' or sense=='ready_to_broadcast')), 'If STAR communication is not enabled, then the sensing action can be either only "ready_to_listen" or only "ready_to_broadcast"!'
        self.is_charging = True

        def listening() -> None:    
            self.Rx = True
            self.Sx = False

        def broadcast() -> None:
            self.Rx = False
            self.Sx = True

        if STAR:
            self.Rx = True
            self.Sx = True
        else:
            if sense=='ready_to_listen':
                listening()
            else:
                broadcast()

    def update_current_t(self, t: float) -> None:
        """
        Update the ENode elapsed time with the absolute elapsed time.

        :param t: absolute time value

        :return: None
        """
        self.cur_t = t

    def update_POI_status(self, status: bool, t: float, flight_ID: int, elem: Optional[Union[int, Point]] = True) -> None:
        """
        Update the status and the related time of the considered POI (either by its coords or by its ID).

        :param status: the new status to assign to the considered POI
        :param t: time at which the assigned status is verified
        :param flight_ID: flight ID of the agent located at the current 'elem'
        :param elem: either the point or the ID representing the considered POI

        :return: None
        """
        e_type = type(elem)
        assert e_type==int or e_type==Point, 'You can update a POI status either using its ID or point!'
        is_valid_point = True

        def by_ID(ID: int) -> None:
            self.POIs[ID][1] = status

        def by_point(p: Tuple[float, float]) -> bool:
            try:
                self.POIs2[p][1] = status
                return True
            except:
                #print('\nTrying to update the status of a point which is not a landing point: no update!\n')
                # automatically assign the larger number to the new landing point (this will not follow the number
                # ordering of of the POIs): 
                #self.POIs2[p][1] = status
                return False

        def get_point_coords(p: Point) -> Tuple[float]:
            return (p.x, p.y)

        # Current updates (to perform now):
        if self.cur_t==t:
            # ID case:
            if e_type==int:
                by_ID(elem)
                p_to_update = self.POIs[elem][0]
                p_to_update = get_point_coords(p_to_update)  
                self.POIs2[p_to_update][1] = status
                self.POIs2[p_to_update][2] = flight_ID
            # Point case:
            else:
                elem = get_point_coords(elem)
                if by_point(elem):
                    id_to_update = self.POIs2[elem][0]
                    self.POIs[id_to_update][1] = status
                    self.POIs[id_to_update][2] = flight_ID

            self.lps = [p[0] for p in self.POIs.values()]
            lps = self.lps
        # Future updates (to perform later on) stored in the info queue:
        else:
            if self.scnr_cfg.action_delay:
                future_POI_to_update = (status, elem, t, flight_ID)
                '''
                Update the info queue only if the current info is related to a FREE position, and it is not yet in the info queue
                (the position can be updated in the 'future' only if their status will be FREE, indeed a BOOKED position and a BUSY
                position are both considered to be BUSY):
                '''
                if status==FREE and future_POI_to_update not in self.queue_info:
                    self.queue_info.append(future_POI_to_update)

    def update_broadcast_obs(self, local_flight_obs: Dict, spawn_obs_time: float, n_observations_to_be_spawn: int) -> None:
        """
        Update the observation computed based on all the infos received by each UAV.
        
        :param local_flight_obs: UAV local observation picked from the OBSERVATION BUFFER
        :param spawn_obs_time: spawning/scheduling time of ''local_flight_obs'
        :param n_observations_to_be_spawn: number of spawning observations

        :return: None 
        """
        '''
        ________________________________________________________________________________________________________________________
        Update the global observation only IF (the upper integer part of) the 'spawning/scheduling-time'
        of the local observation (picked from the observations' buffer) is equal to the current elapsed time.
        We consider the upper integer part of the scheduling time obviously because we can schedule it only after receving it:
        ________________________________________________________________________________________________________________________
        '''
        if math.ceil(spawn_obs_time)==self.cur_t:
            # For each communication coming from a different flight, increase the number of observations received (at the current time instant):
            self.n_obs_received += 1
            
            #  ----------------- Update the global observation based on local observation of the current flight: -----------------
            local_pos_features = local_flight_obs[POS_FEATURES_KEY]
            local_agent_features = local_flight_obs[AGENT_FEATURES_KEY]
            local_shot_features = local_flight_obs[LAST_SHOT_KEY]
            local_aoi = local_flight_obs[AOI_KEY]
            local_pos_ID = local_flight_obs[ID_KEY]
            global_shot_features = self.broadcast_obs[LAST_SHOT_KEY]
            # In 'local_pos_features' are stored info (including the time) related to both starting and target points (here referred as 'p'):
            for p in local_pos_features:
                local_pos_status = p[0]
                local_pos_coords = p[1]
                local_pos_t = p[2]
                # Update POIs:
                self.update_POI_status(status=local_pos_status, t=local_pos_t, flight_ID=local_pos_ID, elem=local_pos_coords)
            
            '''
            Reset (for the next time instant) the number of the received observations in the current time instant only if they are equal
            to the number of observations to be spawn on the current time instant: 
            '''
            if self.n_obs_received==n_observations_to_be_spawn:
                self.n_obs_received = 0 # -> reset the number of observation received (for the next observations to be received)
                self.current_spotting_agents = [] # -> reset the list of the agents detecting the source at the current time instant

            # Store the time of the most recent shot and its position and also the list of all the agent spotting the source at the current time instant:
            last_shot_features_modified = CommunicationModule.store_specific_shot_features(enode=self, flights=self.flights, local_ID=local_pos_ID, shot_info=local_shot_features)
            if local_aoi in self.memory:
                '''
                if 'local_aoi' is not in memory, then either the info is the most recent one or it is too old.
                In the former case it will be added later on in 'memory_update_buffer' (called in 'env.py'),
                whilst in the latter case it has been discharged due to the memory buffer limit.
                '''
                old_last_shot_features_modified = CommunicationModule.store_specific_shot_features(enode=self, flights=self.flights, local_ID=local_pos_ID, shot_info=local_shot_features, aoi=local_aoi) # -> 'self.flights' is used only to get the ID of each flight based on their occurence order

            # Last 'n_stored_pos' of the current agent:
            last_stored_pos = self.broadcast_obs[local_pos_ID][3]
            # Latest arrived local obs update (regardless of the fresheness of the local obs):
            self.broadcast_obs[LAST_SHOT_KEY] = copy.deepcopy(last_shot_features_modified)
            self.broadcast_obs[local_pos_ID] = copy.deepcopy(local_pos_features)
            self.broadcast_obs[local_pos_ID].append(copy.deepcopy(local_agent_features))
            # Enode Memory update associated with the latest local obs:
            if local_aoi in self.memory:
                self.memory[local_aoi]['orig_obs'][LAST_SHOT_KEY] = copy.deepcopy(old_last_shot_features_modified)
                self.memory[local_aoi]['orig_obs'][local_pos_ID] = copy.deepcopy(local_pos_features)
                self.memory[local_aoi]['orig_obs'][local_pos_ID].append(copy.deepcopy(local_agent_features))            
            
            # ------------------------ Reassignment associated with the latest obs (regardless of its freshness) ------------------------
            
            '''
            Re-assign the features (status, pos, time) related to the current agent position
            (compute local agent start position and access the last 'n_stored_pos' positions of the current agent):
            '''
            local_start = local_pos_features[0]
            local_start_pos = local_start[1]
            # Remove the last agent (start) position out of 'n_stored_pos': 
            last_stored_pos.pop(0)
            # Append the new agent (start) position out of 'n_stored_pos':
            last_stored_pos.append(local_start_pos)
            # Update the osbervation containing the last 'n_stored_pos' (start) positions of the current agent:
            self.broadcast_obs[local_pos_ID].append(copy.deepcopy(last_stored_pos))
            # Update the observation containing the AoI of the local observation of the current agent:
            self.broadcast_obs[local_pos_ID].append(copy.deepcopy(local_aoi))
            '''
            On ENode-side we only look at the time (but still indicated with AoI) at which we generate (i.e., receive)
            the (cumulative/global) info and not when it has been generated.
            '''
            self.broadcast_obs[AOI_KEY] = self.cur_t # -> updated to the same value for all the times that it receives an observation from a UAV at the current time instant

            # ------------------------ Enode Memory update associated with the latest local obs ------------------------
            if local_aoi in self.memory:
                '''
                The features associated with the positions are obviously the same either if they belong to the most recent obs or not 
                (update the osbervation containing the last 'n_stored_pos' (start) positions of the current agent):
                '''
                self.memory[local_aoi]['orig_obs'][local_pos_ID].append(copy.deepcopy(last_stored_pos))
                '''
                Inside memory we need to specify that now the obs associated with the processed flight is related to the correct AoI
                (i.e., the global and the local AoI of the current flights are the same as they have been properly ordered):
                '''
                self.memory[local_aoi]['orig_obs'][local_pos_ID].append(copy.deepcopy(local_aoi))
                '''
                The following update is actually useless, while in self.broadcast_obs it is needed as in the case the observation
                has being processing for the first time and thus the AOI_KEY is not already defined:
                '''
                self.memory[local_aoi]['orig_obs'][AOI_KEY] = local_aoi

            # -----------------------------------------------------------------------------------------------------------

    @property
    def update_info_buffer(self):
        """
        Update the ENode buffer related to the info that need to be scheduled at the current time.
        
        :param: None 
        
        :return: None
        """
        # If actions are istantaneous, then there is no info that must be stored and scheduled in a future time instant:
        if not self.scnr_cfg.action_delay:
            self.queue_info = []
        else:
            p_scheduled = []
            # Update based on the currently info (if any) scheduled that is stored in the info queue:
            for p in self.queue_info:
                pos_t = p[2]
                pos_ID = p[3]
                if self.cur_t>=pos_t:
                    pos_status = p[0]
                    pos_coords = p[1]
                    self.update_POI_status(status=pos_status, t=pos_t, flight_ID=pos_ID, elem=pos_coords)
                    stored_info = (pos_status, pos_coords, pos_t)
                    
                    # Remove the eldest observation:
                    old_t = math.inf
                    info_to_remove = None 
                    pos_features = copy.deepcopy(self.broadcast_obs[pos_ID][:2])        
                    for p_feature in pos_features:
                        t_feature = p_feature[2] 
                        if t_feature<old_t:
                            old_t = t_feature
                            info_to_remove = p_feature
                    pos_features.remove(info_to_remove)
                    # Add the current stored information to the list of the scheduled info:
                    pos_features.append(stored_info)
                    self.broadcast_obs[pos_ID][:2] = pos_features
                    p_scheduled.append(p)
            
            # Remove the processed info from the queue:
            for p in p_scheduled:
                self.queue_info.remove(p)

@dataclass
class Source:
    """
    Source class
    """
    flat_walk: List
    times: List
    spot_types: List # -> indicates the type of the hotstpots to cross
    position: Point
    start: Point
    target: Point
    local_time: float # -> indicates the time for which the audio source is has not moved 
    bcb_T: float # -> time T needed to travel a specific path through a bang-coast-bang (bcb) approach
    bcb_satisfied: bool # -> if the bang-coast-bang constraints satisfied or not
    cur_moving_time: float # -> current time in which the audio source is actually moving since the last time it stopped
    local_groundspeed: float 
    is_shooting: bool # -> if the audio source is emitting a signal or not
    path_traveled: bool # -> if the audio source has travelled all its path or not
    global_max_speed: float # -> scnr_cfg.src_global_max_speed
    local_max_speed: float # -> scnr_cfg.src_local_max_speed
    global_max_acc: float # -> scnr_cfg.src_global_max_acc
    local_max_acc: float  # -> scnr_cfg.src_local_max_acc
    db_level: float
    power: float
    static: bool
    constant: bool
    x_traj_history: List
    y_traj_history: List 

    groundspeed: float = field(init=False)
    track: float = field(init=False)

    def __post_init__(self) -> None:
        """
        Initialises the track and the groundspeed.
            
        :param: None

        :return: None
        """
        self.track = self.bearing
        self.groundspeed = self.local_groundspeed

    @property
    def shooting_decision(self):
        """
        Generate a random 'shoot' coming from the source to be spotted.

        :param: None

        :return: None 
        """
        if not self.constant:
            self.is_shooting = np.random.randint(low=0, high=2)
        else:
            self.is_shooting = 1.

    def bang_coast_bang(self, dt: float) -> Tuple:
        """
        If possible, apply the bang-coast-bang (trapezoidal speed profile) to the selected 'item',
        otherwise a cruise speed will be assigned to the considered 'item'.
        
        :param dt: resolution time step

        :return: dx,dy position components based on the current time step OR a cruise speed (if estimate is desired) 
        """
        # Inner spot subcase:
        if self.spot_types[0]=='inner': 
            max_speed = self.local_max_speed
            max_acc = self.local_max_acc#[0]
            #self.src_tol = self.local_max_speed * 1.05 * self.dt
        # Hotspot subcase:
        else:
            max_speed = self.global_max_speed
            max_acc = self.global_max_acc#[0]
            #self.src_tol = self.global_max_speed * 1.05 * self.dt
        
        Ts = max_speed/max_acc
        L = self.travel_phase_distance

        # B-C-B constraints to be satisfied:
        constraint1 = max_speed > L/self.bcb_T
        constraint2 = max_speed <= 2*L/self.bcb_T

        # Bang-coast-bang constraints satisfied:
        if constraint1 and constraint2:
            self.bcb_satisfied = True
            Ts1 = self.bcb_T-Ts

            if self.cur_moving_time <= Ts:
                current_speed = max_acc*self.cur_moving_time
            elif self.cur_moving_time <= Ts1:
                current_speed = max_speed
            else:
                current_speed = max_acc*(self.bcb_T-self.cur_moving_time)
            
            self.groundspeed = current_speed
            
        # Bang-coast-bang constraints not satisfied:
        else:
            self.bcb_satisfied = False
            self.groundspeed = max_speed/2
            # Assign a new travel time T based on the value of max_speed:
            self.bcb_T = (L*max_acc + pow(max_speed, 2))/(max_acc*max_speed) # -> b-c-b time expressed in [s]

        dx, dy = self.components

        # Increase the time in which the source is in motion:
        self.cur_moving_time += dt
        self.cur_moving_time = np.clip(self.cur_moving_time, a_min=0, a_max=self.bcb_T)
        
        return dx, dy

    @property
    def bearing(self) -> float:
        """
        Bearing from current position to target.
        :param: None
        :return: None
        """
        dx = self.target.x - self.position.x
        dy = self.target.y - self.position.y
        compass = math.atan2(dy, dx)
        return (compass + u.circle) % u.circle

    @property
    def components(self) -> Tuple:
        """
        X and Y Speed components.
        :parama: None
        :return dx, dy: speed components
        """
        dx = self.groundspeed * math.cos(self.track)
        dy = self.groundspeed * math.sin(self.track)
        return dx, dy

    @property
    def distance(self) -> float:
        """
        Current distance to the target.
        :param: None
        :return: distance to the target
        """
        return round(self.position.distance(self.target), 2)

    @property
    def drift(self) -> float:
        """
        Drift angle (difference between track and bearing) to the target.
        :param: None
        :return: None
        """
        drift = self.bearing - self.track

        if drift > math.pi:
            return -(u.circle - drift)
        elif drift < -math.pi:
            return (u.circle + drift)
        else:
            return drift

    @property
    def travel_phase_distance(self) -> float:
        # Planar (horizontal) distance to be traveled during the motion phase
        L = self.start.distance(self.target)
        return L

    @staticmethod
    def source_walk(pol: Polygon, n_hotspots: Optional[int] = 10, n_p_min: Optional[int] = 3,
                    n_p_max: Optional[int] = 7, local_d: Optional[float] = 50,
                    first_hotspot: Optional[Union[Point, None]] = None) -> Dict:
        """
        Define all the path/walk traveled by the audio source.

        :param pol: operative polygon
        :param n_hotspots: number of macro hotspot to be crossed by the audio source
        :param n_p_min: minimum number of micro hotspot (around each macro hotspot) to be travelled by the audio source
        :param n_p_max: maximum number of micro hotspot (around each macro hotspot) to be travelled by the audio source
        :param local_d: maximum distance that can be traveled by the audio source when it moves among inner-hotspots
        :param first_hotspot: is the already defined first hotspot (if any)

        :return walk: a dictionary containing the path/walk to be traveled by the audio source
        """
        hotspots = []
        minx, miny, maxx, maxy = pol.bounds

        xy_pol = pol.boundary.coords.xy
        # Outer rectangular bounding box for the considered polygon:
        #outer_rec = box(min(xy_pol[0]), min(xy_pol[1]), max(xy_pol[0]), max(xy_pol[1]))

        def random_point_in_circle(center: Point, radius: float) -> Point:
            """
            Compute a random point inside a circle of a given radius.
            :param center: (x,y) point of the the center of the circle
            :param radius: radius of the circle

            :return Point: (x,y) point inside the give circle
            """
            theta = random.uniform(0, 2*math.pi)
            surf = pow(radius, 2)*math.pi
            new_radius = math.sqrt(random.uniform(0, surf/math.pi))
            x = center.x+new_radius*math.cos(theta)
            y = center.y+new_radius*math.sin(theta)
            return Point(x, y)
        
        count_hotspots = 1
        n_points = 500

        while True:
            if first_hotspot!=None and hotspots==[]:
                point = first_hotspot
            else:
                point = Point(random.uniform(minx, maxx),
                              random.uniform(miny, maxy))

            if first_hotspot!=None and hotspots==[]:
                assert pol.contains(point) and pol.exterior.distance(point) >= local_d, 'The manually selected point for the audio source is NOR inside the desired polygon and its distance from the closest edge of the polygon is NOT greater than "local_d"!'

            '''
            Store the generated point IF the random point is inside the desired polygon, and its distance from
            the closest edge of the polygon is greater than 'local_d': 
            '''
            if pol.contains(point) and pol.exterior.distance(point) >= local_d:
                hotspots.append(point)
                count_hotspots +=1 
            
            if len(hotspots)==n_points:
                break

        # The hotspots of the source are the centroids of the clusters generated by applying KMeans on the 'n_points':
        kmeans = KMeans(n_hotspots).fit(np.array([[hs.x, hs.y] for hs in hotspots])) 
        hotspots = kmeans.cluster_centers_
        walk = {i:[] for i in range(1, n_hotspots+1)}
        for hs_idx in range(n_hotspots):
            # Number of random points inside a circle of radius 'local_d' and centered in 'point':
            n_p = random.randint(n_p_min, n_p_max)
            inner_p = []
            if first_hotspot!=None:
                if hs_idx>0:
                    current_hotspot = Point(hotspots[hs_idx][0], hotspots[hs_idx][1])
                else:
                    current_hotspot = first_hotspot
            else:
                current_hotspot = Point(hotspots[hs_idx][0], hotspots[hs_idx][1])

            for n_i in range(n_p):
                p = random_point_in_circle(current_hotspot, local_d)
                inner_p.append(p)   
            
            walk[hs_idx+1] = {'hotspot': current_hotspot, 'inners': inner_p} #[Point(hotspots[hs_idx][0], hotspots[hs_idx][1]), inner_p]

        return walk

    @classmethod
    def flat_source_walk(cls, scnr_cfg: "ScenarioConfig", walk: Dict, dt: float) -> None:
        """
        Create the audio source with all its parameters.
        It also turns the dictionary containing path/walk of the audio source into a flatten array.
        
        :param scnr_cfg: parameters defined in 'scenario_parameters.ini'
        :param dt: resolution time step
        :param walk: dictionary containing the path/walk associated to the audio source

        :return: audio source   
        """ 
        flat_walk = []
        spot_types = []
        for hotspot_idx, hv in walk.items():
            flat_walk.append(walk[hotspot_idx]['hotspot'])
            spot_types.append('hotspot')
            for inner_p in walk[hotspot_idx]['inners']:
                flat_walk.append(inner_p)
                spot_types.append('inner')

        def time_on_spot(min_t: Optional[float] = 60, max_t: Optional[float] = 360) -> List:
            """
            Generate random times (in a given time range) where the source is supposed to be on some specific spots.

            :param min_t: minimum time per spot
            :param max_t: maximum time per spot

            :return: a list of times related to the source stop
            """
            return [random.randrange(min_t, max_t, dt) for w in flat_walk]

        def sound_power(db_level, factor: Optional[float]=10) -> float:
            """
            Return the sound power of a source at 'zero' distance (i.e., r=0).
            
            :param db_level: source sound level in dB
            :param factor: 10 (SIL) or 20 (SPL)
            
            : return power: the sound power
            """
            # We use r=1 as we are considering the propagation of a point sound source (and hence it will be infinite for r=0):
            r = 1

            def sound_intensity(factor: float) -> float:
                return I0*pow(10, db_level/factor)
            
            sound_int = sound_intensity(factor)
            sound_p = sound_int*(4*math.pi*pow(r, 2))

            return sound_p

        times = time_on_spot()
        position = flat_walk[0]
        start = copy.deepcopy(position)
        target = flat_walk[1]
        local_time = times[0]
        bcb_T = 10000
        bcb_satisfied = True
        cur_moving_time = 0.
        groundspeed = 0.
        is_shooting = np.random.randint(low=0, high=2)
        path_traveled = False
        global_max_speed = scnr_cfg.src_global_max_speed
        local_max_speed = scnr_cfg.src_local_max_speed 
        global_max_acc = scnr_cfg.src_global_max_acc
        local_max_acc = scnr_cfg.src_local_max_acc
        db_level = scnr_cfg.db_level
        power = sound_power(db_level=db_level)
        static = scnr_cfg.source_static
        constant = scnr_cfg.source_constant
        x_traj_history = []
        y_traj_history = []

        return cls(flat_walk, times, spot_types, position, start, target, local_time, bcb_T,
                   bcb_satisfied, cur_moving_time, groundspeed, is_shooting, path_traveled,
                   global_max_speed, local_max_speed, global_max_acc, local_max_acc, db_level,
                   power, static, constant, x_traj_history, y_traj_history)

@dataclass
class Flight:
    """
    Flight class
    """
    position: Point
    start: Point
    target: Point
    ID: Union[int, str]
    optimal_airspeed: float
    cur_moving_time: float
    min_db_level: float
    footprint_radius: float
    fla: int # -> Flight Level Assigned
    takingoff_time: float
    landing_time: float
    hflight_time: float
    sensing_time: float
    bcb_T: float # -> time T needed to travel a specific path through a bang-coast-bang (bcb) approach
    bcb_satisfied: bool # -> bang-coast-bang constraints satisfied or not
    battery: int
    is_charging: float
    p_eng: float # -> mean engine power (on an average path at a cruise speed) [KWh]
    v_ref: float # -> Speed reference related to the Pe value provided 'p_eng' value [m/s]
    p_bat: float # -> power battery capacity [KWh]
    p_bat_charging: float # -> charging power [KW]
    b_efficiency: float # -> battery efficiency [%]
    b_res: float # -> battery resolution
    battery_levels: List
    takeoff: bool
    landing: bool 
    local_obs: Dict
    received_obs: Dict
    achievable_pos: List[Point]
    max_speed: float
    max_acc: float
    Rx: bool
    Sx: bool
    sensing_range_enabled: bool
    clock: float
    memory: Dict
    memory_size: int
    actions_memory: List
    actions_size: int
    flight_altitude: float # -> FL assigned to the current UAV
    altitude: float # Current FLA
    p1: List
    p2: List
    p3: List
    p4: List
    roll: float
    pitch: float
    yaw: float
    roll_speed: float
    pitch_speed: float
    yaw_speed: float
    cur_t_takeoff: float
    cur_t_landing: float
    cur_t_hflight: float
    cur_t_sensing: float
    waypoints: List
    x_c: List
    y_c: List
    z_c: List
    x_speed: float
    y_speed: float
    z_speed: float
    traj_type: int
    x_traj_history: list # -> store the X coordinates of the trajectory that is currently being traveled
    y_traj_history: list
    z_traj_history: list
    n_motions_before_spotting: int 

    airspeed: float = field(init=False) # -> refers only to the horizontal speed (thus, it will be 0 during takeoff and landing phases)
    track: float = field(init=False)
    T_e_to_fc: float = field(init=False)

    def __post_init__(self) -> None:
        """
        Initialises the track and the airspeed.
        :param: None
        :return: None
        """
        self.track = self.bearing
        self.airspeed = self.optimal_airspeed
        self.T_e_to_fc = (self.p_bat/self.p_bat_charging)*3600

    def sensing(self, STAR: Optional[bool] = True, sense: Optional[str] = 'ready_to_listen') -> None:
        """
        Update the agent sensing mode.

        :param STAR: enalble or disable a STAR communication
        :param sense: set the sensing status ('ready_to_listen' or 'ready_to_broadcast'): it is used only if STAR=False
        
        :return: None
        """
        assert STAR or (not STAR and (sense=='ready_to_listen' or sense=='ready_to_broadcast')), 'If STAR communication is not enabled, then the sensing action can be either only "ready_to_listen" or only "ready_to_broadcast"!'
        self.is_charging = True

        def listening() -> None:    
            self.Rx = True
            self.Sx = False

        def broadcast() -> None:
            self.Rx = False
            self.Sx = True

        if STAR:
            self.Rx = True
            self.Sx = True
        else:
            if sense=='ready_to_listen':
                listening()
            else:
                broadcast() 

    @property
    def flight_mode(self) -> None:
        """
        Set all the features related to the agent flight mode.
        :param: None
        :return: None
        """
        self.Rx = False
        self.Sx = False
        self.is_charging = False
        self.sensing_range_enabled = False

    def sensing_mode(self, communication_module: 'CommunicationModule') -> None:
        """
        Set all the features related to the agent sensing mode.
        :param: None
        :return: None
        """
        if communication_module.STAR:
            communication_module.status(self)
        else:
            raise NotImplementedError('The communication behaviour (i.e., either Rx or Sx) during the non flying mode in not implemented!')
            '''
            sense = '' # 'ready_to_listen' OR 'ready_to_broadcast' 
            CommunicationModule.status(self, sense=sense)
            '''

        self.is_charging = True
        self.sensing_range_enabled = True

    def bang_coast_bang(self, dt: Optional[float] = 5., estimate: Optional[bool] = False, ref_point: Optional[Union[Point, None]] = None) -> Union[Tuple, float]:
        """
        If possible, apply the bang-coast-bang (trapezoidal speed profile) to the selected 'item',
        otherwise a cruise speed will be assigned to the considered 'item'.
        
        :param dt: time resolution
        :param estimate: if estimatation is desired, then 'item' holds its position (useful for instantaneous actions) and a cruise speed is returned
        :param ref_point: reference point w.r.t. perform the estimate (by using the 'item' position)

        :return: dx,dy position components based on the current time step OR a cruise speed (if estimate is desired) 
        """
        assert estimate==False or (estimate==True and ref_point!=None), 'If you want to perform an estimation of the speed needed to achieve a reference point, then you need to specify also that reference point!'
        
        max_speed = self.max_speed
        max_acc = self.max_acc

        if estimate:
            L = self.position.distance(ref_point)
            avg_speed_along_L = L/self.bcb_T
            
            return avg_speed_along_L

        Ts = max_speed/max_acc
        L = self.travel_phase_distance
        self.bcb_T = (L*max_acc + pow(max_speed, 2))/(max_acc*max_speed) # -> b-c-b time expressed in [s]

        # B-C-B constraints to be satisfied:
        constraint1 = max_speed > L/self.bcb_T
        constraint2 = max_speed <= 2*L/self.bcb_T

        # Bang-coast-bang constraints satisfied:
        if constraint1 and constraint2:
            self.bcb_satisfied = True
            Ts1 = self.bcb_T-Ts

            if self.cur_moving_time <= Ts:
                current_speed = max_acc*self.cur_moving_time
            elif self.cur_moving_time <= Ts1:
                current_speed = max_speed
            else:
                current_speed = max_acc*(self.bcb_T-self.cur_moving_time)
             
            self.airspeed = current_speed
            
        # Bang-coast-bang constraints not satisfied:
        else:
            self.bcb_satisfied = False
            self.airspeed = max_speed/2 # -> 'emergency' speed
            # Assign a new travel time T based on the value of 'max_speed':
            self.bcb_T = (L*max_acc + pow(max_speed, 2))/(max_acc*max_speed) # -> b-c-b time expressed in [s]

        dx, dy = self.components
        
        self.cur_moving_time += dt
        self.cur_moving_time = np.clip(self.cur_moving_time, a_min=0, a_max=self.bcb_T)
    
        return dx, dy

    def update_achievable_positions(self, all_positions: List[Point], energy_constraint: Optional[bool] = False) -> None:
        """
        Update the set of the achievable positions based on the current battery level of the considered UAV.

        :param all_positions: all the available positions in the current operative polygon
        :param energy_constraint: a boolean indicating if the battery constraints must be taken into account or not

        :return: None 
        """
        self.achievable_pos = copy.deepcopy(all_positions)
        for key, obs_set in self.received_obs.items():
            '''
            Consider only the keys related to the IDs of the UAVs.
            Remark: if you add in the broadcast_obs other features different from an 'ID' and from 'LAST_SHOT_TIME',
            then you must add other conditions on these new features (i.e., key!=new_features).
            To be made it more generic in a future work.
            '''
            if (key!='LAST_SHOOT_TIME') and (key!='AOI'):
                pos_obs = obs_set[:2]
                for obs in pos_obs:
                    point = obs[1]
                    status = obs[0]
                    tuple_point = (point.x, point.y) 

                    # Estimate the speed to achieved 'ref_point': 
                    supposed_speed = self.bang_coast_bang(estimate=True, ref_point=point)
                    # Estimate the percentage of needed battery to travel from the current UAV position to the considered POI:
                    supposed_needed_battery, _ = self.energy_consumption(actual_speed=supposed_speed, estimate=True, takeoff_land_time=self.takingoff_time)
                    '''
                    If the considered position is free AND current UAV battery will be greater or equal
                    than 0 once reached the considered POI, then the considered POI is said to be achievable:
                    '''
                    if status==FREE:
                        # If energy constraints is enabled, then check also the battery level:
                        if energy_constraint:
                            if (self.battery - supposed_needed_battery) >= 0:
                                self.achievable_pos[tuple_point] = [obs[0], status, obs[2]]
                        # Otherwise do not check the battery level:
                        else:
                            self.achievable_pos[tuple_point] = [obs[0], status, obs[2]]
                    else:
                        if tuple_point in self.achievable_pos: 
                            self.achievable_pos.pop(tuple_point)
                        else:
                            #print('Trying to update an achievable position which does not belong to the set of available landing points!')
                            pass

        self.achievable_pos = [Point(p[0], p[1]) for p in self.achievable_pos.keys()]

    def update_local_obs(self, source: Source, enode: ENode, t: float, external_obs: Optional[Union[pd.DataFrame, None]] = None, geodesics_pol_centroid: Optional[Union[Tuple[float, float], None]] = None) -> None:
        """
        Local observation must be sent right before the takeoff, and it is made up by a double position update.
        Indeed the operative area is small enough to assume that even a 'booked' target position can be considered
        as busy as the considered UAV takeoff to reach that target.   
        
        :param source: source to be spotted
        :param enode: ENode used to communicate
        :param t: absolute time elapsed when current local observation has being updated
        :param external_obs: external observation (if any)
        :param geodesics_pol_centroid: geodesic coordinates of the centroid
        """
        if not isinstance(external_obs, pd.DataFrame):

            '''
            The check on the receiving's agent status is performed inside the communication module (thus, here it is not needed).
            The local observation are locally updated every time it is possible:
            '''
            self.local_obs[ID_KEY] = self.ID

            '''
            In the particular case in which start=target, then the associated position will be updated
            as FREE and immediately after as BUSY (inside this scope 'without letting it know to any other').
            Thus also this particular case in which the UAV will not move is correctly managed. 
            '''

            # Start features:
            pos1 = self.start
            time_status1 = t+self.takingoff_time # -> takeoff and landing times are equal (but differs from an UAV to another one)
            '''
            Starting position was considered as FREE only if different from the target: in this way the
            current UAV is telling us that that starting position will be free in a moment as it needs to go to another position):
            '''
            pos_status1 = FREE
            
            # Target features:
            pos2 = self.target
            time_status2 = self.hflight_time + 2*self.takingoff_time
            pos_status2 = BUSY # -> even if it is just BOOKED, the target position is considered as BUSY

            if pos1==pos2:
                pos_status1 = BUSY 
            
            # Local observation updating:
            pos_features = [(pos_status1, pos1, time_status1), (pos_status2, pos2, time_status2)]  
            self.local_obs[POS_FEATURES_KEY] = pos_features
            self.local_obs[AGENT_FEATURES_KEY] = (self.track, self.observable_battery_level) # -> use the observable battery level

            # Update also the position of the source (if detected) only when the sensing range of the current UAV is enabled:
            if self.sensing_range_enabled:
                source_pos = source.position
                # Audio detection case:
                if source.is_shooting and self.footprint.contains(source_pos):
                    # It is communicated the position of the agent where the source is detected:
                    self.local_obs[LAST_SHOT_KEY] = (t, self.position, 1) # -> last feature is set to 1, because the current agent is detecting a shot
                # No detection case:
                else:
                    shot1 = self.received_obs[LAST_SHOT_KEY]
                    shot2 = self.local_obs[LAST_SHOT_KEY]
                    most_recent_shot = CommunicationModule.store_most_recent_shot(shot1, shot2)
                    most_recent_shot = most_recent_shot[0:2] + (0,) # -> add 0, because the current agent is not detecting any shot
                    self.local_obs[LAST_SHOT_KEY] = most_recent_shot
            
            # AoI of the current local observation:
            self.local_obs[AOI_KEY] = t

        else:
            def filterout_NA_value(pd_cell = pd.DataFrame) -> pd.DataFrame:
                """
                Check for 'NA' values (i.e., missing ones) in the external observation file
                related to a specific UAV ID.

                :param pd_cell: single cell DataFrame associated with a specific UAV ID

                :return  cell_value: the values contained inside 'pd_cell' (or math.nan if pd_cell=='NA')
                """
                if pd_cell.isnull().bool():
                    cell_value = math.nan
                else:
                    cell_value = float(pd_cell)

                return cell_value
            
            df = external_obs
            # Observation associated with the current UAV ID:
            ext_obs = df[df['ID']==self.ID]

            # Case in which the observation related to the current UAV is present in the external observation file:
            if not ext_obs.empty: 
                self.local_obs[ID_KEY] = self.ID
                source_t = filterout_NA_value(pd_cell=ext_obs['source time'])
                UAV_lat = filterout_NA_value(pd_cell=ext_obs['lat'])
                UAV_lon = filterout_NA_value(pd_cell=ext_obs['lon'])
                # For the time being, the altitude from external info is not needed since UAV are assumed to communicate only when landed:
                #UAV_FLA = filterout_NA_value(pd_cell=ext_obs['FLA'])
                latlon_p = (UAV_lat, UAV_lon)
                # In this particular case, 'xy_p' is a list made up by just one single tuple (i.e., the (X,Y) desired point):
                xy_p = Airspace.latlon2xy(latlon_ref=geodesics_pol_centroid, latlon_points=[latlon_p])
                UAV_pos = Point(xy_p[0][1], xy_p[0][1])
                source_spotted = filterout_NA_value(pd_cell=ext_obs['source spotted'])
                track = math.radians(filterout_NA_value(pd_cell=ext_obs['track']))
                # 'takeoff-landing time' is no more needed (but it could be usefule to allow external UAV to fly among landing points: but it will not actually be used except for the computation of start and target features!!!):
                takeoff_land_time = filterout_NA_value(pd_cell=ext_obs['takeoff-landing time'])
                battery = filterout_NA_value(pd_cell=ext_obs['battery'])
                UAV_aoi = filterout_NA_value(pd_cell=ext_obs['AoI'])
                '''
                Split all the receved AoI based on a step=0.5 (in such a way it will be avoided to store too many
                different observations coming from very close time instants: if the acquisition time of two different observations from
                the same UAV differs for a time<0.5 seconds, then these observations will be considered to be generated at the
                same time and only the last one will be considered):
                '''
                UAV_aoi_int, UAV_aoi_dec = math.modf(UAV_aoi)
                if UAV_aoi_dec<0.5:
                    UAV_aoi = UAV_aoi_int
                else:
                    UAV_aoi = UAV_aoi_int+0.5
                '''
                Start and target posistions are both set equal to the current position
                This is a corrent assumption when there is no action delay in the scenario, and this is exactly the case
                if an external observation as the action delay is already involved in the and motion observation themselves
                '''
                start_features = (BUSY, UAV_pos, t+takeoff_land_time)
                target_features = (BUSY, UAV_pos, self.hflight_time+2*takeoff_land_time)
                shot1 = self.received_obs[LAST_SHOT_KEY] # -> shot features based on the the global observation received by the ENode
                shot2 = (source_t, UAV_pos, source_spotted) # -> shot features based on the local external observation

                # Set the local observation based on the features extracted from the external observation file:
                self.local_obs[POS_FEATURES_KEY] = [start_features, target_features]
                self.local_obs[AGENT_FEATURES_KEY] = (track, battery)  
                # Store the most recent shot among those perceived at local and global level:
                most_recent_shot = CommunicationModule.store_most_recent_shot(shot1, shot2)
                most_recent_shot = most_recent_shot[0:2] + (source_spotted,)
                self.local_obs[LAST_SHOT_KEY] = most_recent_shot
                self.local_obs[AOI_KEY] = UAV_aoi
                #self.flight_altitude = UAV_FLA
            else:
                print('Missing external observation for UAV {}: no update performed for this UAV!'.format(self.ID)) 

    def update_received_obs(self, enode: ENode, energy_constraint: Optional[bool] = False) -> None:
        """
        Update the 'global' observation of the current UAV received by the ENode.
        The latest shot feature is constantly updated by the ENode through its observation broadcast to all the UAVs. 
        
        :param enode: ENode
        :return: None
        """
        self.received_obs = copy.deepcopy(enode.broadcast_obs)
        # Update the achievable positions based on the latest 'global' observation received by the ENode:
        self.update_achievable_positions(all_positions=enode.POIs2, energy_constraint=energy_constraint)

    @property
    def footprint(self) -> Polygon:
        """
        Sensing range (related to the audio source detection) of the current UAV.
        """
        return self.position.buffer(self.footprint_radius)

    @property
    def observable_battery_level(self) -> int:
        """
        Set the current battery level to the observable battery level based on the selected battery resolution.     
        :param: None
        :return obs_b_level: the observable battery level
        """
        for l in self.battery_levels:
            if self.battery>=l[0] and self.battery<=l[1]:
                if self.battery-self.b_res<l[0]:
                    obs_b_level = l[0]
                else:
                    obs_b_level = l[1]
                break

        return obs_b_level 

    def energy_consumption(self, actual_speed: float, dt: Optional[float] = 5., tl_e_perc: Optional[float] = -0.1, estimate: Optional[bool] = False, takeoff_land_time: Optional[Union[float, None]] = None) -> Tuple[float, float]:
        """
        Energy consumption for the considered time interval 'dt' or estimate of the battery needed
        to achieve a specific point specified in 'bang_coast_bang' method.

        :param actual_speed: speed of the current UAV (it is needed since it can be used also to perform an estimate)
        :param dt: time interval used to compute the battery consumption # --> dt must be the same as the one set in the 'env'!
        :param tl_e_perc: percentage (positive or negative) of battery consumption for taking off and landing (vertical) phases w.r.t. the horizontal motion 
        :param estimate: if estimation is desired, then UAV the battery energy needed for all the phases (takeoff, flight landing) is considered
        :param takeoff_land_time: time needed for takeoff and landing phases (it is the same for both and here it is considered only one of the 2 phases) [seconds]

        :return: residual battery level for the current flight and the time elapsed during the battery consumption
        """
        assert estimate==False or (estimate==True and takeoff_land_time!=None), 'If you are computing an estimation of the energy consumption, you need to specify also the time needed for the takeoff and landing phases!'
        actual_motion_consumption = self.p_bat*self.b_efficiency

        def flight_phase() -> float:
            # Compute the required engine power based on the current UAV speed: 
            cur_p_eng = (self.p_eng*actual_speed)/self.v_ref
            return cur_p_eng

        def takeoff_and_landing_consumption(motion_consumption: float, tl_e_perc: float) -> List[float]:
            actual_takeoff_and_landing_consumption = motion_consumption*(1.+tl_e_perc)
            # Use the provided engine power:
            cur_p_eng = self.p_eng
            return actual_takeoff_and_landing_consumption, cur_p_eng

        def perc_battery_per_hour(cur_p_eng: float, actual_motion_consumption: float) -> float:
            b_perc_per_hour = (100*cur_p_eng)/(actual_motion_consumption)
            return b_perc_per_hour

        def real_consumed_battery(b_perc_per_hour: float, timestep: float) -> float:
            battery_consumed = b_perc_per_hour*timestep/3600
            return battery_consumed

        time_needed = 0

        # Real-time battery consumption depending on the current flight phase (either takeoff or flight or landing):
        if not estimate:
            # Taking off or landing phases:
            if self.takeoff or self.landing:
                actual_takeoff_and_landing_consumption, cur_p_eng = takeoff_and_landing_consumption(actual_motion_consumption, tl_e_perc)
                actual_motion_consumption = actual_takeoff_and_landing_consumption
            # Flight (horizontal) phase:
            else:
                cur_p_eng = flight_phase()
            b_perc_per_hour = perc_battery_per_hour(cur_p_eng, actual_motion_consumption)
            timestep = dt

            battery_consumed = real_consumed_battery(b_perc_per_hour, timestep)
            time_needed += timestep
        # Estimation of the battery consumption (considering all the flight phases, i.e., takeoff, flight and landing):
        else:
            # Estimate the battery consumption for the flight phase:
            cur_p_eng = flight_phase()
            b_perc_per_hour = perc_battery_per_hour(cur_p_eng, actual_motion_consumption)
            timestep = self.hflight_time
            time_needed += timestep
            battery_consumed = real_consumed_battery(b_perc_per_hour, timestep)
            # estimate the battery consumption for both takeoff and landing phases:
            actual_takeoff_and_landing_consumption, cur_p_eng = takeoff_and_landing_consumption(actual_motion_consumption, tl_e_perc)
            timestep = takeoff_land_time
            b_perc_per_hour = perc_battery_per_hour(cur_p_eng, actual_takeoff_and_landing_consumption)
            battery_consumed += 2*real_consumed_battery(b_perc_per_hour, timestep)
            time_needed += 2*timestep

        return battery_consumed, time_needed

    def energy_charged(self, dt: Optional[float] = 5.) -> float:
        """
        Battery charging phase in time 'dt' ('dt' and 'T_e_to_fc' must be expressed in the same unit measure)

        :param dt: considered interval time
        :return: percentage of battery charged 
        """
        battery_charged = (100*dt)/self.T_e_to_fc
        
        return battery_charged

    def landing_point(self, POIs: List[Point]) -> Point:
        """
        Select a valid landing point for the current UAV.

        :param POIs: list of the available Point of Interests (i.e., landing points)
        :return: the AVAILABLE landing point closest to the selected target (i.e., the chosen landing point)
        """
        distances = [self.target.distance(p) for p in POIs]
        min_idx = np.argmin(distances)
        closest_POI = POIs[min_idx]
        
        return closest_POI

    @property
    def bearing(self) -> float:
        """
        Bearing from current position to target.
        :param: None
        :return: None
        """
        dx = self.target.x - self.position.x
        dy = self.target.y - self.position.y
        compass = math.atan2(dy, dx)
        return (compass + u.circle) % u.circle

    @property
    def shot_bearing(self) -> float:
        """
        Bearing from the current position to the shooring source
        """
        shot_pos = self.local_obs[LAST_SHOT_KEY][1]
        dx = flight.position.x - shot_pos.x
        dy = flight.position.y - shot_pos.y
        compass = math.atan2(dy, dx)
        return (compass + u.circle) % u.circle

    @property
    def shot_distance(self):
        # Last position where shoot was detected:
        shot_pos = self.local_obs[LAST_SHOT_KEY][1]         
        return self.position.distance(shot_pos)

    def shot_time(self, cur_abs_time: float) -> float:
        shot_time = self.local_obs[LAST_SHOT_KEY][0]
        elapsed_shot_time = cur_abs_time - shot_time 
        return elapsed_shot_time

    @property
    def shot_detected(self):
        shot = self.local_obs[LAST_SHOT_KEY][2]
        return shot

    def rotation_matrix(self) -> List:
        """
        Calculates the ZYX rotation matrix based on the roll, pitch, yaw angles of the UAVs.
        :param: None
        return: 3x3 rotation matrix as NumPy array
        """
        return np.array(
            [[cos(self.yaw) * cos(self.pitch), -sin(self.yaw) * cos(self.roll) + cos(self.yaw) * sin(self.pitch) * sin(self.roll), sin(self.yaw) * sin(self.roll) + cos(self.yaw) * sin(self.pitch) * cos(self.roll)],
             [sin(self.yaw) * cos(self.pitch), cos(self.yaw) * cos(self.roll) + sin(self.yaw) * sin(self.pitch) *
              sin(self.roll), -cos(self.yaw) * sin(self.roll) + sin(self.yaw) * sin(self.pitch) * cos(self.roll)],
             [-sin(self.pitch), cos(self.pitch) * sin(self.roll), cos(self.pitch) * cos(self.yaw)]
             ])

    @property
    def components(self) -> Tuple:
        """
        Compute X and Y Speed components.
        :param: None
        :return dx, dy: speed components
        """
        dx = self.airspeed * math.cos(self.track)
        dy = self.airspeed * math.sin(self.track)
        return dx, dy

    def new_components(self, communication: 'CommunicationModule', d: int, dt: Optional[float]=5.) -> Tuple[float]:
        """
        Compute the X,Y,Z position components based on a polynomial trajectory.
        
        :param communication: Communication class module
        :param d: degree of the polynomial trajectory to be used
        :param dt: resolution time step

        :return x_pos, y_pos, z_pos: X,Y,Z position components
        """
            
        # Check if the trajectory coeffs. on X-axis are empty. If so, then also the other on Y and Z axes will be empty (thus, do not update):
        if (self.x_c[0]==[]) and (self.x_c.count(self.x_c[0])==len(self.x_c)):
            return  

        trajectory_time = self.takingoff_time+self.hflight_time+self.landing_time
        cur_t_traj = self.cur_t_takeoff+self.cur_t_hflight+self.cur_t_landing

        # Already landed (on the ground):
        if (cur_t_traj>trajectory_time):
            self.takeoff = False
            self.flying = False
            self.landing = False
            self.start = copy.deepcopy(self.target) 
            self.position = copy.deepcopy(self.start)
            self.altitude = 0.
            self.sensing_mode(communication)
            return self.position.x, self.position.y, self.altitude # -> return the current position without updating it (the flight mode will be not enabled!)

        # Takeoff phase:
        if self.cur_t_takeoff<=self.takingoff_time:
            self.takeoff = True
            self.flying = False
            self.landing = False
            cur_t_traj = copy.deepcopy(self.cur_t_takeoff)
            self.cur_t_takeoff += dt
            x_c = self.x_c[0]
            y_c = self.y_c[0]
            z_c = self.z_c[0]
        # 'Pure' flight phase:
        elif self.cur_t_hflight<=self.hflight_time:
            self.takeoff = False
            self.flying = True
            self.landing = False
            cur_t_traj = copy.deepcopy(self.cur_t_hflight)
            self.cur_t_hflight += dt
            x_c = self.x_c[1]
            y_c = self.y_c[1]
            z_c = self.z_c[1]
        # Landing phase:
        elif self.cur_t_landing<=self.landing_time:
            self.takeoff = False
            self.flying = False
            self.landing = True
            cur_t_traj = copy.deepcopy(self.cur_t_landing)
            self.cur_t_landing += dt
            x_c = self.x_c[2]
            y_c = self.y_c[2]
            z_c = self.z_c[2]

        self.flight_mode
        self.cur_t_sensing = 0.

        # Compute position, velocity and acceleration based on the selected polynomial trajectory (uncomment the needed parameters):
        des_x_pos = TrajsSystem.calculate_position(x_c, cur_t_traj, d)
        des_y_pos = TrajsSystem.calculate_position(y_c, cur_t_traj, d)
        des_z_pos = TrajsSystem.calculate_position(z_c, cur_t_traj, d)
        #des_x_speed = TrajsSystem.calculate_velocity(x_c, cur_t_traj, d)
        #des_y_speed = TrajsSystem.calculate_velocity(y_c, cur_t_traj, d)
        #des_z_speed = TrajsSystem.calculate_velocity(z_c, cur_t_traj, d)
        #des_x_acc = TrajsSystem.calculate_acceleration(x_c, cur_t_traj, d)
        #des_y_acc = TrajsSystem.calculate_acceleration(y_c, cur_t_traj, d)
        #des_z_acc = TrajsSystem.calculate_acceleration(z_c, cur_t_traj, d)

        '''
        # Here you can use the most suitable controller:
        #CONTROLLER_TYPE can be set in global_vars.py
        self.controller(des_x_pos=des_x_pos, des_y_pos=des_y_pos, des_z_pos=des_z_pos,
                        des_x_speed=des_x_speed, des_y_speed=des_y_speed, des_z_speed=des_y_speed,
                        des_x_acc=des_x_acc, des_y_acc=des_y_acc, des_z_acc=des_z_acc,
                        controller_type=CONTROLLER_TYPE)

        New positions based on the controller of type CONTROLLER_TYPE:
        x_pos, y_pos, z_pos = self.position.x + self.x_speed*dt, self.position.y + self.y_speed*dt, self.altitude + self.z_speed*dt
        
        '''

        # New positions assuming that the controller allow the UAVs to perfectly interpolate all the waypoints of the flight path: 
        x_pos, y_pos, z_pos = des_x_pos[0], des_y_pos[0], des_z_pos[0]

        return x_pos, y_pos, z_pos

    def controller(self, des_x_pos: Optional[Union[float, None]]=None, des_y_pos: Optional[Union[float, None]]=None,
                         des_z_pos: Optional[Union[float, None]]=None, des_x_speed: Optional[Union[float, None]]=None,
                         des_y_speed: Optional[Union[float, None]]=None, des_z_speed: Optional[Union[float, None]]=None,
                         des_x_acc: Optional[Union[float, None]]=None, des_y_acc: Optional[Union[float, None]]=None,
                         des_z_acc: Optional[Union[float, None]]=None, control_type: Optional[float] = 1) -> None:
        """
        Apply the desired type type of controller. MOdify the scope of this method to add new controlers. 
        
        :param des_x_pos: desired X position
        :param des_y_pos: desired Y position 
        :param des_z_pos: desired Z position
        :param des_x_speed: desired X speed
        :param des_y_speed: desired Y speed
        :param des_z_speed: desired Z speed
        :param des_x_acc: desired X acceleration
        :param des_y_acc: desired Y acceleration
        :param des_z_acc: desired Z acceleration
        :param control_type: integer indicating the controller type to be used

        :return: None
        """

        if control_type==1:
            assert des_z_pos!=None and des_z_speed!=None and \
                   des_x_acc!=None and des_y_acc!=None and \
                   des_z_acc!=None, 'Incorrect values for desired parameters when using the controller type ' + str(control_type) + '!'

            # Compute the UAV components:
            thrust = M * (G + des_z_acc + Kp_z*(des_z_pos-self.altitude) + Kd_z*(des_z_speed-self.z_speed))

            # Desired Yaw is always set equal to 0:
            des_yaw = 0

            roll_torque = Kp_roll*( ( (des_x_acc*math.sin(des_yaw) - des_y_acc*math.cos(des_yaw))/G ) - self.roll)
            pitch_torque = Kp_pitch * ( ( (des_x_acc*math.cos(des_yaw) - des_y_acc*math.sin(des_yaw))/G ) - self.pitch)
            yaw_torque = Kp_yaw*(des_yaw-self.yaw)

            self.roll_speed += (roll_torque*dt)/Ixx
            self.pitch_speed += (pitch_torque*dt)/Iyy
            self.yaw_speed += (yaw_torque*dt)/Izz

            self.roll += self.roll_speed*dt
            self.pitch += self.pitch_speed*dt
            self.yaw += self.yaw_speed*dt

            R = self.rotation_matrix()
            acc = ( np.matmul(R, np.array([0, 0, thrust.item()]).T) - np.array([0, 0, M * G]).T )/M
            x_acc = acc[0]
            y_acc = acc[1]
            z_acc = acc[2]

            self.x_speed += x_acc*dt
            self.y_speed += y_acc*dt
            self.z_speed += z_acc*dt

        else:
            # Here you can add other types of controllers ...
            NotImplementedError('Controller type selected not implemented!')

    @property
    def distance(self) -> float:
        """
        Current distance to the target.
        :param: None
        :return: distance to the target
        """
        return round(self.position.distance(self.target), 2)

    @property
    def drift(self) -> float:
        """
        Drift angle (difference between track and bearing) to the target.
        :param: None
        :return: None
        """
        drift = self.bearing - self.track

        if drift > math.pi:
            return -(u.circle - drift)
        elif drift < -math.pi:
            return (u.circle + drift)
        else:
            return drift

    @property
    def shot_drift(self) -> float:
        """
        Drift angle between track and the bearing to the last position where a shoot was detected and the current agent.
        :param: None
        :return drift
        """
        drift = self.shot_bearing - self.track

        if drift > math.pi:
            return -(u.circle - drift)
        elif drift < -math.pi:
            return (u.circle + drift)
        else:
            return drift    

    @property
    def travel_phase_distance(self) -> float:
        # Planar (horizontal) distance needed to travel during the bang-coast-bang flight phase
        L = self.start.distance(self.target)
        return L

    @property
    def update_waypoints(self) -> None:
        """
        Update the waypoints of the current UAV.
        :param: None
        :return None
        """
        n_waypoints = len(self.waypoints)  
        start3D = [self.start.x, self.start.y, self.altitude]
        target3D = [self.target.x, self.target.y, 0.] # We always need to land on 0 altitude (CHANGE THIS IF YOU NEED TO LAND AT DIFFERENT ALTITTUDE!)
        # Whenever the waypoints are updated, then reset the current times taken for 3 flight phases:  
        self.cur_t_takeoff = 0.
        self.cur_t_hflight = 0.
        self.cur_t_landing = 0. 

        for wp_idx in range(n_waypoints):
            if wp_idx==0:
                wp = start3D
            elif wp_idx==1:
                wp = start3D[:2]
                wp.append(self.flight_altitude)
            elif wp_idx==2:
                wp = [target3D[0], target3D[1], self.flight_altitude]
            elif wp_idx==3:
                wp = target3D
            
            self.waypoints[wp_idx] = wp

    def traj_coefficient_update(self) -> None:
        """
        Compute and update the coefficient of the polynomial trajectory of the current UAV based on the normal
        conditions associated with the TRAJ_TYPE degree selected as the order of the polynomial trajectory.
        :param: None
        :return: None
        """

        self.update_waypoints

        # If the first and the last waypoint are the same, then no trajectory coeffs. computation is needed (it would also return a singular case!):
        if self.waypoints[0]==self.waypoints[-1]:
            self.x_c = [[] for flight_phase in range(3)]
            self.y_c = [[] for flight_phase in range(3)]
            self.z_c = [[] for flight_phase in range(3)]
            self.takingoff_time = 0.
            self.hflight_time = 0.
            self.landing_time = 0.
            return

        def compute_min_flight_time(p1: List, p2: List) -> float: #p1: float, p2: float
            """
            Compute the minimum flight time to interpolate 'p1' and 'p2' for the
            cartesian trajectory planning based on the UAV velocity and acceleration constranints. 
            
            :param p1: list containing the coordinates of the first 3D point
            :param p2: list containing the coordinates of the second 3D point
            
            :return T_min: The minimum time needed to interpolate 'p1' and 'p2'
            """
            #T_min = 0.

            '''
            _____________________________________________________________________________________________________
            We are assuming that max_speed and max_acc are the maximum speed and acceleration, respectively, for
            that can be reached on each axis (and hence they are also equal for each axis).
            _____________________________________________________________________________________________________
            '''

            # In teoria questo controllo andrebbe fatto su ogni asse; in pratica stai assumendo che max_acc
            # e max_speed siano la massima velocità e la massima accelerazione raggiungibili su ogni asse (e uguali per ogni asse) !!!!!!!!!!!!
            # Poi qui rifai i vari calcoli a seconda del type di p1 e p2 che passi !!!!!!!!!!!!!!!!!!!
            L = np.array(p1) - np.array(p2)
            L = np.linalg.norm(L)
            if TRAJ_TIPE==3:
                Tmin_v = (3/2)*(L/self.max_speed)
                Tmin_a = math.sqrt(6*(L/self.max_acc))
                T_min = max(Tmin_v, Tmin_a) # ceil(max(Tmin_v, Tmin_a))

            elif TRAJ_TIPE==5:
                Tmin_v = (15/8)*(L/self.max_speed)
                Tmin_a = math.sqrt((10*math.sqrt(3)/3)*(L/self.max_acc))
                T_min = max(Tmin_v, Tmin_a) # ceil(max(Tmin_v, Tmin_a))

            else: # TRAJ_TIPE==7 case --> it can be only this case since an 'assert' check on it has already been performed in 'scenario.py'
                Tmin_v = (35/16)*(L/self.max_speed)
                Tmin_a = math.sqrt((84*math.sqrt(5)/25)*(L/self.max_acc))
                # Tmin_j = math.sqrt((105/2)*(L/self.max_jerk)) # max jerk
                T_min = max(Tmin_v, Tmin_a) # #max(Tmin_v, Tmin_a, Tmin_j)  ceil(max(Tmin_v, Tmin_a, Tmin_j))

            return T_min

        wp1 = self.waypoints[0]
        wp2 = self.waypoints[1]
        wp3 = self.waypoints[2]
        wp4 = self.waypoints[3]

        # Compute the time needed for the three flight phases:
        self.takingoff_time = compute_min_flight_time(wp1, wp2)
        self.hflight_time = compute_min_flight_time(wp2, wp3)
        self.landing_time = compute_min_flight_time(wp3, wp4)

        traj_takeoff = TrajsSystem(start_pos=wp1, des_pos=wp2, T=self.takingoff_time, d=self.traj_type)
        traj_takeoff.solve()
        
        traj_hflight = TrajsSystem(start_pos=wp2, des_pos=wp3, T=self.hflight_time, d=self.traj_type)
        traj_hflight.solve()
        
        traj_landing = TrajsSystem(start_pos=wp3, des_pos=wp4, T=self.landing_time, d=self.traj_type)
        traj_landing.solve()

        self.x_c = [traj_takeoff.x_c, traj_hflight.x_c, traj_landing.x_c]
        self.y_c = [traj_takeoff.y_c, traj_hflight.y_c, traj_landing.y_c]
        self.z_c = [traj_takeoff.z_c, traj_hflight.z_c, traj_landing.z_c]

    @classmethod
    def generation(cls, scnr_cfg: "ScenarioConfig", train_cfg: "TrainingConfig", airspace: Airspace, fla: int, source_power: float,
                   POIs: List[Point], POIs_left: List[Point], flight_ID: int, flight_altitude: float,
                   desired_point: Optional[Union[Point, type(None)]] = None, tol: Optional[float] = 0):
        """
        Creates a random flight or a fixed flight.
        
        :param scnr_cfg: all the scenario parameters
        :param train_cfg: all the training parameters
        :param airspace: airspace where the flight is located
        :param fla: flight level assigned
        :param POIs: Point of Interests (landing points)
        :param POIs_left: a copy of 'POIs' to track the already selected POIs and remove them from the ones still available
        :param flight_ID: ID of the current flight (UAV)
        :param flight_altitude: altitude of the current flight
        :param desired_point: desired point where to deploy a specific UAV (to set only if a random deployment is not the desired choice)
        :param tol: tolerance to consider that the target has been reached (in meters)
        
        :return: random flight
        """
        NoneType = type(None)
        desired_p_type = type(desired_point)
        assert desired_p_type==NoneType or desired_p_type==Point, "The selected point for flights generation is not valid!"

        def random_point_in_polygon(polygon: Polygon) -> Point:
            minx, miny, maxx, maxy = polygon.bounds
            while True:
                point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
                if polygon.contains(point):
                    return point

        def random_deployment_among_landing_points(POIs: List[Point]) -> Point:
            p = random.sample(POIs, 1)[0]
            POIs.remove(p)

            return p

        def starting_target(POIs: List[Point], target: Point) -> Point:
            distances = [target.distance(p) for p in POIs]
            min_idx = np.argmin(distances)
            closest_POI = POIs[min_idx]

            return closest_POI

        # Random position
        if desired_p_type==NoneType:
            # Select a random position among the available landable points:
            position = random_deployment_among_landing_points(POIs_left)
        # Fixed position
        else:
            assert airspace.cartesian_polygon.contains(desired_point), "The selected fixed point for the flight is not inside the given polygon!"
            position = desired_point
        start = copy.deepcopy(position)
        
        # Target initialization (it is set equal to start):
        target = copy.deepcopy(start)

        max_speed = scnr_cfg.max_speed
        max_acc = scnr_cfg.max_acc
        ID = flight_ID
        # Null initial speed
        airspeed = 0.
        # Here give the correct values in the correct unit measures:
        cur_moving_time = 0.
        min_db_level = scnr_cfg.min_db_level
        factor = 10 # -> 10 for SIL, 20 for SPL
        min_sound_int = I0*pow(10, min_db_level/factor) # -> conversion from minumum dB level to minumum sound intensity
        # UAV footprint radius computed based on the minimum dB vevel that a UAV can detect:
        footprint_radius = math.sqrt(source_power/(4*math.pi*min_sound_int))
        fla = fla # -> no more effectively used (?)
        takingoff_time = 0.
        landing_time = 0.
        hflight_time = 0.
        sensing_time = scnr_cfg.sensing_time
        bcb_T = 10000.
        bcb_satisfied = True
        battery = 100
        is_charging = True
        p_eng = scnr_cfg.p_eng
        v_ref = scnr_cfg.v_ref
        p_bat = scnr_cfg.p_bat
        p_bat_charging = scnr_cfg.p_bat_charging
        b_efficiency = scnr_cfg.b_efficiency
        b_res = scnr_cfg.b_res
        battery_levels = [[b, b+b_res] for b in range(0, 100, b_res)]
        takeoff = False
        landing = False
        local_obs = {}
        local_obs[ID_KEY] = ID
        local_obs[POS_FEATURES_KEY] = [(BUSY, start, 0.), (BUSY, target, 0.)]
        # For the time being, only communicate the track and the battery which is set obvoiusly to 0 a the beginning (here you could add current speed, acceleration, ...):
        local_obs[AGENT_FEATURES_KEY] = (0., battery)
        local_obs[LAST_SHOT_KEY] = (math.nan, math.nan, 0) # -> amplitude and frequency could be also used
        local_obs[AOI_KEY] = 0.
        # 'received_obs' is initialized during the reset of the environment (at the initial step, every UAV knows the initial position of the other ones):
        received_obs = {}
        achievable_pos = []
        Rx = True
        Sx = True
        sensing_range_enabled = True
        clock = scnr_cfg.agent_clock
        memory = {}
        memory_size = train_cfg.batch_size
        '''
        The actions size only refers to the capacity to store the past actions (excluding the current one): 
        
        In the line below, the '+1' is due to the fact that during the learning the actions will be stored in memory when they are selected,
        but then they need to be passed to the state vector during the observation in the same time instant (maybe without any new update).
        All the actions in this case need to be passed except for the latest one, which is the one selected at the curren time instant, and thus,
        that action has not been executed yet.
        '''
        actions_size = train_cfg.actions_size +1
        altitude = 0.
        p1 = np.array([UAV_SIZE / 2, 0, 0, 1]).T
        p2 = np.array([-UAV_SIZE / 2, 0, 0, 1]).T
        p3 = np.array([0, UAV_SIZE / 2, 0, 1]).T
        p4 = np.array([0, -UAV_SIZE / 2, 0, 1]).T
        roll = 0.
        pitch = 0.
        yaw = 0.
        roll_speed = 0.
        pitch_speed = 0.
        yaw_speed = 0.
        cur_t_takeoff = 0.
        cur_t_landing = 0.
        cur_t_hflight = 0.
        cur_t_sensing = 0.
        waypoints = [math.nan for i in range(4)] # -> we have 4 waypoint for each UAV trajectory
        x_c = [[] for flight_phase in range(3)] # -> we have 3 flight phases: takeoff, horizontal flight, landing
        y_c = [[] for flight_phase in range(3)]
        z_c = [[] for flight_phase in range(3)]
        x_speed = 0.
        y_speed = 0.
        z_speed = 0.
        traj_type = TRAJ_TIPE
        assert traj_type==3 or traj_type==5 or traj_type==7, 'The polynomial trajectory degree can be only set equal to 2,5 or 7!' 
        x_traj_history = []
        y_traj_history = []
        z_traj_history = []
        n_motions_before_spotting = 0
        # Initialize to deafult values (i.e., 0.) all the actions prior to the first effective observation (or action selection):
        actions_memory = [DEFAULT_PAST_ACTION_VALUE for a in range(actions_size)] 

        return cls(position, start, target, ID, airspeed, cur_moving_time, min_db_level, footprint_radius, fla, takingoff_time, landing_time, hflight_time,
                   sensing_time, bcb_T, bcb_satisfied, battery, is_charging, p_eng, v_ref, p_bat, p_bat_charging, b_efficiency, b_res,
                   battery_levels, takeoff, landing, local_obs, received_obs, achievable_pos, max_speed, max_acc,
                   Rx, Sx, sensing_range_enabled, clock, memory, memory_size,
                   actions_memory, actions_size, flight_altitude, altitude, p1, p2, p3, p4, roll, pitch, yaw, roll_speed,
                   pitch_speed, yaw_speed, cur_t_takeoff, cur_t_landing, cur_t_hflight, cur_t_sensing, waypoints, x_c, y_c, z_c,
                   x_speed, y_speed, z_speed, traj_type, x_traj_history, y_traj_history, z_traj_history, n_motions_before_spotting) #, memory_interval_time

    @ classmethod
    def fixed(cls, airspace: Airspace, position: Point, min_speed: float, max_speed: float, flight_id: int, tol: float = 0.):
        """
        Creates a fixed flight

        :param airspace: airspace where the flight is located
        :param position: flight position
        :param min_speed: minimum speed of the flights (in kt)
        :param max_speed: maximum speed of the flights (in kt)
        :param flight_id: identifier for a flight
        :param tol: tolerance to consider that the target has been reached (in meters)
        
        :return: fixed flight
        """
        assert airspace.contains(
            position), "The point is outside of the Polygon"
        # Random target
        boundary = airspace.cartesian_polygon.boundary
        while True:
            d = random.uniform(0, airspace.cartesian_polygon.boundary.length)
            target = boundary.interpolate(d)
            if target.distance(position) > tol:
                break

        # Random speed
        airspeed = random.uniform(min_speed, max_speed)

        return cls(position, target, airspeed, flight_id)

class CommunicationModule:
    """
    CommunicationModule class
    """
    def __init__(self, source: Source, STAR: Optional[bool] = True, always_update_t: Optional[bool] = True, dt: Optional[float] = 5.) -> None:
        self.source = source
        self.STAR = STAR # -> enabled or not the STAR (Simultaneous Transmit and Receive) Antennas communication
        self.always_update_t = always_update_t # -> set or not the updating time of the communication (if it is not set up, then it is constantly and always updated)
        self.dt = dt # -> time step resolution
        self.delayed_obs_buffer = {} # -> the key is the time at which it will be received: the value is the 'key-delayed' local observation 
        self.data_losses = [] # -> contains the IDs of the UAVs associated with a data loss
        assert self.always_update_t==False or self.always_update_t==True, 'Constant communication updating time must be either True or False!' 

    def status(self, device: Union[Flight, ENode], sense: Optional[str] = 'ready_to_listen') -> None:
        """
        Update the communication status of a specific device (either the node or the agent).
        
        :param device: either ENode or agent
        :param sense: set the sensing status ('ready_to_listen' or 'ready_to_broadcast'): it is used only if STAR=False

        :return: None
        """
        assert self.STAR or (not self.STAR and (sense=='ready_to_listen' or sense=='ready_to_broadcast')), 'If STAR communication is not enabled, then the sensing action can be either only "ready_to_listen" or only "ready_to_broadcast"!'

        def listening(device) -> None:    
            device.Rx = True
            device.Sx = False

        def broadcast(device) -> None:
            device.Rx = False
            device.Sx = True

        if self.STAR:
            device.Rx = True
            device.Sx = True
        else:
            if sense=='ready_to_listen':
                listening(device=device)
            else:
                broadcast(device=device)

    @abc.abstractmethod
    def update_RxSx_mode(self, enode: ENode, agent: Flight):
        raise NotImplementedError('Method for Rx and Sx (for both UAVs and ENode) time/condition settings must be implemented!')
        '''
        self.sensing(enode)
        self.sensing(agent)
        if enode.queue_info!=[]:
            enode.Rx = True
        # set also all the orher conditions for both Rx and Sx for both enode and UAVs
        '''

    def info_exchange(self, enode: ENode, agent: Flight, cur_t: float, local_flight_obs: Dict, spawn_obs_time: float, n_observations_to_be_spawn: int, energy_constraint: Optional[bool] = False) -> None:
        """
        Infos exchanging through ENode-UAVs communication.
        
        :param enode: ENode
        :param agent: the UAV with which the ENode is communicating (it could be either an input or an output communication)
        :param cur_t: current absolute time
        :param local_flight_obs: the local observation to communicate (which obviously cannot be the one associated with 'flight')
        :param spawn_obs_time: the time at which 'local_flight_obs' is 'spawning' (it could be subjet to some delay, if any)
        :param n_observations_to_be_spawn: number of obseravations to spawn
        :param energy_constraint: a boolean indicating if the battery contraints must be taken into account or not

        :return: None
        """

        # Update the current absolute time:
        enode.update_current_t(t=cur_t)
        '''
        Apply the info exchange only if 'always_update_t==True' (i.e., the 'explicit_clock' is NOT enabled)
        OR in case in which the 'explicit_clock' IS ENABLE, then apply it according to the ENode clock: 
        '''
        if self.always_update_t or cur_t%enode.clock==0:
            # Potential inbound communication to the ENode (ENode is ready to receive):
            if enode.Rx:
                # always update the old infos (if any) stored in its buffer:
                enode.update_info_buffer
                '''
                The check on wether the agent is (or 'was', in case of observation delay) ready (i.e., agent.Sx==True)
                or not to send the info has been already done in 'obs_buffer_update' which has been already called at this point.
                '''
                # ENode collect the local infos received from the UAV that sent the 'local_flight_obs':
                enode.update_broadcast_obs(local_flight_obs=local_flight_obs, spawn_obs_time=spawn_obs_time, n_observations_to_be_spawn=n_observations_to_be_spawn)
            
            '''
            Potential inbound communication to the agent (UAV is ready to receive). This communication is performed every time that
            an agent is in Rx mode, even if the ENode is sending a delayed observation (it will be identified by its AoI anyway):
            '''
            if agent.Rx:
                # The ENode is ready to send infos:
                if enode.Sx:
                    # Enode broadcast the collected infos (from all the UAVs) to the current UAV which is receiving this 'global' info:
                    agent.update_received_obs(enode=enode, energy_constraint=energy_constraint)

    @staticmethod
    def store_most_recent_shot(shot1: Tuple, shot2: Tuple) -> Tuple: 
        """
        Store the most recent shot between the two used for the comparison.

        :param shot1: first shot features
        :param shot2: second shot features

        :return shot: most recent shot features
        """
        # If there are more shot features, then add them here!

        t1 = shot1[0]
        p1 = shot1[1]
        
        t2 = shot2[0]
        p2 = shot2[1]

        # It does not matter which shot features are going to save when no shoot has been detected:
        if math.isnan(t1) and math.isnan(t2):
            shot = shot1 # -> math.isnan
        elif math.isnan(t1) and not math.isnan(t2):
            shot = shot2
        elif not math.isnan(t1) and math.isnan(t2):
            shot = shot1
        elif t1>=0 and t2>=0:
            # If t1>t2 then it means that shot1 arrived after shot2 (i.e., shot1 is the latest shot):
            if t1>t2:
                shot = shot1
            else:
                shot = shot2
        # Case in which one or both the shoot times are negatives (and this should not happen!): 
        else:
            shot = shot1
            warnings.warn("Last detected shooting time is negative!")
            time.sleep(3)

        return shot

    def store_specific_shot_features(enode: ENode, flights: List[Flight], local_ID: Union[int, str], shot_info: Tuple, aoi: Optional[Union[float, None]] = None) -> Tuple:
        """
        Store the time associated with the most recent info related to the audio source coming from all the local
        observations of the agents and either the midpoint (if only 2 source infos are available) or the centroid 
        (if 3 or more source infos are available) related to the positions where the source has been spotting. Store
        also the IDs of all the agents spotting the source at the current time instant.
        
        :param ENode: ENode collecting all the observations
        :param flights: list of the flights (UAVs)
        :param local_ID: ID of the considered UAV
        :param shot_info: shot info
        :param aoi: if used, then a BACKupdate associated with the shot feature will be performed

        :return shot: updated shot info
        """

        def midpoint(p1: Union[float, None], p2: Union[float, None]) -> Union[Point, None]:
            return Point((p1.x+p2.x)/2, (p1.y+p2.y)/2)

        current_spotting_agents = []

        enode.update_source_buffer(local_ID=local_ID, shot_info=shot_info)
        # Backupadate case:
        if aoi!=None: 
            enode.memory[aoi]['source_info_buffer'][local_ID] = shot_info
            cur_source_buffer = list(enode.memory[aoi]['source_info_buffer'].values())
        # Take the list of all the local observations stored in the ENode buffer:
        else:
            cur_source_buffer = list(enode.source_info_buffer.values())
        
        # Remove all the 'math.nan' observations:
        available_source_buffer_infos = list(filter(lambda x: not math.isnan(x[0]), cur_source_buffer))
        # Take the list of all the positions (among the ones!=math.nan) associated with the shot local observations stored in the ENode buffer: 
        available_shot_pos = [info[1] for info in available_source_buffer_infos]
        # Mumber of all the info (!=math.nan) associated with the audio source: 
        n_available_shot_infos = len(available_source_buffer_infos)

        # Check the most recent shot (it will be used only to store the time associated to it):
        most_recent_shot = (math.nan, math.nan, [])
        for i, shot_info in enumerate(cur_source_buffer):
            shot_detected = shot_info[2]
            '''
            Observations are filled always in the same order, following the order in which each UAV (flight)
            occurs in the flights lists. thus, it is possible to compute the corresponding ID through the index:
            '''
            cur_ID = flights[i].ID
            if shot_detected==1:
                current_spotting_agents.append(cur_ID) 
            most_recent_shot = CommunicationModule.store_most_recent_shot(shot1=most_recent_shot, shot2=shot_info)
 
        if n_available_shot_infos==0:
            shot_pos = math.nan
        elif n_available_shot_infos==1:
            cur_shot = available_source_buffer_infos[0]
            shot_pos = cur_shot[1]
        elif n_available_shot_infos==2:
            shot1 = available_source_buffer_infos[0]
            shot2 = available_source_buffer_infos[1]
            shot_pos = midpoint(p1=shot1[1], p2=shot2[1])
        else: # -> n_available_shot_infos>2 case
            # Polygon made up by interpolating the positions where the current shots have been hearing:
            cur_shots_pol = Polygon([shot_info for shot_info in available_shot_pos])
            shot_pos = cur_shots_pol.centroid
        
        # The time associated with the shot is always the one referring to the last source detected:
        shot_t = most_recent_shot[0]
        shot = (shot_t, shot_pos, current_spotting_agents) 

        return shot

    def obs_buffer_update(self, flight: "Flight", t: float, spawn_time: Optional[Union[float, None]] = None, obs_delay=False, obs_loss=False, external_obs: Optional[Union[pd.DataFrame, None]] = None) -> None:
        """
        Store local observations in the observation buffer if the spawning observation time is not enabled,
        otherwise remove the local observation associated with the specificied spawning time for the local observation.
        This allows for dealing with delayed observations.

        :param flight:     the UAV agent
        :param t:          elapsed time
        :param spawn_time: if it is a float, then the corresponding key (representing one or more observations associated to it) will be removed from the 'delayed_obs_buffer'
        :param obs_delay: boolean indicating if observation delay is applied or not
        :param obs_delay: boolean indicating if applying data loss or not (this case is still not handled in the proper way)
        :param external_obs: external obseravtion file (if any)
        
        :return: None
        
        REMARK: since every used time (e.g., elapsed_time, dt, ...) in env.py is converted into seconds
                MIN_OBS_DELAY and MAX_OBS_DELAY are already expressed in seconds.
        """
        local_obs = flight.local_obs
        # 'Appending mode' enabled (i.e., when an agent is sensing a local observation):
        if spawn_time is None:
            # Send the local observation only if the Sx mode of the current agent is enabled:
            if flight.Sx:
                # Case in which the observation is not external:
                if not isinstance(external_obs, pd.DataFrame):
                    # Data loss is always set to Faslse as, for the time being, it is not handled:
                    data_loss = False
                    
                    # Data Loss handling:
                    if obs_loss and t!=0.: # -> no data loss 'allowed' at the very first time instant (assumption: no data loss at the beginning)
                        data_loss = random.random()>PROB_OBS_LOSS
                        if data_loss and flight.ID not in self.data_losses:
                            self.data_losses.append(flight.ID)
                    else:
                        if flight.ID in self.data_losses:
                            self.data_losses.remove(flight.ID)

                    # Case in which the observations delay is enabled: 
                    if (not data_loss) and obs_delay: # if obs_delay
                        # Case in which the delay occurs on the current observation (obviously it is applied only if the observation is not external):
                        if random.random()<PROB_OBS_DELAY:
                            obs_delay = random.uniform(MIN_OBS_DELAY, MAX_OBS_DELAY)
                            # Set the observation delay equal to an instant time which is the closest (larger) multiple of the sampling clock 'dt':
                            obs_delay = self.dt*(int(obs_delay/self.dt)+1)
                            # The spawing time of the current observation is given by the sum of the current elapsed time and the observation delay: 
                            spawn_time = t + obs_delay
                        # Case in which the delay does not occur on the current observation:
                        else:
                            # The spawning time of the current observation is 'now' (i.e., equal to the current elapsed time)
                            spawn_time = t
                    # Case in which the observations delay is disabled and/or the current data have been lost:
                    else:
                        spawn_time = t
                # Case in which the observation is external:
                else:
                    # the spawning time is equal to the current one, which is the one in which we are receiving the external observation:
                    spawn_time = t

                # Case in which only one observation is spawning at the current time:
                if spawn_time not in self.delayed_obs_buffer:
                    self.delayed_obs_buffer[spawn_time] = [copy.deepcopy(local_obs)]
                else:
                    '''
                    Case in which more than one observation is spawning at the current time.
                    Some of them could be also associated with the same UAV because of a possible observation delay:
                    '''
                    self.delayed_obs_buffer[spawn_time].append(copy.deepcopy(local_obs))
        # 'Removing mode' enabled (i.e., when an agent is getting the global observation from the ENode):
        else:
            self.delayed_obs_buffer.pop(spawn_time)

class TrajsSystem():
    """
    TrajsSystem class
    """
    def __init__(self, start_pos: List, des_pos: List, T: List, d: int, start_vel: Optional[List]=[0,0,0], des_vel: Optional[List]=[0,0,0], start_acc: Optional[List]=[0,0,0], des_acc: Optional[List]=[0,0,0], start_jerk: Optional[List]=[0,0,0], des_jerk: Optional[List]=[0,0,0]):
        # Position:
        self.start_x = start_pos[0]
        self.start_y = start_pos[1]
        self.start_z = start_pos[2]

        self.des_x = des_pos[0]
        self.des_y = des_pos[1]
        self.des_z = des_pos[2]

        # Velocity:
        self.start_x_vel = start_vel[0]
        self.start_y_vel = start_vel[1]
        self.start_z_vel = start_vel[2]

        self.des_x_vel = des_vel[0]
        self.des_y_vel = des_vel[1]
        self.des_z_vel = des_vel[2]

        # Acceleration:
        self.start_x_acc = start_acc[0]
        self.start_y_acc = start_acc[1]
        self.start_z_acc = start_acc[2]

        self.des_x_acc = des_acc[0]
        self.des_y_acc = des_acc[1]
        self.des_z_acc = des_acc[2]

        # Jerk:
        self.start_x_jerk = start_jerk[0]
        self.start_y_jerk = start_jerk[1]
        self.start_z_jerk = start_jerk[2]

        self.des_x_jerk = des_jerk[0]
        self.des_y_jerk = des_jerk[1]
        self.des_z_jerk = des_jerk[2]

        self.T = T
        self.d = d

    def solve(self) -> None:
        """
        Update the coefficient for the desired trajectory.
        :param: None
        :return: None 
        """
        # Cubic
        if self.d==3:
            A = np.array(
                [[0, 0, 0, 1],
                 [pow(self.T, 3), pow(self.T, 2), self.T, 1],
                 [0, 0, 1, 0],
                 [3*pow(self.T, 2), 2*self.T, 1, 0]
                ])
            b_x = np.array(
                [[self.start_x],
                 [self.des_x],
                 [self.start_x_vel],
                 [self.des_x_vel]
                ])

            b_y = np.array(
                [[self.start_y],
                 [self.des_y],
                 [self.start_y_vel],
                 [self.des_y_vel],
                ])

            b_z = np.array(
                [[self.start_z],
                 [self.des_z],
                 [self.start_z_vel],
                 [self.des_z_vel]
                ])

        # Quintic
        elif self.d==5:
            A = np.array(
                [[0, 0, 0, 0, 0, 1],
                 [pow(self.T, 5), pow(self.T, 4), pow(self.T, 3), pow(self.T, 2), self.T, 1],
                 [0, 0, 0, 0, 1, 0],
                 [5*pow(self.T, 4), 4*pow(self.T, 3), 3*pow(self.T, 2), 2*self.T, 1, 0],
                 [0, 0, 0, 2, 0, 0],
                 [20*pow(self.T, 3), 12*pow(self.T, 2), 6*self.T, 2, 0, 0]
                ])
            b_x = np.array(
                [[self.start_x],
                 [self.des_x],
                 [self.start_x_vel],
                 [self.des_x_vel],
                 [self.start_x_acc],
                 [self.des_x_acc]
                ])

            b_y = np.array(
                [[self.start_y],
                 [self.des_y],
                 [self.start_y_vel],
                 [self.des_y_vel],
                 [self.start_y_acc],
                 [self.des_y_acc]
                ])

            b_z = np.array(
                [[self.start_z],
                 [self.des_z],
                 [self.start_z_vel],
                 [self.des_z_vel],
                 [self.start_z_acc],
                 [self.des_z_acc]
                ])

        # Septic
        elif self.d==7:
            A = np.array(
                [[0, 0, 0, 0, 0, 0, 0, 1],
                 [pow(self.T, 7), pow(self.T, 6), pow(self.T, 5), pow(self.T, 4), pow(self.T, 3), pow(self.T, 2), self.T, 1],
                 [0, 0, 0, 0, 0, 0, 1, 0],
                 [7*pow(self.T, 6), 6*pow(self.T, 5), 5*pow(self.T, 4), 4*pow(self.T, 3), 3*pow(self.T, 2), 2*self.T, 1, 0],
                 [0, 0, 0, 0, 0, 2, 0, 0],
                 [42*pow(self.T, 5), 30*pow(self.T, 4), 20*pow(self.T, 3), 12*pow(self.T, 2), 6*self.T, 2, 0, 0],
                 [0, 0, 0, 0, 6, 0, 0, 0],
                 [210*pow(self.T, 4), 120*pow(self.T, 3), 60*pow(self.T, 2), 24*self.T, 6, 0, 0, 0]
                ])

            b_x = np.array(
                [[self.start_x],
                 [self.des_x],
                 [self.start_x_vel],
                 [self.des_x_vel],
                 [self.start_x_acc],
                 [self.des_x_acc],
                 [self.start_x_jerk],
                 [self.des_x_jerk]
                ])

            b_y = np.array(
                [[self.start_y],
                 [self.des_y],
                 [self.start_y_vel],
                 [self.des_y_vel],
                 [self.start_y_acc],
                 [self.des_y_acc],
                 [self.start_y_jerk],
                 [self.des_y_jerk]
                ])

            b_z = np.array(
                [[self.start_z],
                 [self.des_z],
                 [self.start_z_vel],
                 [self.des_z_vel],
                 [self.start_z_acc],
                 [self.des_z_acc],
                 [self.start_z_jerk],
                 [self.des_z_jerk]
                ])

        def is_invertible(a):
            return a.shape[0]==a.shape[1] and np.linalg.matrix_rank(a)==a.shape[0]

        if not is_invertible(A):
            print('----------------------------')
            print('NON INVERTIBILE:')
            print(A)
            print()
            print((self.start_x, self.start_y, self.start_z))
            print((self.des_x, self.des_y, self.des_z))
            print('----------------------------')
            breakpoint()

        # 'np.linalg.lstsq' can the best solutions even if we have a singular case with infinite solutions:
        self.x_c = np.linalg.solve(A, b_x) # np.linalg.lstsq(A, b_x)[0]
        self.y_c = np.linalg.solve(A, b_y) # np.linalg.lstsq(A, b_y)[0]
        self.z_c = np.linalg.solve(A, b_z) # np.linalg.lstsq(A, b_z)[0]

    @staticmethod
    def calculate_position(c: float, t: float, d: Optional[int]=5):
        """
        Compute a position given a set of quintic coefficients and a time.

        :param c: list of coefficients generated by a quintic polynomial trajectory generator
        :param t: time at which to compute the position
        :param d: degree of the polynomial trajecotry to be used
        
        :return pos: position
        """

        # Cubic
        if d==3:
            pos = c[0]*pow(t, 3) + c[1]*pow(t, 2) + c[2]*t + c[3]
        # Quintic
        elif d==5:
            pos = c[0]*pow(t, 5) + c[1]*pow(t, 4) + c[2]*pow(t, 3) + c[3]*pow(t, 2) + c[4]*t + c[5]
        # Septic
        elif d==7:
            pos = c[0]*pow(t, 7) + c[1]*pow(t, 6) + c[2]*pow(t, 5) + c[3]*pow(t, 4) + c[4]*pow(t, 3) + c[5]*pow(t, 2) + c[6]*t + c[7]

        return pos

    @staticmethod
    def calculate_velocity(c: float, t: float, d: Optional[int]=5) -> List:
        """
        Compute a velocity given a set of quintic coefficients and a time.
        
        :param c: list of coefficients generated by a quintic polynomial trajectory generator.
        :param t: time at which to calculate the velocity
        :param d: degree of the polynomial trajecotry to be used

        return vel: velocity
        """

        # Cubic
        if d==3: 
            vel = 3*c[0]*pow(t, 2) + 2*c[1]*t + c[2]
        # Quintic
        elif d==5:
            vel = 5*c[0]*pow(t, 4) + 4*c[1]*pow(t, 3) + 3*c[2]*pow(t, 2) + 2*c[3]*t + c[4]
        # Septic
        elif d==7:
            vel = 7*c[0]*pow(t, 6) + 6*c[1]*pow(t, 5) + 5*c[2]*pow(t, 4) + 4*c[3]*pow(t, 3) + 3*c[4]*pow(t, 2) + 2*c[5]*t + c[6]

        return vel

    @staticmethod
    def calculate_acceleration(c: float, t: float, d: Optional[int]=5)  -> List:
        """
        Compute an acceleration given a set of quintic coefficients and a time.
        
        :param c: list of coefficients generated by a quintic polynomial trajectory generator.
        :param t: time at which to calculate the velocity
        :param d: degree of the polynomial trajecotry to be used

        return acc: acceleration
        """

        # Cubic
        if d==3: 
            acc = 6*c[0]*t + 2*c[1]
        # Quintic
        elif d==5:
            acc = 20*c[0]*pow(t, 3) + 12*c[1]*pow(t, 2) + 6*c[2]*t + 2*c[3]
        # Septic
        elif d==7:
            acc = 42*c[0]*pow(t, 5) + 30*c[1]*pow(t, 4) + 20*c[2]*pow(t, 3) + 12*c[3]*pow(t, 2) + 6*c[4]*t + 2*c[5]

        return acc

    @staticmethod
    def calculate_jerk(c: float, t: float, d: Optional[int]=5) -> List:
        """
        Compute a jerk given a set of quintic coefficients and a time.
        
        :param c: list of coefficients generated by a quintic polynomial trajectory generator.
        :param t: time at which to calculate the velocity
        :param d: degree of the polynomial trajecotry to be used

        return jerk: jerk
        """

        # Cubic
        if d==3: 
            jerk = 6*c[0]
        # Quintic
        elif d==5:
            jerk = 60*c[0]*pow(t, 2) + 24*c[1]*t + 6*c[2]
        # Septic
        elif d==7:
            jerk = 210*c[0]*pow(t, 4) + 120*c[1]*pow(t, 3) + 60*c[2]*pow(t, 2) + 24*c[3]*t + 6*c[4]

        return jerk