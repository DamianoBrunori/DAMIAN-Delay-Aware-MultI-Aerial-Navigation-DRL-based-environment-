import sys
import os
import math
import warnings
import time
import ast
import pandas as pd
import geopandas as gpd
import pathlib
import json
from shapely.geometry import Point, Polygon
from typing import Optional, Tuple, List, Dict, Union
from muavenv.global_vars import PATH_FOR_EXT_COMMUNICATION, OBS_FILENAME_FOR_EXT_COMMUNICATION

class ExternalInfo:
    """
    ExternalInfo class
    """

    def __init__(self, FIR_file: Optional[Union[pathlib.Path, None]] = None, obs_file: Optional[Union[pathlib.Path, None]] = None, FIR_ID: Optional[Union[str, None]] = None, external_communication: Optional[bool] = False) -> None:
        self.FIR_file = FIR_file # FIR File -> Flight Information Region file in the Eurocontrol format
        self.obs_file = obs_file # -> external observation file
        self.FIR_ID = FIR_ID
        self.standard_FIR_cols = ['Airspace ID', 'Min Flight Level', 'Max Flight Level', 'Sequence Number', 'Latitude', 'Longitude']
        self.standars_obs_cols = ['ID', 'source time', 'lat', 'lon', 'source spotted', 'track', 'takeoff-landing', 'time', 'battery', 'AoI', 'FLA']
        self.external_communication = external_communication

    def from_csv_to_dataframe(self, file: pathlib.Path, reading_error_found: Optional[bool] = False) -> pd.DataFrame:
        """
        Extract a DataFrame after reading and checking a .csv file.

        :param file: .csv file
        :param reading_error_found: a boolean indicating if an error has already been found during the 'file' reading

        :return df: Dataframe associated with .csv file
        :return reading_error_found: it stores the updated value of 'reading_error_found' after the 'file' reading
        """
        def loading_dots_print() -> None:
            """
            print the loading points on the terminal.

            :param: None

            :return: None
            """
            loading_dot = ' .'
            for i in range(5):  
                print('Waiting .' + loading_dot*i, end="\r")
                time.sleep(1)
            # Remove last stdout line:
            sys.stdout.write('\x1b[2K')

        try:
            df = pd.read_csv(file, delimiter=",") #usecols=col_list 
        
        except FileNotFoundError:
            
            if not (file==self.FIR_file):
                df = pd.DataFrame()
                if not reading_error_found:
                    print('Observation file not found: {}'.format(file))
                else:
                    loading_dots_print()
                reading_error_found = True
            else:
                print('FIR file not found: {}'.format(file))
                sys.exit(1)
        
        except Exception as e:
            
            if not (file==self.FIR_file):
                df = pd.DataFrame()
                if not reading_error_found:
                    print('Error in opening observation file: {}'.format(file))
                    print(e)
                else:
                    loading_dots_print()
                reading_error_found = True
            else:
                print('Error in opening FIR file: {}'.format(file))
                print(e)
                sys.exit(1)
        
        return df, reading_error_found

    @property
    def FIR_file_read(self) -> Union[pd.DataFrame, List[Tuple[float, float]]]:
        """
        Extract a DataFrame from .csv FIR_file after checking it.

        :param: None

        :return: DataFrame associated with .csv FIR_file
        """
        file_extension = self.FIR_file_extension_check()

        if file_extension=='.csv':
            
            df, _ = self.from_csv_to_dataframe(file=self.FIR_file)
            FIR_headers = list(df.columns)
            # Check only the needed FIR features:
            for col in FIR_headers:
                if col=='ID' or col=='Latitude' or col=='Longitude':
                    assert (col in self.standard_FIR_cols), '{:s} header is not present inside FIR file {:s}'.format(col, self.FIR_file)    

            airspace_IDs = df['Airspace ID'].tolist()
            unique_airspace_IDs = set(airspace_IDs)       
            n_airspaces = len(unique_airspace_IDs)

            assert (n_airspaces>1 and airspace_ID!='') or (n_airspaces==1), 'There are more than one airpsace inside FIR file {:s}! You need to choose an airspace ID to select one of them!'.format(self.FIR_file)
            assert (self.FIR_ID!=None and (self.FIR_ID in unique_airspace_IDs)) or (self.FIR_ID==None and file_extension=='csv'), 'The selected airspace {:s} is either not inside the selected FIR file {:s} or you are trying to select a FIR_ID through a ".geojson" FIR file: do not use a FIR_ID if you are using a ".geojson" FIR file!'.format(self.FIR_ID, self.FIR_file)
            if n_airspaces==1 and self.FIR_ID==None:
                warnings.warn('Since no airspace ID has been selected and the selected FIR file {:s} contains just 1 airspace ID, then that single airspace ID will be automatically selected.').format(self.FIR_file)
            
            return df

        else:
            
            try:
                data = gpd.read_file(filename=self.FIR_file)
            except FileNotFoundError:
                print('Json file not found {}:'.format(self.FIR_file))
            
            try:
                polygons = data['geometry'].tolist()
            except Exception as e:
                print('Json file format does not match geojson format!')
                print(e)

            n_pol = len(polygons)
            if n_pol>1:
                print('Json file {} contains more than one polygon'.format(self.FIR_file))
                valid_pol = False
                while not valid_pol:
                    pol_id = input('Select a polygon based on their occurring order (from 1 to {}):\n\n'.format(n_pol))
                    pol_id = ast.literal_eval(pol_id)
                    valid_pol = isinstance(pol_id, int) and (0<pol_id<=n_pol)
                    if not valid_pol:
                        print('\nInvalid polygon selected!\n\n')
            else:
                pol_id = 0
            
            json_pol = polygons[pol_id-1]
            lonlat_pol = json_pol.boundary.coords.xy
            n_pol_boundaries = len(lonlat_pol[0])
            latlon_points = [(lonlat_pol[1][i], lonlat_pol[0][i]) for i in range(n_pol_boundaries)]

            return latlon_points

    def FIR_file_extension_check(self) -> str:
        """
        Check the extension fo the FIR file (if the FIR_file option is enabled).

        :param: None

        :return file_extension: the extension of the FIR_file entered by the user into the Terminal
        """
        _, file_extension = os.path.splitext(self.FIR_file)
        assert file_extension=='.csv' or file_extension=='.geojson', 'FIR file must be either ".csv" or "geojson" file!'

        return file_extension

    def obs_file_read(self) -> pd.DataFrame:
        """
        Extract a DataFrame from .csv obs_file after checking it.

        :param: None

        :return df: the DataFrame associated with the .csv obs_file
        """
        df = pd.DataFrame()
        reading_error_found = False
        # Check only the needed observations:            
        while df.empty:
            if not self.external_communication:
                self.obs_file = input('\n\n\n\nWaiting for an external observation: enter the file containing the current UAVs local observations:\n\n')
            else:
                self.obs_file = PATH_FOR_EXT_COMMUNICATION + OBS_FILENAME_FOR_EXT_COMMUNICATION

            external_obs_columns = list(df.columns)
            for col in external_obs_columns:
                assert (col in self.standars_obs_cols), '{:s} header is not present inside obs_file {:s}'.format(col, self.standars_obs_cols)
            
            if not self.external_communication:
                reading_error_found = False
            df, reading_error_found = self.from_csv_to_dataframe(file=self.obs_file, reading_error_found=reading_error_found)

        return df