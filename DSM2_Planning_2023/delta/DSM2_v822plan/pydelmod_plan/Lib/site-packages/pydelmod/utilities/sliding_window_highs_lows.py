# -*- conding: utf-8 -*-

__author__ = "Brad Tom"
__version__ = "0.0.1"
__license__ = "MIT"

import os
import re
import numpy as np
import pandas as pd
import netCDF4 as nc
import pyhecdss


__all__ = ['find_sliding_window_highs_lows', 'remove_duplicate_highs_lows', ]


def find_sliding_window_highs_lows(dataframe, time_window_length):
    """
    Implementation of the 'modified sliding window algorithm', originally developed by Jon Burau,
    and modified by Brad Tom to use a 7 hour time window.
    Finds high and low tides for semidiurnal tidal data. 
    Results are saved to dataframe in columns labeled 'ssw_highs' and 'ssw_lows'

    @dataframe should be a Pandas data frame with a 'value' column containing values, and having 15 minute time intervals
    @time_window_length should be the time window length measured in time steps.
    """
    # Use rolling to do sliding window
    # results are 'semi-square waves', which intersect the original stage time series at highs and lows only
    semi_square_wave_high_values = []
    semi_square_wave_low_values = []
    rolling_window=int(time_window_length)
    min_periods = rolling_window
    window_type=None # all values weighted equally

    semi_square_wave_high_values = dataframe['value'].rolling(rolling_window, win_type=window_type, center=True).max()
    semi_square_wave_low_values = dataframe['value'].rolling(rolling_window, win_type=window_type, center=True).min()

    # write the results to the dataframe
    dataframe = dataframe.assign(ssw_highs = semi_square_wave_high_values,
                                 ssw_lows = semi_square_wave_low_values)

    # find the intersections, and remove all other points
    # remove all semi-square wave highs except those that match a data point
    dataframe.loc[(dataframe.value!=dataframe.ssw_highs),'ssw_highs'] = None
    # remove all semi-square wave lows except those that match a data point
    dataframe.loc[(dataframe.value!=dataframe.ssw_lows),'ssw_lows'] = None
    return remove_duplicate_highs_lows(dataframe, time_window_length)
    
def remove_duplicate_highs_lows(dataframe, time_window_length):
    """
    Remove non-None values in the 'ssw_highse' column that are less then (time_window_length) time steps after the previous non-None 'ssw_highs' value.
    Remove non-None values in the 'ssw_lows' column that are less then (time_window_length) time steps after the previous non-None 'ssw_lows' value.
    Because this loops through the dataframe, it's not very efficient.
    """
    last_high_time_index = None
    last_low_time_index = None
    
    for current_index, row in dataframe.iterrows():
        if current_index==0:
            last_high_time_index = None
            last_low_time_index = None
        date = row['time']
        station = row['station']
        variable = row['variable']
        value = row['value']
        ssw_high_value = row['ssw_highs']
        ssw_low_value = row['ssw_lows']
    
        if ssw_high_value is not None and not np.isnan(ssw_high_value) and not ssw_high_value == 'nan':
            if last_high_time_index is not None:
                if (current_index-last_high_time_index)<time_window_length:
                    # remove the value
                    dataframe.loc[(dataframe.index == current_index) & (dataframe['station'] == station) & 
                                  (dataframe['variable'] == variable), 'ssw_highs'] = None
                else:
                    # retain the new value, remember the index
                    last_high_time_index = current_index
            else:
                last_high_time_index = current_index

        if ssw_low_value is not None and not np.isnan(ssw_low_value) and not ssw_low_value == 'nan':
            if last_low_time_index is not None:
                if (current_index-last_low_time_index)<time_window_length:
                    # remove the value
                    dataframe.loc[(dataframe.index == current_index) & (dataframe['station'] == station) & 
                                  (dataframe['variable'] == variable), 'ssw_lows'] = None
                else:
                    # retain the new value, remember the index
                    last_low_time_index = current_index
            else:
                last_low_time_index = current_index
    return dataframe

