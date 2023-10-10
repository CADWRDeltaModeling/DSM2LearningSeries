# -*- conding: utf-8 -*-

import os
import re
import numpy as np
import pandas as pd
import netCDF4 as nc
import pyhecdss

__all__ = ['read_hist_wateryear_types', 'read_calsim_wateryear_types', 'read_calsim3_wateryear_types', 
           'read_calsim_sacvalley_table', 'read_regulations', 
           'read_D1641FWS_conditional', 'read_dss_to_df', 
           'generate_regulation_timeseries', 'generate_regulation_timeseries_calsim','prep_df']    

MONTHS = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5,
          'JUN': 6, 'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}

WaterYearTypes = {1:'W', 2:'AN', 3:'BN', 4:'D', 5:'C'}

def read_hist_wateryear_types(fpath):
    """ Read a table containing water year types from a text file.
        The text file is the top part of http://cdec.water.ca.gov/reportapp/javareports?name=WSIHIST
        down to the last row of the first table (before min and others).

        Parameters
        ----------
        fpath: string-like
            a text file name to read

        Returns
        -------
        pandas.DataFrame
            A WY table. The column names are:
            'wy', 'sac_oct_mar', 'sac_apr_jul', 'sac_wysum', 'sac_index',
            'sac_yrtype', 'sjr_oct_mar', 'sjr_apr_jul', 'sjr_wysum',
            'sjr_index', 'sjr_yrtype'
    """
    df = pd.read_fwf(fpath, header=None,
                     names=['wy', 'sac_oct_mar', 'sac_apr_jul', 'sac_wysum',
                            'sac_index', 'sac_yrtype', 'sjr_oct_mar',
                            'sjr_apr_jul', 'sjr_wysum', 'sjr_index',
                            'sjr_yrtype'],
                     skiprows=14)
    return df
    
def read_calsim_wateryear_types(fpath):
    """ Read a table containing water year types from CalSim II water year types table file.
        The table file is found in "CONV/Lookup" directory of a CalSim II study.

        Parameters
        ----------
        fpath: string-like
            a text file name to read

        Returns
        -------
        pandas.DataFrame
            A WY table. The column names are:
            'wy', 'sac_index_num', 'sjr_index_num', 'shasta_index_num',
            'amer_D893_num', 'feath_index_num', 'trin_index_num',
            'amer_403030_num', 'dry_years', 'sac_yrtype', 'sjr_yrtype'
    """
    def num_to_letter (row,index):
        if row[index] == 1:
            return 'W'
        if row[index] == 2:
            return 'AN'
        if row[index] == 3:
            return 'BN'
        if row[index] == 4:
            return 'D'
        if row[index] == 5:
            return 'C'
    df = pd.read_csv(fpath, delim_whitespace = True, header=None,
                     names=['wy', 'sac_index_num', 'sjr_index_num', 'shasta_index_num',
                            'amer_D893_num', 'feath_index_num', 'trin_index_num',
                            'amer_403030_num', 'dry_years'],
                     skiprows=13)
    df['sac_yrtype'] = df.apply(lambda row: num_to_letter(row,'sac_index_num'), axis = 1)
    df['sjr_yrtype'] = df.apply(lambda row: num_to_letter(row,'sjr_index_num'), axis = 1)
    return df

def read_calsim_sacvalley_table(fpath):
    """ Read a table containing Sacramento Valley indices from CalSim II SacValleyIndex table file.
        The table file is found in "CONV/Lookup" directory of a CalSim II study.

        Parameters
        ----------
        fpath: string-like
            a text file name to read

        Returns
        -------
        pandas.DataFrame
            A Sac Valley index table. The column names are:
            'wy', 'OctMar', 'AprJul', 'WYsum', 'Index'
    """

    df = pd.read_csv(fpath, delim_whitespace = True, header=None,
                     names=['wy', 'OctMar', 'AprJul', 'WYsum', 'Index'],
                     skiprows=4)
    return df

def read_calsim3_wateryear_types(fpath,bparts_to_read='WYT_SAC_'):
    """ Read Calsim3 output file dv.dss
        'WYT_SAC_' as Sac Water Year Type
        get May value as yearly WYT
        
        Parameters
        ----------
        fpath: string-like
            a text file name to read
            
        Returns
        -------
        pandas.DataFrame
            A Sac Valley index table. The column names are:
            'wy', 'sac_yrtype'
    """
    df_c3wyt = read_dss_to_df(fpath,bparts_to_read=bparts_to_read)
    df_c3wyt = df_c3wyt.assign(year=lambda x: x['time'].map(lambda y: y.year),
                               month=lambda x: x['time'].map(lambda y: y.month))
    
    df = df_c3wyt[df_c3wyt['month']==5]
    df = df.assign(sac_yrtype=lambda x:x['value'].map(lambda y:WaterYearTypes[y]))
    df = df.rename(columns={'year':'wy'})
                   
    return df
    
def read_regulations(fpath, df_wyt):
    """ Read regulations and create irregular time series DataFrame from them.

        Parameters
        ----------
        fpath: string-like
            a CSV file that contains regulations
        df_wyt: Pandas.DataFrame
            a wateryear information DataFrame. Water years and Sac water year
            type from this DataFrame is used to create a time series.
            The expected columns are: location, wyt, month, date, val.
            The first column, 'location,' is the location of the regulation.
            Station names like 'RSAN018' would be easy to use.
            The second column, 'wyt,' is the short name of Sac water year type
            like 'W', 'BN'. The third and fourth columns, 'month' and 'date,'
            are when the regulation starts. The five columns, 'val,' is a
            regulation value.
            Currently no conditional regulation is expected in the CSV.

        Returns
        -------
        pandas.DataFrame
            an irregular time series of regulations in DataFrame.
    """
    # Read a CSV file
    df = pd.read_csv(fpath, comment='#')
    df[['month','date']] = df [['month','date']].applymap(np.int64)
    # Filter out years without yeartype information
    df_wy = df_wyt[df_wyt['sac_yrtype'].notnull()]
    locs = df['location'].unique()
    yr_start = df_wy['wy'].min()
    yr_end = df_wy['wy'].max()
    dfs = []
    for loc in locs:
        timestamps = []
        regs = []
        for wy in range(yr_start, yr_end + 1):
            wyt = df_wyt[df_wyt['wy'] == wy]['sac_yrtype'].iloc[0]
            mask = (df['wyt'] == wyt) & (df['location'] == loc)
            for i, row in df[mask].iterrows():
                mo = row['month']
                yr = wy - 1 if mo >= 10 else wy
                t_start = pd.to_datetime(f"{yr:d}-{mo}-{row['date']}")
                val = row['val']
                timestamps.append(t_start)
                regs.append(val)
        dfs.append(pd.DataFrame(
            data={'value': regs, 'location': loc}, index=timestamps))
    return pd.concat(dfs)

def read_D1641FWS_conditional(fpath1, fpath2, df, df_sri, df_wyt):
    """ Update regulation time series DataFrame with D1641 FWS conditional logic.

        Parameters
        ----------
        fpath1: string-like
            a CSV file that contains D1641 FWS West Suisun Marsh regulations  
            under low flow conditions
        fpath2: string-like
            a CSV file that contains D1641 FWS San Joaquin regulations under 
            low flow conditions
        df: pandas.DataFrame
            Irregular regulation timeseries DataFrame. 
        df_sri: Pandas.DataFrame
            Sacramento River Index DataFrame.
        df_wyt: Pandas.DataFrame
            a wateryear information DataFrame. Water years and Sac water year
            type from this DataFrame is used to create a time series.
            The expected columns are: location, wyt, month, date, val.
            The first column, 'location,' is the location of the regulation.
            Station names like 'RSAN018' would be easy to use.
            The second column, 'wyt,' is the short name of Sac water year type
            like 'W', 'BN'. The third and fourth columns, 'month' and 'date,'
            are when the regulation starts. The five columns, 'val,' is a
            regulation value.

        Returns
        -------
        pandas.DataFrame
            an irregular time series of regulations in DataFrame.
    """
    # Read a CSV file
    df_reg = pd.read_csv(fpath1, comment='#')
    # Filter out years without yeartype information
    df_wy = df_wyt[df_wyt['sac_yrtype'].notnull()]
    locs = df_reg['location'].unique()
    df['year'] = pd.DatetimeIndex(df.index).year
    df['month']  = pd.DatetimeIndex(df.index).month
    yr_start = df_wy['wy'].min()
    yr_end = df_wy['wy'].max()
    for loc in locs:
        timestamps = []
        regs = []
        df_reg_loc = df_reg.loc[df_reg['location'] == loc]
        for wy in range(yr_start + 1, yr_end):
            wyt = df_wyt[df_wyt['wy'] == wy]['sac_yrtype'].iloc[0]
            prev_svi = df_sri[df_sri['wy'] == wy - 1]['WYsum'].iloc[0]
            
            if wy > yr_start:
                prev_wyt = df_wyt[df_wyt['wy'] == wy - 1]['sac_yrtype'].iloc[0]
            else:
                prev_wyt = wyt
                
            if wy > yr_start + 1:
                prev2_wyt = df_wyt[df_wyt['wy'] == wy - 2]['sac_yrtype'].iloc[0]
            elif wy > yr_start:
                prev2_wyt = df_wyt[df_wyt['wy'] == wy - 1]['sac_yrtype'].iloc[0]
            else:
                prev2_wyt = wyt
                
            flag = 0
            if wyt == 'C' and (prev_wyt == 'D' or prev_wyt == 'C'):
                flag = 1
            elif wyt == 'D' and prev_svi < 11.35:
                flag = 1
            elif wyt == 'D' and (prev_wyt == 'D' or prev_wyt == 'C') and prev2_wyt == 'C' and wy > yr_start + 1:
                flag = 1
            
            if flag > 0: 
                for i, row in df_reg_loc.iterrows():
                    mo = row['month']
                    val = row['val']
                    df.loc[(df['location'] == loc) & (df['year'] == wy) & (df['month'] == mo),'value'] = val
    
    # Read a CSV file
    df_reg = pd.read_csv(fpath2, comment='#')
    locs = df_reg['location'].unique()
    for loc in locs:
        timestamps = []
        regs = []
        df_reg_loc = df_reg.loc[df_reg['location'] == loc]
        for wy in range(yr_start + 1, yr_end):
            wyt = df_wyt[df_wyt['wy'] == wy]['sac_yrtype'].iloc[0]
            svi = df_sri[df_sri['wy'] == wy]['WYsum'].iloc[0]
            sjr_flag = 0
            if wyt == 'D' and svi < 8.1:
                sjr_flag = 1
            if sjr_flag > 0: 
                for i, row in df_reg_loc.iterrows():
                    mo = row['month']
                    val = row['val']
                    df.loc[(df['location'] == loc) & (df['year'] == wy) & (df['month'] == mo),'value'] = val
    return(df)
    
def generate_regulation_timeseries(df_reg, df, freq=None):
    time = df['time'].unique()
    t_begin = time.min()
    t_end = time.max()
    if freq is None:
        raise NotImplementedError(
            'Auto interval detection not implemented yet')
    dfs = []
    for station in df_reg['location'].unique():
        df_reg_station = df_reg[df_reg['location'] == station]
        mask = (df_reg_station.index >= (t_begin - pd.Timedelta('365 days'))) & (
            df_reg_station.index <= (t_end + pd.Timedelta('365 days')))
        times = []
        values = []
        for i in range(df_reg_station[mask].shape[0] - 1):
            row_i = df_reg_station[mask].iloc[i]
            row_next = df_reg_station[mask].iloc[i + 1]
            r = pd.date_range(
                start=row_i.name, end=row_next.name, freq=freq, closed='left').to_list()
            v = np.full_like(r, row_i['value'])
            times.extend(r)
            values.extend(v)
        df_ec_reg_ts = pd.DataFrame(
            data={'time': times, 'value': values, 'station': station})
        mask = (df_ec_reg_ts['time'] >= t_begin) & (
            df_ec_reg_ts['time'] <= t_end)
        dfs.append(df_ec_reg_ts[mask])
    return pd.concat(dfs)

def read_dss_to_df(fpath, bparts_to_read=None,
                   cparts_to_read=None,
                   eparts_to_read=None,
                   fparts_to_read=None,
                   start_date_str=None, end_date_str=None,
                   with_metadata=False):
    """
        Convert a DSS File into a dataframe.

        Parameters:
        -----------
        fpath : str
            path to the DSS File
        bparts_to_read: list, optional
            list of part B to read in.
            If it is none, all available paths are read.
        fparts_to_read: list, optional
            list of part F to read in.
            If it is none, all available paths are read.
        start_date_str : str, optional
            this string should be a date in the format '%Y%m%d'
            and should refer to the earliest date to fetch
            (see http://strftime.org/)
        end_date_str : str, optional
            this string should be a date in the format '%Y%m%d'
            and should refer to the last date to fetch
            (see http://strftime.org/)
        with_metadata: boolean, optional
            If true, add two columns for unit and time.

        Returns:
        --------
        pandas.DataFrame :
            This data frame will contain all data contained within
            the entire DSS file given as input.
    """
    pyhecdss.set_message_level(2)
    dssfile = pyhecdss.DSSFile(fpath)  # create DSSFile object
    paths = dssfile.get_pathnames()  # fetch all internal paths
    dfs = []
    for path in paths:
        parts = path.split('/')
        if bparts_to_read is not None and parts[2] not in bparts_to_read:
            continue
        if cparts_to_read is not None and parts[3] not in cparts_to_read:
            continue
        if eparts_to_read is not None and parts[5] not in eparts_to_read:
            continue
        if fparts_to_read is not None and parts[6] not in fparts_to_read:
            continue
        data, cunits, ctype = dssfile.read_rts(path, start_date_str,
                                               end_date_str)
        try:
            data.index = data.index.to_timestamp()
        except:
            pass
        data = pd.melt(data.reset_index(), id_vars=[
                       'index'], value_vars=[path], var_name='pathname')
        data.rename(columns={'index': 'time'}, inplace=True)
        if with_metadata:
            data['cunits'] = cunits
            data['ctype'] = ctype
        dfs.append(data)
    if dfs is None:
        raise ValueError('No timeseries is read')
    df = pd.concat(dfs)
    return df


def generate_regulation_timeseries_calsim(fpath,freq='1MON'):
    """ Generate regulation timeseries (monthly)
        from Calsim output dv dss
        only for Emmaton, Jersey, Rock Slough
        
        Parameters:
        -----------
        fpath : str
            path to the DSS File
            
        Returns:
        --------
        pandas.DataFrame :
            an irregular time series of regulations in DataFrame
    """
    CALSIM_STD_STA = {'EM_EC_STD':'RSAC092',
                      'RS_EC_STD':'ROLD024',
                      'JP_EC_STD':'RSAN018'}
    stations = np.array([key for key in CALSIM_STD_STA], dtype=object)
    df_stds = read_dss_to_df(fpath, bparts_to_read=stations,
                             cparts_to_read=['SALINITY'], eparts_to_read=[freq])
    
    df_stds = df_stds[df_stds['time'] > '1922-9-30']
    df_stds = df_stds[df_stds['time'] < '2015-10-1']

    df_stds['station'] = df_stds['pathname'].str.split('/',expand=True)[2]
    df_stds = df_stds.drop('pathname',1)

    df_stds = df_stds.replace({"station": CALSIM_STD_STA})
    
    df_stds['variable'] = 'EC-MEAN-MONTHLY'
    df_stds['scenario_name'] = 'D1641 Monthly'
    
    # adjust values for notebook visualization
    df_stds['value'] = np.where(df_stds['value'] > 99000-1, 0, df_stds['value']) #Calsim 3 uses 99000 for non-regulation period
    df_stds['value'] = np.where(
        (df_stds['station'] =='RSAN018') & (df_stds['time'].dt.month.isin([9,10,11,12,1,2,3])), 
        0, df_stds['value']) #Calsim 3 has guidance standard for JER out of Apr-Aug
    
    ec_250chlr = df_stds[df_stds['station'] =='ROLD024']['value'].unique().max() #convert Chloride to EC
    df_stds['value'] = np.where(
        (df_stds['station'] =='ROLD024') , 
        ec_250chlr, df_stds['value']) #only check standard of Chloride 250
        
    return df_stds


def prep_df(scenarios,sta,var,intvl,df_wyt,period,src='ALL'):
    """ Generate DataFrame of required station + variable + time interval + time period
        from sets of DSM2 input or output or postprocess dss
        associated with water year type info
        
        Returns:
        --------
        pandas.DataFrame :
            a DataFrame containing all 
    """
    dfs = []
    for scenario in scenarios:
        fpath = scenario['fpath']
        fparts=None
        if (src!='ALL'):
            with pyhecdss.DSSFile(fpath) as d:
                catdf=d.read_catalog()
            fparts = catdf.F[0].split('-')[0]+'-'+src
            fparts = [fparts.upper()]
        name = scenario['name']
        df_sn = read_dss_to_df(fpath, bparts_to_read=sta,
                               cparts_to_read=var, eparts_to_read=intvl,
                               fparts_to_read=fparts,)
        df_sn['scenario_name'] = name
        dfs.append(df_sn)

    df = pd.concat(dfs)
    ds_parts = df['pathname'].map(lambda x: (x.split('/')))
    df = df.assign(station=ds_parts.map(lambda x: x[2]),
                   variable=ds_parts.map(lambda x: x[3]),
                   interval=ds_parts.map(lambda x: x[5]),
                   year=lambda x: x['time'].map(lambda y: y.year),
                   month=lambda x: x['time'].map(lambda y: y.month))

    # Add a wateryear type column
    df = df.assign(wateryear=lambda x: x['year'])
    mask = df['month'] > 9
    df.loc[mask, 'wateryear'] += 1
    # Join water year types
    df = df.join(df_wyt.set_index('wy')['sac_yrtype'], on='wateryear')

    df = df[df['time'] >= period[0]]
    df = df[df['time'] <= period[1]]
    
    return df
    