# Postpro-Model
from distutils.command.config import config
import os
import pydsm
from pydsm import postpro
import json
import sys


# dask is a parallel processing library. Using it will save a lot of time, but error
# messages will not be as helpful, so set to False for debugging.
# Another option is to set scheduler (below) to single_threaded

import dask
from dask.distributed import Client, LocalCluster
from pydelmod import calibplot
import panel as pn
import pandas as pd
import holoviews as hv
import json
import sys


class DaskCluster:
    def __init__(self, config_data):
        self.client=None
        self.dask_options_dict = config_data['dask_options_dict']

    def start_local_cluster(self):
        cluster = LocalCluster(n_workers=self.dask_options_dict['n_workers'],
                       threads_per_worker=self.dask_options_dict['threads_per_worker'],
                       memory_limit=self.dask_options_dict['memory_limit']) # threads_per_worker=1 needed if using numba :(
        self.client = Client(cluster)

    def stop_local_cluster(self):
        self.client.shutdown()
        self.client=None


def run_all(processors):
    tasks=[dask.delayed(postpro.run_processor)(processor,dask_key_name=f'{processor.study.name}::{processor.location.name}/{processor.vartype.name}') for processor in processors]
    dask.compute(tasks)
    # to use only one processor. Also prints more helpful messages
    #     dask.compute(tasks, scheduler='single-threaded')


def postpro_model(cluster, config_data, use_dask):
    # Setup for EC, FLOW, STAGE
    #import logging
    #logging.basicConfig(filename='postpro-model.log', level=logging.DEBUG)
    vartype_dict = config_data['vartype_dict']
    process_vartype_dict = config_data['process_vartype_dict']
    # this specifies the files that are to be post-processed. Using study_files_dict resulted in all study files being post-processed.
    postpro_model_dict = config_data['postpro_model_dict']
    location_files_dict = config_data['location_files_dict']
    try:
        for var_name in vartype_dict:
            vartype = postpro.VarType(var_name, vartype_dict[var_name])
            if process_vartype_dict[vartype.name]:
                print('processing model ' + vartype.name + ' data')
                for study_name in postpro_model_dict:
                    dssfile=postpro_model_dict[study_name]
                    locationfile = location_files_dict[vartype.name]
                    units=vartype.units
                    observed=False
                    processors=postpro.build_processors(dssfile, locationfile, vartype.name, units, study_name, observed)
                    print(f'Processing {vartype.name} for study: {study_name}')
                    if use_dask:
                        run_all(processors)
                    else:
                        for p in processors:
                            postpro.run_processor(p)
    except:
        print('exception caught in postpro-model.py.run_processes. exiting.')
    finally:
        # Always shut down the cluster when done.
        if use_dask:
            cluster.stop_local_cluster()

def postpro_observed(cluster, config_data, use_dask):
    # Setup for EC, FLOW, STAGE
    vartype_dict = config_data['vartype_dict']
    process_vartype_dict = config_data['process_vartype_dict']
    observed_files_dict = config_data['observed_files_dict']
    location_files_dict = config_data['location_files_dict']
    
    try:
        for vartype in vartype_dict:
            if process_vartype_dict[vartype]:
                print('processing observed ' + vartype + ' data')
                dssfile = observed_files_dict[vartype]
                location_file = location_files_dict[vartype]
                units = vartype_dict[vartype]
                study_name='Observed'
                observed=True
                processors=postpro.build_processors(dssfile, location_file, vartype, units, study_name, observed)
                if use_dask:
                    run_all(processors)
                else:
                    for p in processors:
                        postpro.run_processor(p)
    finally:
        # Always shut down the cluster when done.
        if use_dask:
            cluster.stop_local_cluster()

# Build and save plot for each location
def export_svg(plot, fname):
    ''' This doesn't work. It would be very useful. '''
    ''' export holoview object to filename fname '''
    from bokeh.io import export_svgs
    p =  hv.render(plot, backend='bokeh')
    p.output_backend = "svg"
    export_svgs(p, filename=fname)


def save_to_graphics_format(calib_plot_template,fname):
    #     hvobj=calib_plot_template[1][0]
    #     hvobj.object=hvobj.object.opts(toolbar=None) # remove the toolbar from the second row plot
    calib_plot_template.save(fname)

def build_plot(config_data, studies, location, vartype, gate_studies=None, gate_locations=None, gate_vartype=None):
# def build_plot(config_data, studies, location, vartype):
    options_dict = config_data['options_dict']
    inst_plot_timewindow_dict = config_data['inst_plot_timewindow_dict']
    inst_plot_timewindow = inst_plot_timewindow_dict[vartype.name]
    timewindow_dict = config_data['timewindow_dict']
    timewindow = timewindow_dict[timewindow_dict['default_timewindow']]
    zoom_inst_plot = options_dict['zoom_inst_plot']
    gate_file_dict = config_data['gate_file_dict'] if 'gate_file_dict' in config_data else None
    flow_or_stage = (vartype.name == 'FLOW') or (vartype.name == 'STAGE')
    if location=='RSAC128-RSAC123':
        print('cross-delta flow')
        flow_or_stage = False
    flow_in_thousands = (vartype.name == 'FLOW')
    units = vartype.units
    include_kde_plots = options_dict['include_kde_plots']

    calib_plot_template, metrics_df = \
        calibplot.build_calib_plot_template(studies, location, vartype, timewindow, \
            tidal_template=flow_or_stage, flow_in_thousands=flow_in_thousands, units=units,inst_plot_timewindow=inst_plot_timewindow, include_kde_plots=include_kde_plots,
            zoom_inst_plot=zoom_inst_plot, gate_studies=gate_studies, gate_locations=gate_locations, gate_vartype=gate_vartype)
    # calib_plot_template, metrics_df = \
    #     calibplot.build_calib_plot_template(studies, location, vartype, timewindow, \
    #         tidal_template=flow_or_stage, flow_in_thousands=flow_in_thousands, units=units,inst_plot_timewindow=inst_plot_timewindow, include_kde_plots=include_kde_plots,
    #         zoom_inst_plot=zoom_inst_plot)
    if calib_plot_template is None:
        print('failed to create plots')
    if metrics_df is None:
        print('failed to create metrics')
    else:
        location_list = []
        for r in range(metrics_df.shape[0]):
            location_list.append(location)
        metrics_df['Location'] = location_list
        # move Location column to beginning
        cols = list(metrics_df)
        cols.insert(0, cols.pop(cols.index('Location')))
        metrics_df = metrics_df.loc[:, cols]
    return calib_plot_template, metrics_df


def build_and_save_plot(config_data, studies, location, vartype, gate_studies=None, gate_locations=None, gate_vartype=None, write_html=False, write_graphics=True, output_format='png'):
# def build_and_save_plot(config_data, studies, location, vartype, write_html=False, write_graphics=True, output_format='png'):
    study_files_dict = config_data['study_files_dict']
    output_plot_dir = config_data['options_dict']['output_folder']
    output_plot_dir = config_data['options_dict']['output_folder']
    print('Building plot template for location: ' + str(location))    

    calib_plot_template, metrics_df = build_plot(config_data, studies, location, vartype, gate_studies=gate_studies, gate_locations=gate_locations, gate_vartype=gate_vartype)
    # calib_plot_template, metrics_df = build_plot(config_data, studies, location, vartype)
    if calib_plot_template is None:
        print('failed to create plots')
    if metrics_df is None:
        print('failed to create metrics')
    os.makedirs(output_plot_dir, exist_ok=True)
    # save plot to html and/or png file
    if calib_plot_template is not None and metrics_df is not None:
        if write_html: 
            calib_plot_template.save(f'{output_plot_dir}{location.name}_{vartype.name}.html')
        if write_graphics:
            save_to_graphics_format(calib_plot_template,f'{output_plot_dir}{location}_{vartype.name}.png')
    #         export_svg(calib_plot_template,f'{output_plot_dir}{location.name}_{vartype.name}.svg')
    if metrics_df is not None:
        location_list = []
        for r in range(metrics_df.shape[0]):
            location_list.append(location)
        metrics_df['Location'] = location_list
        # move Location column to beginning
        cols = list(metrics_df)
        cols.insert(0, cols.pop(cols.index('Location')))
        metrics_df = metrics_df.loc[:, cols]

        # files for individual studies
        for study in study_files_dict:
            metrics_df[metrics_df.index == study].to_csv(
                output_plot_dir + '0_summary_statistics_' + study + '_' + vartype.name + '_' + location.name + '.csv')
            # metrics_df[metrics_df.index==study].to_html(output_plot_dir+'0_summary_statistics_'+study+'_'+vartype.name+'_'+location.name+'.html')
    return

# merge study statistics files
def merge_statistics_files(vartype, config_data):
    """Statistics files are written for each individual run, which is necessary (?) with dask
    This method merges all of them.
    """
    options_dict = config_data['options_dict']

    import glob, os
    print('merging statistics files')
    output_dir = options_dict['output_folder']
    os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(output_dir + '0_summary_statistics_*'+vartype.name+'*.csv')
    frames = []
    for f in files:
        frames.append(pd.read_csv(f))
    result_df = pd.concat(frames)
    result_df.sort_values(by=['Location', 'DSM2 Run'], inplace=True, ascending=True)
    result_df.to_csv(output_dir + '1_summary_statistics_all_'+vartype.name+'.csv', index=False)
    for f in files:
        os.remove(f)

def postpro_plots(cluster, config_data, use_dask):
    vartype_dict = config_data['vartype_dict']
    process_vartype_dict = config_data['process_vartype_dict']
    location_files_dict = config_data['location_files_dict']
    observed_files_dict = config_data['observed_files_dict']
    study_files_dict = config_data['study_files_dict']
    inst_plot_timewindow_dict = config_data['inst_plot_timewindow_dict']
    gate_file_dict = config_data['gate_file_dict'] if 'gate_file_dict' in config_data else None
    gate_location_file_dict = config_data['gate_location_file_dict'] if 'gate_location_file_dict' in config_data else None
    ## Set options and run processes. If using dask, create delayed tasks
    try:
        gate_vartype = postpro.VarType('POS', '')
        for var_name in vartype_dict:
            vartype = postpro.VarType(var_name, vartype_dict[var_name])
            print('vartype='+str(vartype))
            if process_vartype_dict[vartype.name]:
                # set a separate timewindow for instantaneous plots
                inst_plot_timewindow = inst_plot_timewindow_dict[vartype.name]
                ## Load locations from a .csv file, and create a list of postpro.Location objects
                #The .csv file should have atleast 'Name','BPart' and 'Description' columns
                locationfile=location_files_dict[vartype.name]
                dfloc = postpro.load_location_file(locationfile)
                locations = [postpro.Location(r['Name'],r['BPart'],r['Description']) for i,r in dfloc.iterrows()]

                # now get gate data
                gate_studies = None
                gate_locations = None
                if gate_file_dict is not None and gate_location_file_dict is not None:
                    gate_locationfile=gate_location_file_dict['GATE']
                    df_gate_loc = postpro.load_location_file(gate_locationfile, gate_data=True)
                    print('df_gate_loc='+str(df_gate_loc))
                    gate_locations = [postpro.Location(r['Name'],r['BPart'],r['Description']) for i,r in df_gate_loc.iterrows()]
                    print('gate_locations: '+str(gate_locations))
                    gate_studies = [postpro.Study('Gate',gate_file_dict[name]) for name in gate_file_dict]

                # create list of postpro.Study objects, with observed Study followed by model Study objects
                obs_study=postpro.Study('Observed',observed_files_dict[vartype.name])
                model_studies=[postpro.Study(name,study_files_dict[name]) for name in study_files_dict]
                studies=[obs_study]+model_studies


                # now run the processes
                if use_dask:
                    print('using dask')
                    tasks = [dask.delayed(build_and_save_plot)(config_data, studies, location, vartype, 
                                                    write_html=True,write_graphics=False, gate_studies = gate_studies, 
                                                    gate_locations=gate_locations, gate_vartype=gate_vartype,
                                                    dask_key_name=f'build_and_save::{location}:{vartype}') for location in locations]
                    # tasks = [dask.delayed(build_and_save_plot)(config_data, studies, location, vartype, 
                    #                                 write_html=True,write_graphics=False,                                                     
                    #                                 dask_key_name=f'build_and_save::{location}:{vartype}') for location in locations]
                    dask.compute(tasks)
                else:
                    print('not using dask')
                    for location in locations:
                        build_and_save_plot(config_data, studies, location, vartype, write_html=True,write_graphics=False,
                                            gate_studies = gate_studies, gate_locations=gate_locations, gate_vartype=gate_vartype)
                merge_statistics_files(vartype, config_data)
    finally:
        if use_dask:
            cluster.stop_local_cluster()


def run_process(process_name, config_filename, use_dask):
    '''
    process_name (str): should be 'model', 'observed', or 'plots'
    config_filename (str): filename of config (json) file
    use_dask (boolean): if true, dask will be used
    '''
    # Read Config file
    with open(config_filename) as f:
        config_data = json.load(f)

    # Create cluster if using dask
    cluster = None
    c_link = None
    if use_dask:
        cluster = DaskCluster(config_data)
        cluster.start_local_cluster()
        c_link=cluster.client
        print(c_link)
    c_link
    if process_name.lower() == 'model':
        postpro_model(cluster, config_data, use_dask)
    elif process_name.lower() == 'observed':
        postpro_observed(cluster, config_data, use_dask)
    elif process_name.lower() == 'plots':
        postpro_plots(cluster, config_data, use_dask)
    else:
        print('Error in pydelmod.postpro: process_name unrecognized: '+process_name)

# if __name__ == "__main__":
#     main()