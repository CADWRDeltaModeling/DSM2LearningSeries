# A module to give a higher level view of a DSM2 study
# FIXME: Move to pydsm
import os
from functools import lru_cache
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
# our imports
import pyhecdss
import pydsm
from pydsm.input import parser, network
from pydsm import hydroh5
from vtools.functions.filter import godin
# viz imports
import geoviews as gv
import hvplot.pandas
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')
import colorcet as cc
#
import param
import panel as pn
pn.extension()


def load_echo_file(fname):
    with open(fname, 'r') as file:
        df = parser.parse(file.read())
    return df

def abs_path(filename, hydro_echo_file, study_dir='../'):
    '''
    builds absolute path to file using the path to the hydro_echo_file and the study_dir
    Assumes the default study directory to be one above the location of the hydro_echo_file
    '''
    return os.path.normpath(os.path.join(os.path.dirname(hydro_echo_file),study_dir,filename))
#
def get_hydro_tidefile(tables):
    '''
    Get the hydro tidefile from the IO_FILE table in all tables list loaded by load_echo_file
    '''
    io_file_table = tables['IO_FILE']
    hdf5_file_io_row = io_file_table[io_file_table.TYPE=='hdf5']
    return hdf5_file_io_row.iloc[0]['FILE']
#
def get_in_out_edges(gc, nodeid):
    out_edges = []
    in_edges = []
    for n in gc.successors(nodeid):
        out_edges += list(gc[nodeid][n].values())
    for n in gc.predecessors(nodeid):
        in_edges += list(gc[n][nodeid].values())
    return {'in': in_edges, 'out': out_edges}

def get_in_out_channel_numbers(gc, nodeid):
    ine, oute = get_in_out_edges(gc, nodeid).values()
    ine, oute = [e['CHAN_NO'] for e in ine],[e['CHAN_NO'] for e in oute]
    return ine, oute

#
def get_inflows_outflow(hydro, in_channels, out_channels, timewindow):
    '''
    get the inflows and outflows corresponding to the list of channels for the given timewindow
    The in_channels and out_channels are assumed to be related by being in channel 
    '''
    in_flows = [hydro.get_channel_flow(cid,'upstream',timewindow='01JUL2008 0000 - 01AUG2008 0000') for cid in in_channels]
    out_flows = [hydro.get_channel_flow(cid,'downstream',timewindow='01JUL2008 0000 - 01AUG2008 0000') for cid in out_channels]

def get_data_for_source(sn):
    datasign = sn.sign
    dssfile = sn.file
    dsspath = sn.path
    return datasign*next(pyhecdss.get_ts(dssfile, dsspath))[0]
    
def load_dsm2_channelline_shapefile(channel_shapefile):
    return gpd.read_file(channel_shapefile).to_crs(epsg=3857)

def join_channels_info_with_dsm2_channel_line(dsm2_chan_lines, tables):
    return dsm2_chan_lines.merge(tables['CHANNEL'], right_on='CHAN_NO', left_on='id')

def load_dsm2_flowline_shapefile(shapefile):
    dsm2_chans = gpd.read_file(shapefile).to_crs(epsg=3857)
    # dsm2_chans.geometry=dsm2_chans.geometry.simplify(tolerance=50)
    dsm2_chans.geometry = dsm2_chans.geometry.buffer(250, cap_style=1, join_style=1)
    return dsm2_chans

def join_channels_info_with_shapefile(dsm2_chans, tables):
    return dsm2_chans.merge(tables['CHANNEL'], right_on='CHAN_NO', left_on='id')

def load_dsm2_node_shapefile(node_shapefile):
    return gpd.read_file(node_shapefile).to_crs(epsg=3857)

def to_node_tuple_map(nodes):
    nodes = nodes.set_index(nodes['id'])
    nodes = nodes.drop(['geometry','id'],axis=1)
    node_dict = nodes.to_dict(orient='index')
    return {k:(node_dict[k]['x'],node_dict[k]['y']) for k in node_dict}

def get_location_on_channel_line(channel_id, distance, dsm2_chan_lines):
    chan = dsm2_chan_lines[dsm2_chan_lines.CHAN_NO == channel_id]
    # chan_line = chan.boundary # chan is from a polygon
    try:
        pt = chan.interpolate(distance/chan.LENGTH, normalized=True)
    except: # if not a number always default to assuming its length
        pt = chan.interpolate(1, normalized=True)
    # chan.hvplot()*gpd.GeoDataFrame(geometry=pt).hvplot() # to check plot of point and line
    return pt

def get_runtime(tables):
    scalars = tables['SCALAR']
    rs=scalars[scalars['NAME'].str.contains('run')]
    tmap = dict(zip(rs['NAME'],rs['VALUE']))
    stime = tmap['run_start_date']+' '+tmap['run_start_time']
    etime = tmap['run_end_date']+' '+tmap['run_end_time']
    return pd.to_datetime(stime), pd.to_datetime(etime)


class DSM2Study:

    def __init__(self, echo_file):
        self.echo_file = echo_file
        self.tables = load_echo_file(echo_file)
        # build network view
        self.gc = network.build_network_channels(self.tables)
        # get handle to hydro tidefile
        self.hydro_tidefile = abs_path(get_hydro_tidefile(self.tables), self.echo_file)
        self.hydro = pydsm.hydroh5.HydroH5(self.hydro_tidefile)
        # replace file with absolute based on echo file location
        output_channels=self.tables['OUTPUT_CHANNEL']
        output_channels['FILE']=output_channels.apply(lambda r: abs_path(r['FILE'], self.echo_file), axis=1)
        # source flow from tidefile, so uses tidefile location for relative paths
        self.source_flow = self.hydro.get_input_table('/hydro/input/source_flow')
        self.source_flow['file']=self.source_flow.apply(lambda r: abs_path(r['file'], self.hydro_tidefile), axis=1)

    def load_channelline_shapefile(self, channel_shapefile):
        self.dsm2_chan_lines = load_dsm2_channelline_shapefile(channel_shapefile)
        self.dsm2_chan_lines = join_channels_info_with_dsm2_channel_line(self.dsm2_chan_lines, self.tables)

    def get_runtime(self):
        return get_runtime(self.tables)

    def get_output_channels(self):
        return self.tables['OUTPUT_CHANNEL']        

    def get_inflows_outflows(self, nodeid:int, timewindow:str):
        """
        For the node id, get the in flows (upstream channels) and out flows (downstream channels)

        Parameters
        ----------
        nodeid : int
            The node id
        timewindow : str
            timewindow string as two times separted by "-", e.g. 01JAN2000 - 05APR2001

        Returns
        -------
        tuple of arrays
           a tuple of 2 arrays i.e. inflows, outflows
        """        
        in_channels, out_channels = get_in_out_channel_numbers(self.gc, nodeid)
        in_flows = [self.hydro.get_channel_flow(cid,'upstream',timewindow=timewindow) for cid in in_channels]
        out_flows = [self.hydro.get_channel_flow(cid,'downstream',timewindow=timewindow) for cid in out_channels]
        return in_flows, out_flows

    def get_source_flows(self, nodeid: int):
        """
        get source flows  

        Parameters
        ----------
        nodeid : int
            node id

        Returns
        -------
        array of pandas dataframes
            array of source flow time series for the node typically diversion, return, seepage
        """        
        sflow_node = self.source_flow[self.source_flow['node']==nodeid]
        return [get_data_for_source(sn) for _, sn in sflow_node.iterrows()]

    def get_net_source_flow(self, nodeid: int):
        """
        uses the sign information from source flow table to addup the net source/sink value

        Parameters
        ----------
        nodeid : int
            node id

        Returns
        -------
        pandas DataFrame
            net source flow time series
        """        
        sdata = self.get_source_flows(nodeid)
        return sum([df.iloc[:,0] for df in sdata])

