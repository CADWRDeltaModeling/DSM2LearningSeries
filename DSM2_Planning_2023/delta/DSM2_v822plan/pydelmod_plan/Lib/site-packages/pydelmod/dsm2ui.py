# User interface components for DSM2 related information
import panel as pn
import param
import colorcet as cc
from pandas import date_range
from .dsm2study import *
# viz imports
import geoviews as gv
import hvplot.pandas
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')
#
pn.extension()


def build_output_plotter(channel_shapefile, hydro_echo_file, variable='flow'):
    hydro_tables = load_echo_file(hydro_echo_file)
    dsm2_chan_lines = load_dsm2_channelline_shapefile(channel_shapefile)
    dsm2_chan_lines = join_channels_info_with_dsm2_channel_line(dsm2_chan_lines, hydro_tables)
    output_channels = hydro_tables['OUTPUT_CHANNEL']
    output_dir = os.path.dirname(hydro_echo_file)
    output_channels['FILE'] = output_channels['FILE'].str.replace(
        './output', output_dir, regex=False)
    pts = output_channels.apply(lambda row: get_location_on_channel_line(row['CHAN_NO'], row['DISTANCE'], dsm2_chan_lines).values[0],
                                axis=1, result_type='reduce')
    output_channels = gpd.GeoDataFrame(output_channels, geometry=pts, crs={'init': 'epsg:26910'})
    time_range = get_runtime(hydro_tables)
    plotter = DSM2StationPlotter(output_channels[output_channels.VARIABLE == variable], time_range)
    return plotter


class DSM2StationPlotter(param.Parameterized):
    """Plots all data for single selected station

    """
    selected = param.List(
        default=[0], doc='Selected node indices to display in plot')
    date_range = param.DateRange()  # filter by date range
    godin = param.Boolean()  # godin filter and display

    def __init__(self, stations, time_range, **kwargs):
        super().__init__(**kwargs)
        self.date_range = time_range
        self.godin = False
        self.stations = stations
        self.points_map = self.stations.hvplot.points('easting', 'northing',
                                                      geo=True, tiles='CartoLight', crs='EPSG:3857',
                                                      project=True,
                                                      frame_height=400, frame_width=300,
                                                      fill_alpha=0.9, line_alpha=0.4,
                                                      hover_cols=['CHAN_NO', 'NAME', 'VARIABLE'])
        self.points_map = self.points_map.opts(opts.Points(tools=['tap', 'hover'], size=5,
                                                           nonselection_color='red', nonselection_alpha=0.3,
                                                           active_tools=['wheel_zoom']))
        self.map_pane = pn.Row(self.points_map)
        # create a selection and add it to a dynamic map calling back show_ts
        self.select_stream = hv.streams.Selection1D(
            source=self.points_map, index=[0])
        self.select_stream.add_subscriber(self.set_selected)
        self.meta_pane = pn.Row()
        self.ts_pane = pn.Row()

    def set_selected(self, index):
        if index is None or len(index) == 0:
            pass  # keep the previous selections
        else:
            self.selected = index

    @lru_cache(maxsize=25)
    def _get_all_sensor_data(self, name, var, intvl, file):
        # get data for location
        return [next(pyhecdss.get_ts(file, f'//{name}/{var}//{intvl}//'))[0]]

    def _get_selected_data_row(self):
        index = self.selected
        if index is None or len(index) == 0:
            index = self.selected
        # Use only the first index in the array
        first_index = index[0]
        return self.stations.iloc[first_index, :]

    # @@ callback to get data for index
    def get_selected_data(self):
        dfselected = self._get_selected_data_row()
        # NAME	CHAN_NO	DISTANCE	VARIABLE	INTERVAL	PERIOD_OP	FILE
        stn_name = dfselected['NAME']
        chan_id = dfselected['CHAN_NO']
        dist = dfselected['DISTANCE']
        var = dfselected['VARIABLE']
        intvl = dfselected['INTERVAL']
        file = dfselected['FILE']
        data_array = self._get_all_sensor_data(stn_name, var, intvl, file)
        stn_id = f'{chan_id}-{dist}_{var}'
        return data_array, stn_id, stn_name

    # @@ callback to display ..
    @param.depends('selected')
    def show_meta(self):
        dfselected = self._get_selected_data_row()
        # Points is not serializable to JSON https://github.com/bokeh/bokeh/issues/8423
        dfselected = dfselected.drop('geometry')
        self.data_frame = pn.widgets.DataFrame(dfselected.to_frame())
        return self.data_frame

    # @ callback to display ..
    @param.depends('selected', 'date_range', 'godin')
    def show_ts(self):
        data_array, stn_id, stn_name = self.get_selected_data()
        crv_list = []  # left here for multi curve later
        for data in data_array:
            if self.godin:
                el = hv.Curve(godin(data), label='godin filtered')
            else:
                el = hv.Curve(data)
            el = el.opts(title=f'Station: {stn_id} :: {stn_name}',
                         xlim=self.date_range, ylabel='Time')
            crv_list.append(el)
        layout = hv.Layout(crv_list).cols(1).opts(opts.Curve(width=900))
        return layout.opts(title=f'{stn_id}: {stn_name}')

    def get_panel(self):
        slider = pn.Param(self.param.date_range, widgets={
                          'date_range': pn.widgets.DatetimeRangePicker})
        godin_box = pn.Param(self.param.godin, widgets={'godin': pn.widgets.Checkbox})
        self.meta_pane = pn.Row(self.show_meta)
        self.ts_pane = pn.Row(self.show_ts)
        return pn.Column(pn.Row(pn.Column(self.map_pane, slider, godin_box), self.meta_pane), self.ts_pane)


class DSM2FlowlineMap:

    def __init__(self, shapefile, hydro_echo_file, hydro_echo_file_base=None):
        self.shapefile = shapefile
        self.hydro_echo_file = hydro_echo_file
        self.hydro_echo_file_base = hydro_echo_file_base
        self.dsm2_chans = load_dsm2_flowline_shapefile(self.shapefile)
        self.dsm2_chans.geometry = self.dsm2_chans.geometry.buffer(250, cap_style=1, join_style=1)
        self.tables = load_echo_file(self.hydro_echo_file)
        if self.hydro_echo_file_base:
            self.tables_base = load_echo_file(self.hydro_echo_file_base)
            # assumption that there is a match on the index of the tables
            for column in ['MANNING', 'LENGTH', 'DISPERSION']:
                self.tables['CHANNEL'].loc[:, column] = self.tables['CHANNEL'].loc[:,
                                                                                   column] - self.tables_base['CHANNEL'].loc[:, column]
        self.dsm2_chans_joined = self._join_channels_info_with_shapefile(
            self.dsm2_chans, self.tables)
        self.map = hv.element.tiles.CartoLight().opts(width=800, height=600, alpha=0.5)

    def _join_channels_info_with_shapefile(self, dsm2_chans, tables):
        return dsm2_chans.merge(tables['CHANNEL'], right_on='CHAN_NO', left_on='id')

    def show_map_colored_by_length_matplotlib(self):
        return self.dsm2_chans.plot(figsize=(10, 10), column='length_ft', legend=True)

    def show_map_colored_by_mannings_matplotlib(self):
        return self.dsm2_chans_joined.plot(figsize=(10, 10), column='MANNING', legend=True)

    def show_map_colored_by_column(self, column_name='MANNING'):
        titlestr = column_name
        cmap = cc.b_rainbow_bgyrm_35_85_c71
        if self.hydro_echo_file_base:
            titlestr = titlestr + ' Difference from base'
            cmap = cc.b_diverging_bwr_20_95_c54
            # make diffs range centered on 0 difference
            amin=abs(self.dsm2_chans_joined[column_name].min())
            amax=abs(self.dsm2_chans_joined[column_name].max())
            val = max(amin,amax)
            clim = (-val, val)

        plot = self.dsm2_chans_joined.hvplot(c=column_name, hover_cols=['CHAN_NO', column_name, 'UPNODE', 'DOWNNODE'],
                                             title=titlestr).opts(opts.Polygons(color_index=column_name, colorbar=True, line_alpha=0, cmap=cmap))
        if self.hydro_echo_file_base:
            plot = plot.opts(clim=clim)
        return self.map*plot

    def show_map_colored_by_manning(self):
        return self.show_map_colored_by_column('MANNING')

    def show_map_colored_by_dispersion(self):
        return self.show_map_colored_by_column('DISPERSION')

    def show_map_colored_by_length(self):
        return self.show_map_colored_by_column('LENGTH')


class DSM2GraphNetworkMap(param.Parameterized):
    selected = param.List(
        default=[0], doc='Selected node indices to display in plot')
    date_range = param.DateRange()  # filter by date range
    godin = param.Boolean()  # godin filter and display
    percent_ratios = param.Boolean()  # show percent ratios instead of total flows

    def __init__(self, node_shapefile, hydro_echo_file, **kwargs):
        super().__init__(**kwargs)

        nodes = load_dsm2_node_shapefile(node_shapefile)
        nodes['x'] = nodes.geometry.x
        nodes['y'] = nodes.geometry.y
        node_map = to_node_tuple_map(nodes)

        self.study = DSM2Study(hydro_echo_file)
        stime, etime = self.study.get_runtime()
        # tuple(map(pd.Timestamp,time_window.split('-')))
        self.param.set_param('date_range', (etime - pd.Timedelta('10 days'), etime))
        # self.param.set_default('date_range', (stime, etime)) # need to set bounds

        # should work but doesn't yet
        tiled_network = hv.element.tiles.CartoLight() * hv.Graph.from_networkx(self.study.gc, node_map).opts(
            opts.Graph(directed=True,
                       arrowhead_length=0.001,
                       labelled=['index'],
                       node_alpha=0.5, node_size=10
                       )
        )

        selector = hv.streams.Selection1D(source=tiled_network.Graph.I.nodes)
        selector.add_subscriber(self.set_selected)

        self.nodes = nodes
        self.tiled_network = tiled_network
        # this second part of overlay needed only because of issue.
        # see https://discourse.holoviz.org/t/selection-on-graph-nodes-doesnt-work/3437
        self.map_pane = self.tiled_network*(self.tiled_network.Graph.I.nodes.opts(alpha=0))

    def set_selected(self, index):
        if index is None or len(index) == 0:
            pass  # keep the previous selections
        else:
            self.selected = index

    def display_node_map(self):
        return hv.element.tiles.CartoLight()*self.nodes.hvplot()

    def _date_range_to_twstr(self):
        return '-'.join(map(lambda x: x.strftime("%d%b%Y %H%M"), self.date_range))

    @param.depends('selected', 'date_range', 'percent_ratios')
    def show_sankey(self):
        nodeid = int(self.tiled_network.Graph.I.nodes.data.iloc[self.selected].values[0][2])

        inflows, outflows = self.study.get_inflows_outflows(nodeid, self._date_range_to_twstr())
        mean_inflows = [df.mean() for df in inflows]
        mean_outflows = [df.mean() for df in outflows]
        if self.percent_ratios:
            total_inflows = sum([f.values[0] for f in mean_inflows])
            total_outflows = sum([f.values[0] for f in mean_outflows])
            mean_inflows = [df/total_inflows*100 for df in mean_inflows]
            mean_outflows = [df/total_outflows*100 for df in mean_outflows]
        inlist = [[x.index[0], str(nodeid), x[0]] for x in mean_inflows]
        outlist = [[str(nodeid), x.index[0], x[0]] for x in mean_outflows]
        edges = pd.DataFrame(inlist+outlist, columns=['from', 'to', 'value'])
        sankey = hv.Sankey(edges, label=f'Flows in/out of {nodeid}')
        sankey = sankey.opts(label_position='left', edge_fill_alpha=0.75, edge_fill_color='value',
                             node_alpha=0.5, node_color='index', cmap='blues', colorbar=True)
        return sankey.opts(frame_width=300, frame_height=300)

    @param.depends('selected', 'date_range', 'godin')
    def show_ts(self):
        nodeid = int(self.tiled_network.Graph.I.nodes.data.iloc[self.selected].values[0][2])
        inflows, outflows = self.study.get_inflows_outflows(nodeid, self._date_range_to_twstr())
        if godin:
            inflows = [godin(df) for df in inflows]
            outflows = [godin(df) for df in outflows]
        tsin = [df.hvplot(label=df.columns[0]) for df in inflows]
        tsout = [df.hvplot(label=df.columns[0]) for df in outflows]
        return (hv.Overlay(tsin).opts(title='Inflows')+hv.Overlay(tsout).opts(title='Outflows')).cols(1)

    def get_panel(self):
        slider = pn.Param(self.param.date_range, widgets={
                          'date_range': pn.widgets.DatetimeRangePicker})
        godin_box = pn.Param(self.param.godin, widgets={'godin': pn.widgets.Checkbox})
        percent_ratios_box = pn.Param(self.param.percent_ratios, widgets={
                                      'percent_ratios': pn.widgets.Checkbox})
        self.sankey_pane = pn.Row(self.show_sankey)
        self.ts_pane = pn.Row(self.show_ts)
        return pn.Column(pn.Row(pn.Column(pn.pane.HoloViews(self.map_pane, linked_axes=False),
                                          slider, godin_box, percent_ratios_box), self.sankey_pane), self.ts_pane)
