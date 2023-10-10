from operator import add
import panel as pn
from scipy import stats
import pandas as pd
# display stuff
import hvplot.pandas
import holoviews as hv
from holoviews import opts
# styling plots
from bokeh.themes.theme import Theme
from bokeh.themes import built_in_themes
#
from pydsm.functions import tsmath
from pydsm import postpro
import datetime
import sys
import logging
## - Generic Plotting Functions ##
import pyhecdss


def parse_time_window(timewindow):
    """
    Args:
        timewindow (str, optional): time window for plot. Must be in format: 'YYYY-MM-DD:YYYY-MM-DD'

    Returns:
        list:str containing starting and ending times
    """
    return_list = []
    try:
        parts = timewindow.split(":")
        for p in parts:
            date_parts = [int(i) for i in p.split('-')]
            return_list.append(date_parts)
    except:
        errmsg = 'error in calibplot.parse_time_window, while parsing timewindow. Timewindow must be in format yyyy-mm-dd:yyyy-mm-dd or ' + \
            'yyyy-mm-dd hhmm:yyyy-mm-dd hhmm. Ignoring timewindow'
        print(errmsg)
        logging.error(errmsg)
    return return_list


def tsplot(dflist, names, timewindow=None, zoom_inst_plot=False):
    """Time series overlay plots

    Handles missing DataFrame, just put None in the list

    Args:
        dflist (List): Time-indexed DataFrame list
        names (List): Names list (same size as dflist)
        timewindow (str, optional): time window for plot. Must be in format: 'YYYY-MM-DD:YYYY-MM-DD'
        zoom_inst_plot (bool): if true, display only data in timewindow for plot
    Returns:
        Overlay: Overlay of Curve
    """
    start_dt = None
    end_dt = None
    if dflist[0] is not None:
        start_dt = dflist[0].index.min()
        end_dt = dflist[0].index.max()
    if zoom_inst_plot and (timewindow is not None):
        try:
            parts = timewindow.split(':')
            start_dt = parts[0]
            end_dt = parts[1]
        except:
            errmsg = 'error in calibplot.tsplot'
            print(errmsg)
            logging.error(errmsg)
    # This doesn't work. Need to find a way to get this working.
    # plt = [df[start_dt:end_dt].hvplot(label=name, x_range=(timewindow)) if df is not None else hv.Curve(None, label=name)
    plt = [df[start_dt:end_dt].hvplot(label=name) if df is not None else hv.Curve(None, label=name)
        for df, name in zip(dflist, names)]
    plt = [c.redim(**{c.vdims[0].name:c.label, c.kdims[0].name: 'Time'})
        if c.name != '' else c for c in plt]
    return hv.Overlay(plt)


def scatterplot(dflist, names, index_x=0):
    """Scatter plot overlay with index_x determining the x (independent variable)

    Args:
        dflist (List): DataFrame list
        names (List): Names list
        index_x (int, optional): Index for determining independent variable. Defaults to 0.

    Returns:
        Overlay: Overlay of Scatter
    """
    dfa = pd.concat(dflist, axis=1)
    dfa.columns = names
    dfa = dfa.resample('D').mean()
    return dfa.hvplot.scatter(x=dfa.columns[index_x], hover_cols='all')


def calculate_metrics(dflist, names, index_x=0):
    """Calculate metrics between the index_x column and other columns


    Args:
        dflist (List): DataFrame list
        names (List): Names list
        index_x (int, optional): Index of base DataFrame. Defaults to 0.

    Returns:
        DataFrame: DataFrame of metrics
    """
    dfa = pd.concat(dflist, axis=1)
    dfa = dfa.dropna()
    # x_series contains observed data
    # y_series contains model output for each of the studies
    x_series = dfa.iloc[:, index_x]
    dfr = dfa.drop(columns=dfa.columns[index_x])
    names.remove(names[index_x])
    slopes, interceps, equations, r2s, pvals, stds = [], [], [], [], [], []
    mean_errors, nmean_errors, ses, nmses, mses, nmses, rmses, nrmses, percent_biases, nses, rsrs = [], [], [], [], [], [], [], [], [], [], []

    metrics_calculated = False
    if len(x_series) > 0:
        for col in dfr.columns:
            y_series = dfr.loc[:, col]
            if len(y_series) > 0:
                slope, intercep, rval, pval, std = stats.linregress(x_series, y_series)
                slopes.append(slope)
                interceps.append(intercep)
                sign = '-' if intercep <= 0 else '+'
                equation = 'y=%.2fx%s%.2f' % (slope, sign, abs(intercep))
                equations.append(equation)
                r2s.append(rval*rval)
                pvals.append(pval)
                stds.append(std)
                mean_errors.append(tsmath.mean_error(y_series, x_series))
                nmean_errors.append(tsmath.nmean_error(y_series, x_series))
                mses.append(tsmath.mse(y_series, x_series))
                nmses.append(tsmath.nmse(y_series, x_series))
                rmses.append(tsmath.rmse(y_series, x_series))
                nrmses.append(tsmath.nrmse(y_series, x_series))
                percent_biases.append(tsmath.percent_bias(y_series, x_series))
                nses.append(tsmath.nash_sutcliffe(y_series, x_series))
                rsrs.append(tsmath.rsr(y_series, x_series))
                metrics_calculated = True
            else:
                errmsg = 'calibplot.calculate_metrics: no y_series data found. Metrics can not be calculated.\n'
                print(errmsg)
                logging.info(errmsg)

    else:
        errmsg = 'calibplot.calculate_metrics: no x_series data found. Metrics can not be calculated.\n'
        print(errmsg)
        logging.info(errmsg)
    dfmetrics = None
    if metrics_calculated:
        dfmetrics = pd.concat([pd.DataFrame(arr)
                            for arr in (slopes, interceps, equations, r2s, pvals, stds, mean_errors, nmean_errors, mses, nmses, rmses, nrmses, \
                                percent_biases, nses, rsrs)], axis=1)
        dfmetrics.columns = ['regression_slope', 'regression_intercep', 'regression_equation',
                            'r2', 'pval', 'std', 'mean_error', 'nmean_error', 
                            'mse', 'nmse', 'rmse', 'nrmse', 'percent_bias', 'nash_sutcliffe', 'rsr']

        dfmetrics.index = names
    return dfmetrics


def regression_line_plots(dfmetrics):
    """Create Slope from the metrics DataFrame (calculate_metrics function)

    Args:
        dfmetrics (List): DataFrame list

    Returns:
        tuple: Slope list, equations(str) list
    """
    slope_plots = None
    for i, row in dfmetrics.iterrows():
        slope = row['regression_slope']
        intercep = row['regression_intercep']
        slope_plot = hv.Slope(slope, y_intercept=intercep)
        slope_plots = slope_plot if slope_plots == None else slope_plots*slope_plot
    return slope_plots


def shift_cycle(cycle):
    """Shift cycle to left

    Args:
        cycle (Cycle): Holoview Cycle

    Returns:
        Cycle: Shifted to left with value shifted at right end
    """
    v = cycle.values
    v.append(v.pop(0))
    return hv.Cycle(v)


def tidalplot(df, high, low, name):
    """Tidal plot of series as Curve with high and low as Scatter with markers

    Args:
        df (DataFrame): Tidal signal
        high (DataFrame): Tidal highs
        low (DataFrame): Tidal lows
        name (str): label

    Returns:
        Overlay: Overlay of Curve and 2 Scatter
    """
    h = high.hvplot.scatter(label='high').opts(marker='^')
    l = low.hvplot.scatter(label='low').opts(marker='v')
    o = df.hvplot.line(label=name)
    plts = [h, l, o]
    plts = [c.redim(**{c.vdims[0].name:c.label, c.kdims[0].name: 'Time'}) for c in plts]
    return hv.Overlay(plts)


def kdeplot(dflist, names, xlabel):
    """Kernel Density Estimate (KDE) plots

    Args:
        dflist (List): DataFrame list
        names (List): str list (same size as dflist)
        xlabel (str): label for x axis

    Returns:
        Overlay : Overlay of Distribution
    """
    kdes = [df.hvplot.kde(label=name, xlabel=xlabel) for df, name in zip(dflist, names)]
    return hv.Overlay(kdes)


# - Customized functions for calibration / validation templates
# Needed because of name with . https://github.com/holoviz/holoviews/issues/4714
def sanitize_name(name):
    return name.replace('.', ' ')

class DataMaskingTimeSeries:
    def __init__(self, gate_studies, gate_location, gate_vartype, timewindow):
        #     ----------------------------------------------------------
        # gate_studies, gate_locations,gate_vartype=
        # ----------------------------------------------------------
        # [Study(name='Gate', dssfile='../../../timeseries2019/gates-v8-201912.dss')]
        # [Location(name='DLC', bpart='DLC', description='Delta Cross-Channel Gate')]
        # VarType(name='POS', units='')
        # ----------------------------------------------------------
        self.dssfile = gate_studies[0].dssfile
        self.location = gate_location
        self.bpart = self.location.bpart.upper()
        self.vartype = gate_vartype.name
        self.timewindow = timewindow
        self.gate_time_series_tuple = None
        self.gate_time_series_tuple = next(pyhecdss.get_ts(self.dssfile, '//%s/%s////' % (self.bpart, self.vartype)))
        # try:
        #     # self.gate_time_series_tuple = next(pyhecdss.get_ts(self.dssfile, '//%s/%s/%s///' % (self.bpart, self.vartype, self.timewindow)))
        #     self.gate_time_series_tuple = next(pyhecdss.get_ts(self.dssfile, '//%s/%s////' % (self.bpart, self.vartype)))
        #     print('DataMaskingTimeSeries constructor: type of df='+str(type(self.gate_time_series_tuple)))



        # except StopIteration as e:
        #     print('no data found for ' + self.dssfile + ',//%s/%s/%s///' % (self.bpart, self.vartype, self.timewindow))
        #     logging.exception('pydsm.postpro.PostProCache.load: no data found')
        self.time_series_df = self.get_time_series_df()

    def get_time_series_df(self):
        '''
        for now, assume only one data set is being read
        self.gate_time_series_tuple has 3 elements:
        1. the dataframe
        2. string = 'UNSPECIF' (probably units)
        3. string = 'INST-VAL' (averaging)
        '''
        return_df = None
        for t in self.gate_time_series_tuple:
            if isinstance(t, pd.DataFrame):
                return_df = t
        return return_df






    def get_gate_value(self, location, datetime):
        '''
        returns the value of the gate time series at or before given date.
        reference: https://kanoki.org/2022/02/09/how-to-find-closest-date-in-dataframe-for-a-given-date/
        '''
        gate_value_index = self.gate_time_series_df.get_loc(datetime) - 1
        return self.gate_time_series_df[[gate_value_index]]['POS']







def build_calib_plot_template(studies, location, vartype, timewindow, tidal_template=False, flow_in_thousands=False, units=None,
                              inst_plot_timewindow=None, layout_nash_sutcliffe=False, obs_data_included=True, include_kde_plots=False,
                              zoom_inst_plot=False, gate_studies=None, gate_locations=None, gate_vartype=None):
    """Builds calibration plot template

    Args:
        studies (List): Studies (name,dssfile)
        location (Location): name,bpart,description
        vartype (VarType): name,units
        timewindow (str): timewindow as start_date_str "-" end_date_str or "" for full availability
        tidal_template (bool, optional): If True include tidal plots. Defaults to False.
        flow_in_thousands (bool, optional): If True, template is for flow data, and
            1) y axis title will include the string '(1000 CFS)', and
            2) all flow values in the inst, godin, and scatter plots will be divided by 1000.
        units (str, optional): a string representing the units of the data. examples: CFS, FEET, UMHOS/CM.
            Included in axis titles if specified.
        inst_plot_timewindow (str, optional): Defines a separate timewindow to use for the instantaneous plot.
            Must be in format 'YYYY-MM-DD:YYYY-MM-DD'
        layout_nash_sutcliffe (bool, optional): if true, include Nash-Sutcliffe Efficiency in tables that are
            included in plot layouts. NSE will be included in summary tables--separate files containing only
            the equations and statistics for all locations.
        obs_data_included (bool, optional): If true, first study in studies list is assumed to be observed data.
            calibration metrics will be calculated.
        include_kde_plots (bool): If true, kde plots will be included. This is temporary for debugging
        zoom_inst_plot (bool): If true, instantaneous plots will display on data in the inst_plot_timewindow

    Returns:
        panel: A template ready for rendering by display or save
        dataframe: equations and statistics for all locations
    """
    all_data_found, pp = load_data_for_plotting(studies, location, vartype, timewindow)
    if not all_data_found:
        return None, None

    print('build_calib_plot_template')
    gate_pp = []
    print('----------------------------------------------------------')
    print('gate_studies, gate_locations,gate_vartype=')
    print('----------------------------------------------------------')
    print(str(gate_studies))
    print(str(gate_locations))
    print(str(gate_vartype))
    print('----------------------------------------------------------')

    data_masking_time_series_dict= {}
    data_masking_df_dict = {}
    if gate_studies is not None and gate_locations is not None and gate_vartype is not None:
        for gate_location in gate_locations:
            dmts = DataMaskingTimeSeries(gate_studies, gate_location, gate_vartype, timewindow)
            data_masking_time_series_dict.update({gate_location.name: dmts})
            data_masking_df_dict.update({gate_location.name: dmts.get_time_series_df()})
    else:
        print('Not using gate information for plots/metrics data masking because insufficient information provided.')

    tsp = build_inst_plot(pp, location, vartype, flow_in_thousands=flow_in_thousands, units=units, inst_plot_timewindow=inst_plot_timewindow, zoom_inst_plot=zoom_inst_plot)
    gtsp = build_godin_plot(pp, location, vartype, flow_in_thousands=flow_in_thousands, units=units)
    cplot = None
    dfdisplayed_metrics = None
    metrics_table = None
    kdeplots = None

    if obs_data_included:
        cplot = build_scatter_plots(pp, location, vartype, flow_in_thousands=flow_in_thousands, units=units)

        df_displayed_metrics_dict = {}
        metrics_table_dict = {}
        if gate_studies is not None and gate_locations is not None and gate_vartype is not None:
            dfdisplayed_metrics_open, metrics_table_open = build_metrics_table(studies, pp, location, vartype, tidal_template=tidal_template, \
                flow_in_thousands=flow_in_thousands, units=units, layout_nash_sutcliffe=False, data_masking_df_dict=data_masking_df_dict, gate_open=True)
            dfdisplayed_metrics_closed, metrics_table_closed = build_metrics_table(studies, pp, location, vartype, tidal_template=tidal_template, \
                flow_in_thousands=flow_in_thousands, units=units, layout_nash_sutcliffe=False, data_masking_df_dict=data_masking_df_dict, gate_open=False)
            df_displayed_metrics_dict.update({'open': dfdisplayed_metrics_open})
            df_displayed_metrics_dict.update({'closed': dfdisplayed_metrics_closed})
            metrics_table_dict.update({'open': metrics_table_open})
            metrics_table_dict.update({'closed': metrics_table_closed})
        else:
            dfdisplayed_metrics, metrics_table = build_metrics_table(studies, pp, location, vartype, tidal_template=tidal_template, flow_in_thousands=flow_in_thousands, units=units,
                                layout_nash_sutcliffe=False)
            df_displayed_metrics_dict.update({'all': dfdisplayed_metrics})
            metrics_table_dict.update({'all': metrics_table})

        if include_kde_plots: 
            kdeplots = build_kde_plots(pp)
    
    # # create plot/metrics template
    header_panel = pn.panel(f'## {location.description} ({location.name}/{vartype.name})')
    # # do this if you want to link the axes
    # # tsplots2 = (tsp.opts(width=900)+gtsp.opts(show_legend=False, width=900)).cols(1)
    # # start_dt = dflist[0].index.min()
    # # end_dt = dflist[0].index.max()

    column = None
    # temporary fix to add toolbar to all plots. eventually need to only inlucde toolbar if creating html file
    add_toolbar = True
    print('before creating column object (plot layout) for returning')
    if tidal_template:
        if not add_toolbar:
            if obs_data_included:
                if include_kde_plots:
                    column = pn.Column(
                        header_panel,
                        # tsp.opts(width=900, legend_position='right'),
                        tsp.opts(width=900, toolbar=None, title='(a)', legend_position='right'),
                        gtsp.opts(width=900, toolbar=None, title='(b)', legend_position='right'))
                        # pn.Row(tsplots2),
                        # pn.Row(cplot.opts(shared_axes=False, toolbar=None, title='(c)')))
                    metrics_table_column = pn.Column()
                    for metrics_table_name in metrics_table_dict:
                        metrics_table_column.append(metrics_table_dict[metrics_table_name].opts(title='(d) ' + metrics_table_name))
                    scatter_and_metrics_row = pn.Row(cplot.opts(shared_axes=False, toolbar=None, title='(c)'))
                    scatter_and_metrics_row.append(metrics_table_column)
                    column.append(scatter_and_metrics_row)
                    column.append(pn.Row(kdeplots))
                else:
                    column = pn.Column(
                        header_panel,
                        # tsp.opts(width=900, legend_position='right'),
                        tsp.opts(width=900, toolbar=None, title='(a)', legend_position='right'),
                        gtsp.opts(width=900, toolbar=None, title='(b)', legend_position='right'))
                        # pn.Row(tsplots2),
                        # pn.Row(cplot.opts(shared_axes=False, toolbar=None, title='(c)')))
                    metrics_table_column = pn.Column()
                    for metrics_table_name in metrics_table_dict:
                        metrics_table_column.append(metrics_table_dict[metrics_table_name].opts(title='(d) ' + metrics_table_name))

                    scatter_and_metrics_row = pn.Row(cplot.opts(shared_axes=False, toolbar=None, title='(c)'))
                    scatter_and_metrics_row.append(metrics_table_column)
                    column.append(scatter_and_metrics_row)
            else:
                column = pn.Column(
                    header_panel,
                    # tsp.opts(width=900, legend_position='right'),
                    tsp.opts(width=900, toolbar=None, title='(a)', legend_position='right'),
                    gtsp.opts(width=900, toolbar=None, title='(b)', legend_position='right'))
        else:
            if obs_data_included:
                if include_kde_plots:
                    column = pn.Column(
                        header_panel,
                        # tsp.opts(width=900, legend_position='right'),
                        tsp.opts(width=900, title='(a)', legend_position='right'),
                        gtsp.opts(width=900, title='(b)', legend_position='right'))
                        # pn.Row(tsplots2),
                        # pn.Row(cplot.opts(shared_axes=False, title='(c)')))
                    metrics_table_column = pn.Column()
                    for metrics_table_name in metrics_table_dict:
                        metrics_table_column.append(metrics_table_dict[metrics_table_name].opts(title='(d) ' + metrics_table_name))
                    scatter_and_metrics_row = pn.Row(cplot.opts(shared_axes=False, title='(c)'))
                    scatter_and_metrics_row.append(metrics_table_column)
                    column.append(scatter_and_metrics_row)
                    column.append(pn.Row(kdeplots))
                else:
                    column = pn.Column(
                        header_panel,
                        # tsp.opts(width=900, legend_position='right'),
                        tsp.opts(width=900, title='(a)', legend_position='right'),
                        gtsp.opts(width=900, title='(b)', legend_position='right'))
                        # pn.Row(tsplots2),
                        # pn.Row(cplot.opts(shared_axes=False, title='(c)')))
                    metrics_table_column = pn.Column()
                    for metrics_table_name in metrics_table_dict:
                        metrics_table_column.append(metrics_table_dict[metrics_table_name].opts(title='(d) ' + metrics_table_name))
                    scatter_and_metrics_row = pn.Row(cplot.opts(shared_axes=False, title='(c)'))
                    scatter_and_metrics_row.append(metrics_table_column)
                    column.append(scatter_and_metrics_row)
            else:
                column = pn.Column(
                    header_panel,
                    # tsp.opts(width=900, legend_position='right'),
                    tsp.opts(width=900, title='(a)', legend_position='right'),
                    gtsp.opts(width=900, title='(b)', legend_position='right'))
    else:
        if not add_toolbar:
            if obs_data_included:
                column = pn.Column(
                    header_panel,
                    pn.Row(gtsp.opts(width=900, show_legend=True, toolbar=None, title='(a)', legend_position='right')))
                    # pn.Row(cplot.opts(shared_axes=False, toolbar=None, title='(b)')))
                metrics_table_column = pn.Column()
                for metrics_table_name in metrics_table_dict:
                    metrics_table_column.append(metrics_table_dict[metrics_table_name].opts(title='(c) ' + metrics_table_name))
                scatter_and_metrics_row = pn.Row(cplot.opts(shared_axes=False, toolbar=None, title='(b)'))
                scatter_and_metrics_row.append(metrics_table_column)
                column.append(scatter_and_metrics_row)
            else:
                column = pn.Column(
                    header_panel,
                    pn.Row(gtsp.opts(width=900, show_legend=True, toolbar=None, title='(a)', legend_position='right')))

        else:
            if obs_data_included:
                column = pn.Column(
                    header_panel,
                    pn.Row(gtsp.opts(width=900, show_legend=True, title='(a)')))
                    # pn.Row(cplot.opts(shared_axes=False, title='(b)')))
                metrics_table_column = pn.Column()
                for metrics_table_name in metrics_table_dict:
                    metrics_table_column.append(metrics_table_dict[metrics_table_name].opts(title='(c) ' + metrics_table_name))
                scatter_and_metrics_row = pn.Row(cplot.opts(shared_axes=False, title='(b)'))
                scatter_and_metrics_row.append(metrics_table_column)
                column.append(scatter_and_metrics_row)
            else:
                column = pn.Column(
                    header_panel,
                    pn.Row(gtsp.opts(width=900, show_legend=True, title='(a)')))

    # now merge all metrics dataframes, adding a column identifying the gate status
    return_metrics_df = None
    df_index = 0
    for metrics_df_name in df_displayed_metrics_dict:
        metrics_df_name_list = []
        metrics_df = df_displayed_metrics_dict[metrics_df_name]
        for r in range(metrics_df.shape[0]):
            metrics_df_name_list.append(metrics_df_name)
        metrics_df['Gate Pos'] = metrics_df_name_list
        # move Gate Pos column to beginning
        cols = list(metrics_df)
        cols.insert(0, cols.pop(cols.index('Gate Pos')))
        metrics_df = metrics_df.loc[:, cols]
        # merge df into return_metrics_df
        if df_index == 0:
            return_metrics_df = metrics_df
        else:
            return_metrics_df.append(metrics_df)
        df_index += 1
    return column, return_metrics_df


def load_data_for_plotting(studies, location, vartype, timewindow):
    """Loads data used for creating plots and metrics
    """
    # pp = [postpro.PostProcessor(study, location, vartype) for study in studies]
    pp = []
    all_data_found = True
    for study in studies:
        p = postpro.PostProcessor(study, location, vartype)
        pp.append(p)
    # this was commented out before
    # for p in pp:ed
        success = p.load_processed(timewindow=timewindow)
        if not success:
            errmsg = 'unable to load data for study|location %s|%s' % (str(study), str(location))
            print(errmsg)
            logging.info(errmsg)
            all_data_found = False
    if not all_data_found:
        errmsg = 'Not creating plots because data not found for location, vartype, timewindow = ' + str(location) +','+ str(vartype)+','+str(timewindow)+'\n'
        print(errmsg)
        logging.info(errmsg)
        return None, None
    return all_data_found, pp


def get_units(flow_in_thousands=False, units=None):
    """ create axis titles with units (if specified), and modify titles and data if displaying flow data in 1000 CFS 
    """
    unit_string = ''
    if flow_in_thousands and units is not None:
        unit_string = '(1000 %s)' % units
    elif units is not None:
        unit_string = '(%s)' % units
    return unit_string


def build_inst_plot(pp, location, vartype, flow_in_thousands=False, units=None, inst_plot_timewindow=None, zoom_inst_plot=False):
    """Builds calibration plot template

    Args:
        pp (List): postpro.PostProcessor objects created for each study
        location (Location): name,bpart,description
        vartype (VarType): name,units
        flow_in_thousands (bool, optional): If True, template is for flow data, and
            1) y axis title will include the string '(1000 CFS)', and
            2) all flow values in the inst, godin, and scatter plots will be divided by 1000.
        units (str, optional): a string representing the units of the data. examples: CFS, FEET, UMHOS/CM.
            Included in axis titles if specified.
        inst_plot_timewindow (str, optional): Defines a separate timewindow to use for the instantaneous plot.
            Must be in format 'YYYY-MM-DD:YYYY-MM-DD'

    Returns:
        tsp: A plot
    """
    gridstyle = {'grid_line_alpha': 1, 'grid_line_color': 'lightgrey'}
    unit_string = get_units(flow_in_thousands, units)
    y_axis_label = f'{vartype.name} @ {location.name} {unit_string}'
    # plot_data are scaled, if flow_in_thousands == True
    tsp_plot_data = [p.df for p in pp]

    if flow_in_thousands:
        tsp_plot_data = [p.df/1000.0 if p.df is not None else None for p in pp]
    # create plots: instantaneous, godin, and scatter
    tsp = tsplot(tsp_plot_data, [p.study.name for p in pp], timewindow=inst_plot_timewindow, zoom_inst_plot=zoom_inst_plot).opts(
        ylabel=y_axis_label, show_grid=True, gridstyle=gridstyle, shared_axes=False)
    tsp = tsp.opts(opts.Curve(color=hv.Cycle('Category10')))
    return tsp

def build_godin_plot(pp, location, vartype, flow_in_thousands=False, units=None):
    """Builds calibration plot template

    Args:
        pp (List): postpro.PostProcessor objects created for each study
        location (Location): name,bpart,description
        vartype (VarType): name,units
        flow_in_thousands (bool, optional): If True, template is for flow data, and
            1) y axis title will include the string '(1000 CFS)', and
            2) all flow values in the inst, godin, and scatter plots will be divided by 1000.
        units (str, optional): a string representing the units of the data. examples: CFS, FEET, UMHOS/CM.
            Included in axis titles if specified.

    Returns:
        gtsp: A plot
    """
    gridstyle = {'grid_line_alpha': 1, 'grid_line_color': 'lightgrey'}
    unit_string = get_units(flow_in_thousands, units)
    y_axis_label = f'{vartype.name} @ {location.name} {unit_string}'
    godin_y_axis_label = 'Godin '+y_axis_label
    # plot_data are scaled, if flow_in_thousands == True
    gtsp_plot_data = [p.gdf for p in pp]

    # if p.gdf is not None:
    if flow_in_thousands:
        gtsp_plot_data = [p.gdf/1000.0 if p.gdf is not None else None for p in pp]
    # zoom in to desired timewindow: works, but doesn't zoom y axis, so need to fix later
    # if inst_plot_timewindow is not None:
    #     start_end_times = parse_time_window(inst_plot_timewindow)
    #     s = start_end_times[0]
    #     e = start_end_times[1]
    #     tsp.opts(xlim=(datetime.datetime(s[0], s[1], s[1]), datetime.datetime(e[0],e[1],e[2])))
    gtsp = tsplot(gtsp_plot_data, [p.study.name for p in pp]).opts(
        ylabel=godin_y_axis_label, show_grid=True, gridstyle=gridstyle)
    gtsp = gtsp.opts(opts.Curve(color=hv.Cycle('Category10')))
    return gtsp

def build_scatter_plots(pp, location, vartype, flow_in_thousands=False, units=None, gate_pp=None):
# def build_scatter_plots(pp, location, vartype, flow_in_thousands=False, units=None):
    """Builds calibration plot template

    Args:
        pp (List): postpro.PostProcessor objects created for each study
        location (Location): name,bpart,description
        vartype (VarType): name,units
        flow_in_thousands (bool, optional): If True, template is for flow data, and
            1) y axis title will include the string '(1000 CFS)', and
            2) all flow values in the inst, godin, and scatter plots will be divided by 1000.
        units (str, optional): a string representing the units of the data. examples: CFS, FEET, UMHOS/CM.
            Included in axis titles if specified.

    Returns:
        a plot object
    """
    gridstyle = {'grid_line_alpha': 1, 'grid_line_color': 'lightgrey'}
    unit_string = get_units(flow_in_thousands, units)

    y_axis_label = f'{vartype.name} @ {location.name} {unit_string}'
    godin_y_axis_label = 'Godin '+y_axis_label
    # plot_data are scaled, if flow_in_thousands == True
    gtsp_plot_data = [p.gdf for p in pp]

    splot_plot_data = None
    splot_metrics_data = None

    splot_plot_data = [p.gdf.resample('D').mean() if p.gdf is not None else None for p in pp ]
    splot_metrics_data = [p.gdf.resample('D').mean() if p.gdf is not None else None for p in pp]
    if flow_in_thousands:
        gtsp_plot_data = [p.gdf/1000.0 if p.gdf is not None else None for p in pp]
        splot_plot_data = [p.gdf.resample('D').mean()/1000.0 if p.gdf is not None else None for p in pp]

    splot = None
    if splot_plot_data is not None and splot_plot_data[0] is not None:
        splot = scatterplot(splot_plot_data, [p.study.name for p in pp])\
            .opts(opts.Scatter(color=shift_cycle(hv.Cycle('Category10'))))\
            .opts(ylabel='Model', legend_position="top_left")\
            .opts(show_grid=True, frame_height=250, frame_width=250, data_aspect=1)

    cplot = None
    dfdisplayed_metrics = None
    # calculate calibration metrics
    slope_plots_dfmetrics = None
    if gtsp_plot_data is not None and gtsp_plot_data[0] is not None:
        slope_plots_dfmetrics = calculate_metrics(gtsp_plot_data, [p.study.name for p in pp])
    # dfmetrics = calculate_metrics([p.gdf for p in pp], [p.study.name for p in pp])
    dfmetrics = None
    if splot_metrics_data is not None:
        dfmetrics = calculate_metrics(splot_metrics_data, [p.study.name for p in pp])
    dfmetrics_monthly = None
    # if p.gdf is not None:
    dfmetrics_monthly = calculate_metrics(
        [p.gdf.resample('M').mean() if p.gdf is not None else None for p in pp], [p.study.name for p in pp])

    # add regression lines to scatter plot, and set x and y axis titles
    slope_plots = None
    if slope_plots_dfmetrics is not None:
        slope_plots = regression_line_plots(slope_plots_dfmetrics)
        cplot = slope_plots.opts(opts.Slope(color=shift_cycle(hv.Cycle('Category10'))))*splot
        cplot = cplot.opts(xlabel='Observed ' + unit_string, ylabel='Model ' + unit_string, legend_position="top_left")\
            .opts(show_grid=True, frame_height=250, frame_width=250, data_aspect=1, show_legend=False)
    return cplot

def build_metrics_table(studies, pp, location, vartype, tidal_template=False, flow_in_thousands=False, units=None,
                              layout_nash_sutcliffe=False, gate_pp=None, data_masking_df_dict=None, gate_open=True):
    """Builds calibration plot template

    Args:
        studies (List): Studies (name,dssfile)
        pp (List): postpro.PostProcessor objects created for each study
        location (Location): name,bpart,description
        vartype (VarType): name,units
        tidal_template (bool, optional): If True include tidal plots. Defaults to False.
        flow_in_thousands (bool, optional): If True, template is for flow data, and
            1) y axis title will include the string '(1000 CFS)', and
            2) all flow values in the inst, godin, and scatter plots will be divided by 1000.
        units (str, optional): a string representing the units of the data. examples: CFS, FEET, UMHOS/CM.
            Included in axis titles if specified.
        layout_nash_sutcliffe (bool, optional): if true, include Nash-Sutcliffe Efficiency in tables that are
            included in plot layouts. NSE will be included in summary tables--separate files containing only
            the equations and statistics for all locations.
        gate_dss_file (str): path to DSS file with gate data, to be used for creating metrics tables separately 
            for gate open/closed conditions. This will only be done for a given location if a DSS path is
            specified in the location file in the gate_time_series field.
        data_masking_df_dict (dict of df): contains gate time series used for data masking.
        gate_open (bool): if true, calculate metrics for gate open condition (gate pos > 0) only.

    Returns:
        a list containing one or more table object(s). Will contain more then one object if a DSS path is specified
            in the location file in the gate_time_series field.
    """
    gridstyle = {'grid_line_alpha': 1, 'grid_line_color': 'lightgrey'}
    unit_string = get_units(flow_in_thousands, units)
    y_axis_label = f'{vartype.name} @ {location.name} {unit_string}'
    godin_y_axis_label = 'Godin '+y_axis_label
    # plot_data are scaled, if flow_in_thousands == True
    gtsp_plot_data = [p.gdf for p in pp]

    splot_metrics_data = None
    # if p.gdf is not None:
    splot_metrics_data = [p.gdf.resample('D').mean() if p.gdf is not None else None for p in pp]
    if flow_in_thousands:
        # if p.gdf is not None:
        gtsp_plot_data = [p.gdf/1000.0 if p.gdf is not None else None for p in pp]




    # use data_masking_df_dict to mask data (remove rows if gate open/closed)
    # for location in data_masking_df_dict:
        









    dfdisplayed_metrics = None
    column = None
    # calculate calibration metrics
    slope_plots_dfmetrics = None
    if gtsp_plot_data is not None and gtsp_plot_data[0] is not None:
        slope_plots_dfmetrics = calculate_metrics(gtsp_plot_data, [p.study.name for p in pp])
    # dfmetrics = calculate_metrics([p.gdf for p in pp], [p.study.name for p in pp])
    dfmetrics = None
    if splot_metrics_data is not None:
        dfmetrics = calculate_metrics(splot_metrics_data, [p.study.name for p in pp])
    dfmetrics_monthly = None
    # if p.gdf is not None:
    dfmetrics_monthly = calculate_metrics(
        [p.gdf.resample('M').mean() if p.gdf is not None else None for p in pp], [p.study.name for p in pp])

    # display calibration metrics
    # create a list containing study names, excluding observed.
    dfdisplayed_metrics = None
    study_list = [study.name.replace('DSM2', '')
                for study in studies if study.name.lower() != 'observed']

    # calculate amp diff, amp % diff, and phase diff
    amp_avg_pct_errors = []
    amp_avg_phase_errors = []
    for p in pp[1:]:  # TODO: move this out of here. Nothing to do with plotting!
        p.process_diff(pp[0])
        amp_avg_pct_errors.append(float(p.amp_diff_pct.mean(axis=0)))
        amp_avg_phase_errors.append(float(p.phase_diff.mean(axis=0)))

    # using a Table object because the dataframe object, when added to a layout, doesn't always display all the values.
    # This could have something to do with inconsistent types.
    metrics_table = None
    if tidal_template:
        dfdisplayed_metrics = dfmetrics.loc[:, [
            'regression_equation', 'r2', 'nmean_error', 'nmse', 'nrmse', 'nash_sutcliffe', 'percent_bias', 'rsr']]
        dfdisplayed_metrics['Amp Avg pct Err'] = amp_avg_pct_errors
        dfdisplayed_metrics['Avg Phase Err'] = amp_avg_phase_errors

        dfdisplayed_metrics.index.name = 'DSM2 Run'
        dfdisplayed_metrics.columns = ['Equation', 'R Squared',
                                    'N Mean Error', 'NMSE', 'NRMSE', 'NSE', 'PBIAS', 'RSR', 'Amp Avg %Err', 'Avg Phase Err']
        a = dfdisplayed_metrics['Equation'].to_list()
        b = ['{:.2f}'.format(item) for item in dfdisplayed_metrics['R Squared'].to_list()]
        c = ['{:.2f}'.format(item) for item in dfdisplayed_metrics['N Mean Error'].to_list()]
        d = ['{:.2f}'.format(item) for item in dfdisplayed_metrics['NMSE'].to_list()]
        e = ['{:.2f}'.format(item) for item in dfdisplayed_metrics['NRMSE'].to_list()]
        f = ['{:.2f}'.format(item) for item in dfdisplayed_metrics['NSE'].to_list()]
        g = ['{:.2f}'.format(item) for item in dfdisplayed_metrics['PBIAS'].to_list()]
        h = ['{:.2f}'.format(item) for item in dfdisplayed_metrics['RSR'].to_list()]
        i = ['{:.2f}'.format(item) for item in dfdisplayed_metrics['Amp Avg %Err'].to_list()]
        j = ['{:.2f}'.format(item) for item in dfdisplayed_metrics['Avg Phase Err'].to_list()]
        if layout_nash_sutcliffe:
            metrics_table = hv.Table((study_list, a, b, c, d, e, f, g, h, i), [
                                    'Study', 'Equation', 'R Squared', 'N Mean Error', 'NMSE', 'NRMSE', 'NSE', 'PBIAS', 'RSR', \
                                        'Amp Avg %Err', 'Avg Phase Err']).opts(width=580, fontscale=.8)
        else:
            metrics_table = hv.Table((study_list, a, b, c, d, e, g, h, i), [
                                    'Study', 'Equation', 'R Squared', 'N Mean Error', 'NMSE', 'NRMSE', 'PBIAS', 'RSR', \
                                        'Amp Avg %Err', 'Avg Phase Err']).opts(width=580, fontscale=.8)
    else:
        # template for nontidal (EC) data
        dfdisplayed_metrics = dfmetrics.loc[:, [
            'regression_equation', 'r2', 'nmean_error', 'nmse', 'nrmse', 'nash_sutcliffe', 'percent_bias', 'rsr']]
        dfdisplayed_metrics = pd.concat(
            [dfdisplayed_metrics, dfmetrics_monthly.loc[:, ['nmean_error', 'nrmse']]], axis=1)
        dfdisplayed_metrics.index.name = 'DSM2 Run'
        dfdisplayed_metrics.columns = ['Equation', 'R Squared',
                                    'NMean Error', 'NMSE', 'NRMSE', 'NSE', 'PBIAS', 'RSR', 'Mnly Mean Err', 'Mnly RMSE']
        format_dict = {'Equation': '{:,.2f}', 'R Squared': '{:,.2f}', 'NMean Error': '{:,.2f}', 'NMSE': '{:,.2}', 'NRMSE': '{:,.2}',
                    'Amp Avg %Err': '{:,.2f}', 'Avg Phase Err': '{:,.2f}', 'NSE': '{:,.2f}', 'PBIAS': '{:,.2f}', 'RSR': '{:,.2f}'}
        dfdisplayed_metrics.style.format(format_dict)
        # Ideally, the columns should be sized to fit the data. This doesn't work properly--replaces some values with blanks
        # metrics_table = pn.widgets.DataFrame(dfdisplayed_metrics, autosize_mode='fit_columns')
        a = dfdisplayed_metrics['Equation'].to_list()
        b = ['{:.2f}'.format(item) for item in dfdisplayed_metrics['R Squared'].to_list()]
        c = ['{:.2f}'.format(item) for item in dfdisplayed_metrics['NMean Error'].to_list()]
        d = ['{:.2E}'.format(item) for item in dfdisplayed_metrics['NMSE'].to_list()]
        e = ['{:.2E}'.format(item) for item in dfdisplayed_metrics['NRMSE'].to_list()]
        f = ['{:.2E}'.format(item) for item in dfdisplayed_metrics['NSE'].to_list()]
        g = ['{:.2E}'.format(item) for item in dfdisplayed_metrics['PBIAS'].to_list()]
        h = ['{:.2E}'.format(item) for item in dfdisplayed_metrics['RSR'].to_list()]
        i = ['{:.2f}'.format(item) for item in dfdisplayed_metrics['Mnly Mean Err'].to_list()]
        j = ['{:.2f}'.format(item) for item in dfdisplayed_metrics['Mnly RMSE'].to_list()]
        if layout_nash_sutcliffe:
            metrics_table = hv.Table((study_list, a, b, c, d, e, f, g, h, i, j), [
                                    'Study', 'Equation', 'R Squared', 'NMean Error', 'NMSE', 'NRMSE', 'NSE', 'PBIAS', 'RSR', \
                                        'Mnly Mean Err', 'Mnly RMSE']).opts(width=580, fontscale=.8)
        else:
            metrics_table = hv.Table((study_list, a, b, c, d, e, g, h, i, j), [
                                    'Study', 'Equation', 'R Squared', 'NMean Error', 'NMSE', 'NRMSE', 'PBIAS', 'RSR', \
                                        'Mnly Mean Err', 'Mnly RMSE']).opts(width=580, fontscale=.8)
    return dfdisplayed_metrics, metrics_table

def build_kde_plots(pp, amp_title='(e)', phase_title='(f)'):
    """Builds calibration plot template

    Args:
        pp (List): postpro.PostProcessor objects created for each study
        location (Location): name,bpart,description
        vartype (VarType): name,units
        timewindow (str): timewindow as start_date_str "-" end_date_str or "" for full availability
        flow_in_thousands (bool, optional): If True, template is for flow data, and
            1) y axis title will include the string '(1000 CFS)', and
            2) all flow values in the inst, godin, and scatter plots will be divided by 1000.
        units (str, optional): a string representing the units of the data. examples: CFS, FEET, UMHOS/CM.
            Included in axis titles if specified.

    Returns:
        a plot object
    """
    # plot_data are scaled, if flow_in_thousands == True
    # calculate amp diff, amp % diff, and phase diff
    amp_avg_pct_errors = []
    amp_avg_phase_errors = []
    for p in pp[1:]:  # TODO: move this out of here. Nothing to do with plotting!
        p.process_diff(pp[0])
        amp_avg_pct_errors.append(float(p.amp_diff_pct.mean(axis=0)))
        amp_avg_phase_errors.append(float(p.phase_diff.mean(axis=0)))

    # create kernel density estimate plots
    # We're currently not including the amplitude diff plot
    # amp_diff_kde = kdeplot([p.amp_diff for p in pp[1:]], [
    #     p.study.name for p in pp[1:]], 'Amplitude Diff')
    # amp_diff_kde = amp_diff_kde.opts(opts.Distribution(
    #     color=shift_cycle(hv.Cycle('Category10'))))

    amp_pdiff_kde = kdeplot([p.amp_diff_pct for p in pp[1:]], [
        p.study.name for p in pp[1:]], 'Amplitude Diff (%)')
    amp_pdiff_kde = amp_pdiff_kde.opts(opts.Distribution(
        line_color=shift_cycle(hv.Cycle('Category10')), filled=False))
    amp_pdiff_kde.opts(opts.Distribution(line_width=5))

    phase_diff_kde = kdeplot([p.phase_diff for p in pp[1:]], [
        p.study.name for p in pp[1:]], 'Phase Diff (minutes)')
    phase_diff_kde = phase_diff_kde.opts(opts.Distribution(
        line_color=shift_cycle(hv.Cycle('Category10')), filled=False))
    phase_diff_kde.opts(opts.Distribution(line_width=5))
    # create panel containing 3 kernel density estimate plots. We currently only want the last two, so commenting this out for now.
    # amp diff, amp % diff, phase diff
    # kdeplots = amp_diff_kde.opts(
    #     show_legend=False)+amp_pdiff_kde.opts(show_legend=False)+phase_diff_kde.opts(show_legend=False)
    # kdeplots = kdeplots.cols(3).opts(shared_axes=False).opts(
    #     opts.Distribution(height=200, width=300))
    # don't use

    # create panel containing amp % diff and phase diff kernel density estimate plots. Excluding amp diff plot
    kdeplots = amp_pdiff_kde.opts(show_legend=False, title=amp_title) + \
        phase_diff_kde.opts(show_legend=False, title=phase_title)
    kdeplots = kdeplots.cols(2).opts(shared_axes=False).opts(
        opts.Distribution(height=200, width=300))
    return kdeplots


def export_svg(plot, fname):
    ''' export holoview object to filename fname '''
    from bokeh.io import export_svgs
    p = hv.render(plot, backend='bokeh')
    p.output_backend = "svg"
    export_svgs(p, filename=fname)
