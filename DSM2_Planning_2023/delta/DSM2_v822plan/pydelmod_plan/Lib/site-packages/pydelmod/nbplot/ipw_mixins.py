# -*- coding: utf-8 -*-
""" Common mixins for interactive controls
"""

import pandas as pd
import ipywidgets as ipw
import plotly.io as pio
import qgrid
import plotly.graph_objs as go


MONTHS = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5,
          'JUN': 6, 'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
YEARTYPES = ['W', 'AN', 'BN', 'D', 'C']


class FilterVariableMixin():
    """ A mixin dropdown ipywidget to filter variable.
    """

    def __init__(self, *args, **kwargs):
        """ Constructor
        """
        self.variables = None
        self.colname_variable = None
        super().__init__(*args, **kwargs)

    def preprocess_data(self):
        """ Process data.
            Collect available variables in the DataFrame, self.df.
            The column name for variables is from self.colname_variable
        """
        self.variables = self.df[self.colname_variable].unique()
        super().preprocess_data()

    def create_widgets(self):
        """ Create a dropdown widget.
        """
        self.dd_variable = ipw.Dropdown(options=self.variables,
                                        value=self.variables[0],
                                        description='Variable',
                                        layout={'width': 'auto'})
        self.variable_selected = self.dd_variable.value
        self.dd_variable.observe(
            self.response_filter_variable, names='value')
        super().create_widgets()

    def response_filter_variable(self, change):
        """ A response function to the event from the dropdown
        """
        self.variable_selected = self.dd_variable.value
        self.filter_data()
        self.update()

    def filter_data(self):
        if self.variable_selected is not None:
            self.mask = (
                self.mask & (self.df[self.colname_variable] == self.variable_selected))
        super().filter_data()

    def update(self):
        """ Update the figure and others
        """
        super().update()


class FilterStationMixin():
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_widgets(self):
        self.dd_station = ipw.Dropdown(options=self.df_stations['Location'],
                                       value=self.df_stations['Location'].values[0],
                                       description='Station',
                                       layout={'width': 'initial'})
        self.station_selected = self.dd_station.value
        self.station_id_selected = self.df_stations[
            self.df_stations['Location'] == self.station_selected]['ID'].values[0]
        self.dd_station.observe(
            self.response_filter_station, names='value')
        super().create_widgets()

    def response_filter_station(self, change):
        self.station_selected = self.dd_station.value
        self.station_id_selected = self.df_stations[
            self.df_stations['Location'] == self.station_selected]['ID'].values[0]
        self.filter_data()
        self.update()

    def filter_data(self):
        if self.colname_station_id is not None:
            self.mask = (
                self.mask & (self.df[self.colname_station_id] == self.station_id_selected))
        super().filter_data()


class FilterWateryearTypeMixin():
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_widgets(self):
        self.tb_yeartypes = [ipw.ToggleButton(value=True, description=f'{t}', layout={
            'width': '50px'}) for t in YEARTYPES]
        self.box_yeartypes = ipw.HBox(
            [ipw.Label('Wateryear Type')] + self.tb_yeartypes)
        for tb in self.tb_yeartypes:
            tb.observe(self.response_filter_yeartype, names='value')
        self.yt_selected = [YEARTYPES[i]
                            for i, tb in enumerate(self.tb_yeartypes) if tb.value]
        super().create_widgets()

    def response_filter_yeartype(self, change):
        self.yt_selected = [YEARTYPES[i]
                            for i, tb in enumerate(self.tb_yeartypes) if tb.value]
        self.filter_data()
        self.update()

    def filter_data(self):
        mask = self.df['sac_yrtype'].isin(self.yt_selected)
        self.mask = self.mask & mask
        super().filter_data()


class FilterMonthMixin():
    """ A mixin 12 toggle buttons to filter by months.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_widgets(self):
        self.tb_months = [ipw.ToggleButton(
            value=True, description='{}'.format(t), layout=ipw.Layout(width='50px')) for t in MONTHS]
        box_layout = ipw.Layout(flex='auto', width='auto')
        self.box_months = ipw.HBox([ipw.Label('Month')] + self.tb_months)
        self.mo_selected = [MONTHS[tb.description]
                            for i, tb in enumerate(self.tb_months) if tb.value]
        for tb in self.tb_months:
            tb.observe(self.response_filter_month, names='value')
        super().create_widgets()

    def response_filter_month(self, change):
        self.mo_selected = [MONTHS[tb.description]
                            for i, tb in enumerate(self.tb_months) if tb.value]
        self.filter_data()
        self.update()

    def filter_data(self):
        self.mask = (self.mask & self.df['month'].isin(self.mo_selected))
        super().filter_data()


class ShowDataMixin():
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # forwards all unused arguments

    def create_widgets(self):
        # Create a toggle button
        self.tb_showdata = ipw.ToggleButton(
            value=False, description='Show Data')
        # self.widgets.children += (self.tb_showdata, )
        self.qgrid = None
        # Link observer
        self.tb_showdata.observe(self.response_showdata, names='value')
        super().create_widgets()

    def response_showdata(self, change):
        # Show and hide a QGrid widget and update data linked to it
        # It is assumed that QGrid widget would be at the end of the VBox
        if self.tb_showdata.value:
            if self.qgrid is None:
                self.qgrid = qgrid.show_grid(self.df_to_plot)
            else:
                self.qgrid.df = self.df_to_plot
            if not isinstance(self.widgets.children[-1], type(self.qgrid)):
                self.widgets.children += (self.qgrid, )
        else:
            if isinstance(self.widgets.children[-1], type(self.qgrid)):
                self.widgets.children = self.widgets.children[:-1]

    def update(self):
        if self.tb_showdata.value:
            self.qgrid.df = self.df_to_plot
        super().update()


class SaveDataMixin():
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_widgets(self):
        self.tb_savedata = ipw.ToggleButton(
            value=False, description='Save data',
            # layout=Layout(width='100px')
        )
        self.tb_savedata.observe(self.response_savedata, names='value')
        # self.box_savedata = ipw.HBox((self.tb_savedata, self.lb_msg))
        super().create_widgets()

    def response_savedata(self, change):
        fpath_csv = 'export.csv'
        self.lb_msg.value = 'Saving the data into {}'.format(fpath_csv)
        self.df_to_plot.to_csv(fpath_csv)
        self.lb_msg.value = 'Saved the data into {}'.format(fpath_csv)
        self.tb_savedata.value = False


class ExportPlotForStationsMixin():
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_widgets(self):
        # create the 'Export Plots' button
        self.tb_exportplots = ipw.ToggleButton(
            value=False, description='Export Plots',
            # layout=Layout(width='100px')
        )

        # allow user to specify a plot prefix, which is a string that will be used for filenames
        self.t_exportplots_prefix = ipw.Text(
            value='plot',
            placeholder='Prefix for plots',
            description='Plot prefix:',
            disabled=False)
        self.lb_msg = ipw.Label(value='')
        self.tb_exportplots.observe(self.response_exportplots, names='value')
        self.box_exportplots = ipw.HBox(
            (self.tb_exportplots, self.t_exportplots_prefix))
        super().create_widgets()

    def response_exportplots(self, change):
        self.lb_msg.value = f'Exporting plots...'
        self.export_plots()
        self.tb_exportplots.value = False

    def export_plots(self):
        table_plots = {'station': [], 'filename': [], 'station_long_name': []}
        plot_prefix = self.t_exportplots_prefix.value
        for i, row in self.df_stations.iterrows():
            station_name = row['Location']
            self.dd_station.value = station_name
            station_id = row['ID']
            self.update()
            fpath_plot = f'{plot_prefix}_{station_id}.png'
            # don't know why, but for some reason it is necessary to reapply the layout
            self.fig.layout = self.layout
            self.fig.layout.title = station_name
            pio.write_image(self.fig, fpath_plot, scale=3)
            table_plots['station'].append(station_id)
            table_plots['filename'].append(fpath_plot)
            table_plots['station_long_name'].append(station_name)
        fpath_map = f'{plot_prefix}_description.csv'
        df_plots = pd.DataFrame(data=table_plots)
        df_plots.to_csv(fpath_map)
        self.lb_msg.value = f'Finished saving plots. See {fpath_map} for plot information.'
