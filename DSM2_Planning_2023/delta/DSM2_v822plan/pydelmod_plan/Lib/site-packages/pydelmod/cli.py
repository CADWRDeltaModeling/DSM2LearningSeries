# -*- coding: utf-8 -*-
"""Console script for pydelmod."""
from pydelmod import dsm2ui
from pydelmod.dsm2ui import DSM2FlowlineMap, build_output_plotter
from pydelmod import postpro_dsm2
from pydelmod import dsm2_chan_mann_disp
import sys
import click
import panel as pn
pn.extension()


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.group(context_settings=CONTEXT_SETTINGS)
def main():
    pass


@click.command()
@click.argument("flowline_shapefile", type=click.Path(dir_okay=False, exists=True, readable=True))
@click.argument("hydro_echo_file", type=click.Path(dir_okay=False, exists=True, readable=True))
@click.option("-c","--colored-by", type=click.Choice(['MANNING', 'DISPERSION', 'LENGTH', 'ALL'],case_sensitive=False), default='MANNING')
@click.option("--base-file","-b", type=click.Path(dir_okay=False, exists=True, readable=True))
def map_channels_colored(flowline_shapefile, hydro_echo_file, colored_by, base_file):
    mapui = DSM2FlowlineMap(flowline_shapefile, hydro_echo_file, base_file)
    if colored_by == 'ALL':
        return pn.panel(pn.Column(*[mapui.show_map_colored_by_column(c.upper()) for c in ['MANNING','DISPERSION', 'LENGTH']])).show()
    else:
        return pn.panel(mapui.show_map_colored_by_column(colored_by.upper())).show()

@click.command()
@click.argument("channel_shapefile", type=click.Path(dir_okay=False, exists=True, readable=True))
@click.argument("hydro_echo_file", type=click.Path(dir_okay=False, exists=True, readable=True))
@click.option("-v","--variable", type=click.Choice(['flow', 'stage'],case_sensitive=False), default='flow')
def output_map_plotter(channel_shapefile, hydro_echo_file, variable):
    plotter = build_output_plotter(channel_shapefile, hydro_echo_file, variable)
    pn.serve(plotter.get_panel(),kwargs={'websocket-max-message-size':100*1024*1024})

@click.command()
@click.argument("node_shapefile", type=click.Path(dir_okay=False, exists=True, readable=True))
@click.argument("hydro_echo_file", type=click.Path(dir_okay=False, exists=True, readable=True))
def node_map_flow_splits(node_shapefile, hydro_echo_file):
    netmap = dsm2ui.DSM2GraphNetworkMap(node_shapefile, hydro_echo_file)
    pn.serve(netmap.get_panel(),kwargs={'websocket-max-message-size':100*1024*1024})

@click.command()
@click.argument("process_name", type=click.Choice(['observed', 'model', 'plots'], case_sensitive=False), default='')
@click.argument("json_config_file")
@click.option("--dask/--no-dask", default=False)
def exec_postpro_dsm2(process_name, json_config_file, dask):
    print(process_name, dask,json_config_file)
    postpro_dsm2.run_process(process_name, json_config_file, dask)

@click.command()
@click.argument("chan_to_group_filename", type=click.Path(dir_okay=False, exists=True, readable=True))
@click.argument("chan_group_mann_disp_filename", type=click.Path(dir_okay=False, exists=True, readable=True))
@click.argument("dsm2_channels_input_filename", type=click.Path(dir_okay=False, exists=True, readable=True))
@click.argument("dsm2_channels_output_filename", type=click.Path(dir_okay=False, exists=False, readable=False))
def exec_dsm2_chan_mann_disp(chan_to_group_filename, chan_group_mann_disp_filename, dsm2_channels_input_filename, dsm2_channels_output_filename):
    dsm2_chan_mann_disp.prepro(chan_to_group_filename, chan_group_mann_disp_filename, dsm2_channels_input_filename, dsm2_channels_output_filename)

main.add_command(map_channels_colored)
main.add_command(node_map_flow_splits)
main.add_command(output_map_plotter)
main.add_command(exec_postpro_dsm2)
main.add_command(exec_dsm2_chan_mann_disp)

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
