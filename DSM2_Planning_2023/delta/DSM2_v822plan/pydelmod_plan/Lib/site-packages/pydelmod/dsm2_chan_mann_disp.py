import pandas as pd
import numpy as np
import os


def prepro(chan_to_group_filename, chan_group_mann_disp_filename, dsm2_channels_input_filename, dsm2_channels_output_filename):
    '''
    read 3 files
    cg_filename (chngrp_8_3_.csv)
    cg_mann_disp_filename (chngrp_mann_disp.csv)
    ci_filename (channel_std_delta_grid_from_CSDP_NAVD.inp)
    create a new channel_std_delta_grid.inp file which has mannings and dispersion values replaced with values specified
    in the first two files.
    '''

    ci_filename_prefix = 'channel_std_delta_grid_from_CSDP_NAVD'

    # cg_filename = 'chngrp.csv'
    # cg_mann_disp_filename = 'chan_mann_disp.csv'
    # ci_filename = '../geom/'+ci_filename_prefix+'.inp'
    # outfile_name = '../geom/'+ci_filename_prefix+'_new.inp'
    outfile = open(dsm2_channels_output_filename, 'w')
    cd_dir = '//cnrastore-bdo/Delta_Mod/Share/DSM2/full_calibration_8_3/delta/dsm2v8.3/studies/'

    channel_group_df = pd.read_csv(chan_to_group_filename)
    channel_mann_disp_df = pd.read_csv(chan_group_mann_disp_filename)

    # ci_filename = '//cnrastore-bdo/Delta_Mod/Share/DSM2/full_calibration_8_3/channel_groups/channel_std_delta_grid_from_CSDP_NAVD.inp'
    ci_file = open(dsm2_channels_input_filename)
    # outfile = open('//cnrastore-bdo/Delta_Mod/Share/DSM2/full_calibration_8_3/channel_groups/new_channel_std_delta_grid.inp', 'w')
    # assume that CHANNEL section will be first input section in file. Other sections will be copied without modification
    # to the new file
    processing_section = False
    for line in ci_file:
        if 'CHAN_NO' in line and 'MANNING' in line and 'DISPERSION' in line:
            processing_section = True
            outfile.write(line)
        elif 'END' in line:
            processing_section = False
            outfile.write(line)
        else:
            if processing_section:
                if not line.startswith('#'):
                    parts = line.split()
                    chan = int(parts[0].strip())
                    length = int(parts[1].strip())
                    mann = float(parts[2].strip())
                    disp = float(parts[3].strip())
                    upnode = int(parts[4].strip())
                    downnode = int(parts[5].strip())
                    chan_group = channel_group_df.loc[channel_group_df['channel_id'] == chan, 'group_id'].iloc[0]
                    new_mann = channel_mann_disp_df.loc[channel_mann_disp_df['group_id'] == chan_group, 'manning'].iloc[0]
                    new_disp = channel_mann_disp_df.loc[channel_mann_disp_df['group_id'] == chan_group, 'dispersion'].iloc[0]
                    if not np.isnan(new_mann):
                        mann = new_mann
                    if not np.isnan(new_disp):
                        disp = new_disp
                    line = '%-9.0f%-9.0f%-9.4f%-12.4f%-8.0f%-8.0f\n' % (chan, length, mann, disp, upnode, downnode)
                    outfile.write(line)
                else:
                    outfile.write(line)
            else:
                outfile.write(line)
    outfile.close()

    print('You should now execute the following commands:')
    print('cd ../geom')
    print('ren '+ci_filename_prefix+'.inp ', ci_filename_prefix+'_prev.inp')
    print('ren '+ci_filename_prefix+'_new.inp ', ci_filename_prefix+'.inp')

    # These commands don't work--some kind of permission error
    # rename existing file, adding _prev to end
    #os.replace(ci_filename_prefix+'.inp', ci_filename_prefix+'_prev.inp')
    # rename new file to original name
    #os.rename(ci_filename_prefix+'_new.inp', ci_filename_prefix+'.inp')
