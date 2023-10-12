@echo on
::  usage:  prepro hydro|qual|both config-file
::  Planning study for Calsim 3
::  202305

if {%1%}=={} (
echo "usage:  prepro config-file"
echo Prepro is needed only when the following files change:
echo "Calsim outputs, Martinez stage, Delta Consumptive (DCD/SMCD)"
goto fin
)

set CONFIGFILE=%1%
rem do not add spaces to the following command
if NOT EXIST %CONFIGFILE% GOTO noconfig
echo Prepro is needed only when the CALSIM file changes.

set SCRIPT_HOME=../../scripts
call "../../vista/bin/vscript" ../../scripts/extend_calsim_outputs.py %CONFIGFILE%
call "../../vista/bin/vscript" ./scripts/prep_stage.py %CONFIGFILE%
call "../../vista/bin/vscript" ../../scripts/planning_boundary_flow.py %CONFIGFILE%
::for 1st run if CU timeseries is not extended yet
REM call "../../vista/bin/vscript" ../../scripts/extend_cd_flow.py %CONFIGFILE%
REM call "../../vista/bin/vscript" ../../scripts/prep_dcd_flow.py %CONFIGFILE%
call "../../vista/bin/vscript" ../../scripts/prep_gates.py %CONFIGFILE%
call "../../vista/bin/vscript" ./scripts/prep_ec.py %CONFIGFILE%
REM call "../../vista/bin/vscript" ../../scripts/prep_doc.py %CONFIGFILE%

setlocal
set PATH=c:\Windows\System32;c:\Windows;C:\miniconda3
call ..\..\pydelmod_plan\Scripts\activate.bat
call python ../../scripts/postpro_dss.py timeseries/2021ex.dss "/////15MIN//" timeseries/2021ex.dss
endlocal

goto fin

:noconfig
echo %CONFIGFILE%
echo The configuration file must be specified on the command line
echo and be a valid file

:fin
