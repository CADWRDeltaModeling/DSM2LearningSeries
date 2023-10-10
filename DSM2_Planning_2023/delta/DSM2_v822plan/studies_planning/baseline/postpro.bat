:: run with pyhecdss
setlocal
set PATH=c:\Windows\System32;c:\Windows;C:\miniconda3
call ..\..\pydelmod_plan\Scripts\activate.bat

python ../../scripts/postpro_dss.py output/2021EX_EC.dss
python ../../scripts/postpro_dss.py output/2021EX_STAGE.dss
python ../../scripts/postpro_dss.py output/2021EX_FLOW.dss
python ../../scripts/postpro_dss.py output/2021EX_VEL.dss

call ..\..\pydelmod_plan\Scripts\deactivate.bat
endlocal
