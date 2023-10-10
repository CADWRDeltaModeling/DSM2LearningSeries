set PATH=c:\Windows\System32;c:\Windows
call ..\pydelmod_plan\Scripts\activate.bat
rem .\pydelmod_plan\Scripts\conda-unpack.exe
pip install autopep8
rem installation of Jupyter kernel
ipython kernel  install --prefix .\kernel
call jupyter kernelspec install .\kernel\share\jupyter\kernels\python3 --user
call jupyter nbextension enable --py --user widgetsnbextension
rem start jupyter notebook in directory
REM call jupyter notebook --notebook-dir=2020nb
call jupyter notebook --notebook-dir=.