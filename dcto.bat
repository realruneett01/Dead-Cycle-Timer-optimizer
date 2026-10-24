@echo off
rem Pendrive launcher for Windows. Double-click, or run from a terminal with options.
cd /d "%~dp0"
for %%V in (3.12 3.11) do (
  py -%%V -c "import sys" >nul 2>&1 && (py -%%V launch.py %* & goto :done)
)
python -c "import sys; sys.exit(0 if sys.version_info[:2] in ((3,11),(3,12)) else 1)" >nul 2>&1 && (python launch.py %* & goto :done)
echo Python 3.11 or 3.12 is required but was not found on this computer.
:done
if "%~1"=="" pause
