@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto install
where py >nul 2>nul
if errorlevel 1 goto usepython
py -3 -m venv .venv
if errorlevel 1 goto fail
goto install
:usepython
python -m venv .venv
if errorlevel 1 goto fail
:install
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m streamlit run app.py --server.address 127.0.0.1 --server.maxUploadSize 64 --browser.gatherUsageStats false
if errorlevel 1 goto fail
exit /b 0
:fail
echo Check that Python 3.11 or 3.12 is installed and that internet is available for first-time setup.
pause
exit /b 1
