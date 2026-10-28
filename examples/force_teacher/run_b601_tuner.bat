@echo off
cd /d "%~dp0..\.."
set "PYTHONPATH=%CD%\src;%CD%"
uv run --no-sync python examples/force_teacher/b601_tuner.py %*
if errorlevel 1 pause
