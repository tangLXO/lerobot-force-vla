@echo off
cd /d "%~dp0..\.."
uv run --no-sync python examples/force_teacher/b601_tuner.py
if errorlevel 1 pause
