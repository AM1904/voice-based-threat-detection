@echo off
cd /d "%~dp0"
if "%1"=="test" (python -m tests.performance_test) else (python demo.py %*)
