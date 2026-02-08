@echo off
REM Quick Run Script for Audio Capture Module
REM Usage: run.bat [command]
REM   demo     - Run quick demo
REM   test     - Run stability test (5 min)
REM   test30   - Run full stability test (30 min)
REM   viz      - Run visualization
REM   devices  - List audio devices

cd /d "%~dp0"

if "%1"=="" goto demo
if "%1"=="demo" goto demo
if "%1"=="test" goto test
if "%1"=="test30" goto test30
if "%1"=="viz" goto viz
if "%1"=="devices" goto devices
goto demo

:demo
echo Running demo...
python demo.py
goto end

:test
echo Running quick stability test (5 minutes)...
python -m tests.stability_test --quick
goto end

:test30
echo Running full stability test (30 minutes)...
python -m tests.stability_test --duration 1800
goto end

:viz
echo Starting visualization...
python -m tests.visualize
goto end

:devices
echo Listing audio devices...
python -c "from src.device_manager import DeviceManager; DeviceManager().print_devices()"
goto end

:end
