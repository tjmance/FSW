@echo off
REM This batch file activates the faceswap conda environment and launches the GUI.

REM Change these paths to where you cloned DeepFaceLive and roop-cam
set "DEEPFACELIVE_PATH=%~dp0\..\DeepFaceLive"
set "ROOPCAM_PATH=%~dp0\..\roop-cam"

REM Activate conda environment
call conda activate faceswap

REM Launch the dashboard
python "%~dp0\python\pipeline_gui.py"