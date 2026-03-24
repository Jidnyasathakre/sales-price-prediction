@echo off
echo Installing required packages...
pip install -r requirements.txt
echo.
echo Generating sample data...
python generate_sample_data.py
echo.
echo Setup complete! Run run_app.bat to start the application.
pause

