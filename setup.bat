@echo off
echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Setup complete!
echo To run the Web App: streamlit run app.py
echo To run Standalone Inference: python inference.py
pause
