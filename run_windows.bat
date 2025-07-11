@echo off
echo ============================================
echo Text Summarization AI - Windows Launcher
echo ============================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

:: Check if pip is installed
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: pip is not installed
    echo Please install pip or update Python
    pause
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Install requirements
echo Installing requirements...
pip install -r requirements.txt

:: Check if models exist
if not exist "models\tokenizer.pkl" (
    echo.
    echo Models not found. Training models...
    echo This may take a few minutes...
    python train.py --epochs 3 --batch_size 16
)

:: Start the application
echo.
echo Starting Text Summarization AI...
echo The application will be available at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

python app.py

pause