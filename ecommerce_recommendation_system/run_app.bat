@echo off
REM =====================================================================
REM  run_app.bat
REM  Convenience launcher for the E-Commerce Recommendation System.
REM  Creates/activates a virtual environment, installs requirements only
REM  when needed, then launches the Streamlit app.
REM =====================================================================

setlocal

echo ============================================================
echo  E-Commerce Recommendation System - Launcher
echo ============================================================

REM --- Step 1: Check Python is available -------------------------------
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found on PATH.
    echo Please install Python 3.10 or 3.11 from https://www.python.org/downloads/
    echo and make sure "Add Python to PATH" is checked during installation.
    pause
    exit /b 1
)

REM --- Step 2: Create the virtual environment if it doesn't exist ------
if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment in .\venv ...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

REM --- Step 3: Activate the virtual environment -------------------------
call venv\Scripts\activate.bat

REM --- Step 4: Install requirements only if not already installed -------
python -c "import streamlit, pandas, numpy, matplotlib, seaborn, sklearn" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages ^(this may take a few minutes^)...
    pip install --upgrade pip >nul
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install requirements.
        pause
        exit /b 1
    )
) else (
    echo Requirements already installed - skipping install step.
)

REM --- Step 5: Check dataset presence ------------------------------------
if not exist "data\*.csv" (
    if not exist "data\*.xlsx" (
        echo.
        echo [WARNING] No dataset file found in the "data" folder.
        echo Please place the downloaded CSV file inside the "data" folder.
        echo The app will still launch and show an on-screen instruction.
        echo.
    )
)

REM --- Step 6: Run the Streamlit app -------------------------------------
echo.
echo Starting Streamlit app... a browser tab will open automatically.
echo Press CTRL+C in this window to stop the app.
echo.
streamlit run app.py

endlocal
pause
