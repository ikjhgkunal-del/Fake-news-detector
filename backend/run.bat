@echo off
echo ========================================
echo        Starting Truth Shield Backend
echo ========================================
echo.

cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    echo Activating local .venv...
    call .venv\Scripts\activate.bat
) else if exist "..\.venv\Scripts\activate.bat" (
    echo Activating workspace .venv...
    call ..\.venv\Scripts\activate.bat
) else (
    echo Using system Python / Conda...
    call conda activate aiml-cset312 2>nul
)

echo Starting FastAPI server at http://127.0.0.1:8000 ...
python -m uvicorn main:app --port 8000 --reload
pause