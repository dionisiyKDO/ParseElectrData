@echo off
@REM echo Creating virtual environment with uv...
@REM uv venv .venv

echo Activating virtual environment...
call .venv\Scripts\activate

@REM echo Installing dependencies...
@REM uv pip install -r requirements.txt

echo Building executable...
python build.py build

echo Done!
pause