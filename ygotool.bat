@echo off
setlocal enabledelayedexpansion
set PORT=8000
set ACTION=%1
if "%ACTION%"=="" set ACTION=toggle
cd /d "%~dp0"
set KILLPID=
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%PORT% " ^| findstr LISTENING') do set KILLPID=%%a
if /i "%ACTION%"=="status" goto status
if /i "%ACTION%"=="stop" goto do_stop
if defined KILLPID goto do_stop
echo [ygotool] starting http://127.0.0.1:%PORT% ...
start "ygotool-server" /MIN python cli.py serve
echo [ygotool] started. open http://127.0.0.1:%PORT% ; run again to stop.
goto :eof
:do_stop
if not defined KILLPID (echo [ygotool] not running. & goto :eof)
taskkill /F /PID !KILLPID! >nul 2>&1
echo [ygotool] stopped PID !KILLPID!.
goto :eof
:status
if defined KILLPID (echo [ygotool] running PID !KILLPID!) else (echo [ygotool] not running)
goto :eof
