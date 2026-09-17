@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"
set "ROOT=%CD%"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

rem This file stays ASCII-only and CRLF. User-facing Chinese text lives in launcher\msg\*.txt.
rem Variables holding paths are always quoted and never used inside ( ) blocks.

rem ---- developer mode: a .dev marker file (ignored by git, never in release zips) ----
if exist "%ROOT%\.dev" goto dev

set "RT=%ROOT%\runtime"
set "MSG=%ROOT%\launcher\msg"
if not exist "%MSG%\install_failed.txt" goto not_extracted

rem ---- 1. folder location (checked before anything is downloaded) ----
if not "%ROOT:~80,1%"=="" goto bad_path_long
cd | findstr /l /c:"%%" /c:";" /c:"!" /c:"^" /c:"&" >nul
if not errorlevel 1 goto bad_chars
cd | findstr /i /l /c:"\OneDrive" /c:"\Program Files" >nul
if not errorlevel 1 goto bad_location
if "%VS_TEST_ALLOW_TEMP%"=="1" goto location_ok
cd | findstr /i /l /c:"\AppData\Local\Temp" /c:"\Windows\Temp" >nul
if not errorlevel 1 goto bad_location
:location_ok
rem Non-English characters (Chinese, Japanese, e with accent...): nagisa (DyNet) and tar.exe cannot open such paths.
rem Code page 20127 is plain ASCII, so "cd" printed through it differs from the real path when any such character is there.
rem If this code page is missing, launcher\precheck.py still catches it (after uv and Python are downloaded).
chcp 20127 >nul 2>&1
if errorlevel 1 goto ascii_checked
set "ASCII_CD="
for /f "delims=" %%p in ('cd') do set "ASCII_CD=%%p"
chcp 65001 >nul
if not "%ASCII_CD%"=="%CD%" goto non_ascii
:ascii_checked
chcp 65001 >nul

rem ---- 2. NVIDIA driver (no NVIDIA card: stop here, see plan D11) ----
if exist "%SystemRoot%\System32\nvidia-smi.exe" goto nvidia_ok
if exist "%ProgramFiles%\NVIDIA Corporation\NVSMI\nvidia-smi.exe" goto nvidia_ok
goto no_nvidia
:nvidia_ok

rem ---- 3. process-only environment (nothing written outside this folder) ----
if not exist "%RT%\logs" mkdir "%RT%\logs" 2>nul
if not exist "%RT%\logs" goto not_writable
if not exist "%RT%\tmp" mkdir "%RT%\tmp"
if not exist "%RT%\downloads" mkdir "%RT%\downloads"
if not exist "%RT%\uv" mkdir "%RT%\uv"
call "%ROOT%\launcher\env.cmd"

rem ---- 4. uv and Python ----
rem Two start.bat windows opened at once: only one downloads uv and installs Python; the other waits.
rem The lock is runtime\uv\install.lock held open by the 9> redirection (released when that window closes).
set "BASEPY=%RT%\python\cpython-3.12.14-windows-x86_64-none\python.exe"
if not exist "%RT%\uv\uv.exe" goto need_uv
if exist "%BASEPY%" goto have_python
:need_uv
if not exist "%BASEPY%" type "%MSG%\first_run.txt"
set "UV_WAITED="
:uv_lock
set "UV_LOCKED="
(call :uv_and_python 9>"%RT%\uv\install.lock") 2>nul
if defined UV_LOCKED goto uv_checked
if not defined UV_WAITED type "%MSG%\waiting_other_window.txt"
set "UV_WAITED=1"
ping -n 3 127.0.0.1 >nul
goto uv_lock
:uv_checked
if not exist "%RT%\uv\uv.exe" goto fail
if not exist "%BASEPY%" goto fail
:have_python

rem ---- 5. precheck and runtime setup (Python prints the Chinese messages) ----
rem Exit codes are compared with 0: a crash or a kill gives a negative code that "if errorlevel 1" misses.
"%BASEPY%" -I -X utf8 "%ROOT%\launcher\precheck.py"
if not "%errorlevel%"=="0" goto fail_pause
"%BASEPY%" -I -X utf8 "%ROOT%\launcher\setup_runtime.py"
if not "%errorlevel%"=="0" goto fail_pause

rem ---- 6. start the server ----
"%RT%\venv\Scripts\python.exe" -s -m app.server %*
rem Any exit code but 0 is a crash, including negative ones (0xC0000005 when python.exe crashes, -1 from Task Manager)
rem that "if errorlevel 1" misses. Ctrl+C (0xC000013A = -1073741510) is a normal way to stop the server.
set "SERVER_EXIT=%errorlevel%"
if "%SERVER_EXIT%"=="0" exit /b 0
if "%SERVER_EXIT%"=="-1073741510" exit /b 0
goto server_failed

:uv_and_python
rem Runs only while this window holds runtime\uv\install.lock.
set "UV_LOCKED=1"
if exist "%RT%\uv\uv.exe" goto locked_python
type "%MSG%\getting_uv.txt"
call "%ROOT%\launcher\get_uv.cmd"
if not exist "%RT%\uv\uv.exe" exit /b 1
:locked_python
if exist "%BASEPY%" exit /b 0
type "%MSG%\installing_python.txt"
"%RT%\uv\uv.exe" python install 3.12.14 >>"%RT%\logs\install.log" 2>&1
exit /b 0

:dev
rem Same as the start.bat the author used before packaging: conda env "subtitle" or VS_PYTHON.
set "ENV_PY=%VS_PYTHON%"
if not defined VS_PYTHON set "ENV_PY=%USERPROFILE%\.conda\envs\subtitle\python.exe"
if exist "%ENV_PY%" goto dev_run
type "%ROOT%\launcher\msg\dev_no_python.txt"
echo "%ENV_PY%"
echo.
type "%ROOT%\launcher\msg\press_any_key.txt"
pause >nul
exit /b 1
:dev_run
set PYTHONNOUSERSITE=1
set PYTHONIOENCODING=utf-8
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
"%ENV_PY%" -s -m app.server %*
set "SERVER_EXIT=%errorlevel%"
if "%SERVER_EXIT%"=="0" exit /b 0
if "%SERVER_EXIT%"=="-1073741510" exit /b 0
goto dev_stopped
:dev_stopped
echo.
type "%ROOT%\launcher\msg\press_any_key.txt"
pause >nul
exit /b 1

:bad_path_long
type "%MSG%\path_too_long.txt"
goto fail_pause
:bad_chars
type "%MSG%\bad_chars.txt"
goto fail_pause
:bad_location
type "%MSG%\bad_location.txt"
goto fail_pause
:non_ascii
type "%MSG%\non_ascii.txt"
goto fail_pause
:no_nvidia
type "%MSG%\no_nvidia.txt"
goto fail_pause
:not_writable
type "%MSG%\not_writable.txt"
goto fail_pause
:fail
type "%MSG%\install_failed.txt"
echo "%RT%\logs\install.log"
:fail_pause
echo.
type "%MSG%\press_any_key.txt"
pause >nul
exit /b 1
:server_failed
type "%MSG%\server_stopped.txt"
echo.
type "%MSG%\press_any_key.txt"
pause >nul
exit /b 1

:not_extracted
rem The launcher folder is missing (usually start.bat was opened from inside the zip), so there is no
rem msg file to show. This is the only non-ASCII text in this file; keep it at the very end.
echo.
echo 找不到 launcher 資料夾。請先把整個壓縮檔解壓縮到 D:\BilingualSubtitles 這類位置，再從那裡打開 start.bat。按任意鍵關閉這個視窗。
pause >nul
exit /b 1
