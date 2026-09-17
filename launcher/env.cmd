@echo off
rem Process-only environment for the portable runtime. start.bat calls this with RT set to <root>\runtime.
rem Everything here only lives in this cmd window: no registry, no user PATH, nothing outside the program folder.
rem Keep in sync with _ENV_DEFAULTS in launcher\setup_runtime.py (tests\test_launcher.py checks it).
rem VS_HOST is deliberately NOT set: the server listens on 127.0.0.1 unless the LAN switch in settings is on.
if not defined RT exit /b 1

rem the user's real temp folder, for precheck.py (runs after TEMP is redirected)
if not defined VS_ORIG_TEMP set "VS_ORIG_TEMP=%TEMP%"
if not defined VS_ORIG_TMP set "VS_ORIG_TMP=%TMP%"
set "TEMP=%RT%\tmp"
set "TMP=%RT%\tmp"

rem uv: cache, managed Python, tools, credentials, lock files, venv
set "UV_CACHE_DIR=%RT%\cache\uv"
set "UV_PYTHON_INSTALL_DIR=%RT%\python"
set "UV_PYTHON_BIN_DIR=%RT%\python-bin"
set "UV_PYTHON_INSTALL_BIN=0"
set "UV_PYTHON_INSTALL_REGISTRY=0"
set "UV_TOOL_DIR=%RT%\uv-tools"
set "UV_TOOL_BIN_DIR=%RT%\uv-tools\bin"
set "UV_CREDENTIALS_DIR=%RT%\uv-credentials"
set "UV_NO_CONFIG=1"
set "UV_MANAGED_PYTHON=1"
set "UV_PYTHON_PREFERENCE="
set "UV_PROJECT_ENVIRONMENT=%RT%\venv"
set "UV_HTTP_TIMEOUT=120"
set "UV_HTTP_RETRIES=5"
set "UV_SYSTEM_CERTS=1"
set "PIP_CACHE_DIR=%RT%\cache\pip"

rem Python: never read the user's site-packages or Python variables
set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONPATH="
set "PYTHONHOME="
set "PYTHONSTARTUP="
set "VIRTUAL_ENV="
set "CONDA_PREFIX="

rem library caches that default to the user profile
set "HF_HOME=%RT%\cache\huggingface"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "TORCH_HOME=%RT%\cache\torch"
set "TORCHINDUCTOR_CACHE_DIR=%RT%\cache\torchinductor"
set "TRITON_CACHE_DIR=%RT%\cache\triton"
set "CUDA_CACHE_PATH=%RT%\cache\nv-compute"
set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
set "NUMBA_CACHE_DIR=%RT%\cache\numba"
set "MPLCONFIGDIR=%RT%\cache\matplotlib"
set "XDG_CACHE_HOME=%RT%\cache\xdg"
set "NLTK_DATA=%RT%\cache\nltk_data"
set "LIBROSA_DATA_DIR=%RT%\cache\librosa"
set "GRADIO_TEMP_DIR=%RT%\tmp\gradio"
set "LLAMA_CACHE=%RT%\cache\llama.cpp"
set "DENO_DIR=%RT%\cache\deno"
set "DENO_NO_UPDATE_CHECK=1"
set "VS_CACHE_DIR=%RT%\cache"

rem deno (yt-dlp's JavaScript runtime) and other tools installed into the venv
set "PATH=%RT%\venv\Scripts;%PATH%"
exit /b 0
