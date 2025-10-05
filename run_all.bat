@echo off
REM ============================================================================
REM Bay Area Air Quality Prediction Pipeline - Full Execution
REM ============================================================================

echo ============================================================================
echo Bay Area Air Quality Prediction Pipeline
echo ============================================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+
    exit /b 1
)

echo [Step 0] Checking Python environment...
python -c "import pandas, numpy, xarray, earthaccess, lightgbm" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Some required packages are missing.
    echo Installing required packages...
    pip install pandas numpy xarray earthaccess lightgbm scikit-learn matplotlib seaborn pyarrow requests
)

echo.
echo ============================================================================
echo [Step 1/8] Downloading OpenAQ PM2.5 data...
echo ============================================================================
python 01_download_openaq.py
if errorlevel 1 (
    echo ERROR: OpenAQ download failed!
    exit /b 1
)

echo.
echo ============================================================================
echo [Step 2/8] Downloading TEMPO NO2/O3 data...
echo ============================================================================
python 02_download_tempo.py
if errorlevel 1 (
    echo ERROR: TEMPO download failed!
    echo TIP: Check your Earthdata credentials in ~/.netrc
    exit /b 1
)

echo.
echo ============================================================================
echo [Step 3/8] Downloading MERRA-2 meteorological data...
echo ============================================================================
python 03_download_merra2.py
if errorlevel 1 (
    echo ERROR: MERRA-2 download failed!
    exit /b 1
)

echo.
echo ============================================================================
echo [Step 4/8] Preprocessing and merging datasets...
echo ============================================================================
python 04_preprocess_merge.py
if errorlevel 1 (
    echo ERROR: Preprocessing failed!
    exit /b 1
)

echo.
echo ============================================================================
echo [Step 5/8] Feature engineering...
echo ============================================================================
python 05_feature_engineering.py
if errorlevel 1 (
    echo ERROR: Feature engineering failed!
    exit /b 1
)

echo.
echo ============================================================================
echo [Step 6/8] Training baseline (persistence) model...
echo ============================================================================
python 06_train_baseline.py
if errorlevel 1 (
    echo ERROR: Baseline training failed!
    exit /b 1
)

echo.
echo ============================================================================
echo [Step 7/8] Training residual correction models...
echo ============================================================================
python 07_train_residual.py
if errorlevel 1 (
    echo ERROR: Residual model training failed!
    exit /b 1
)

echo.
echo ============================================================================
echo [Step 8/8] Evaluating and generating visualizations...
echo ============================================================================
python 08_evaluate.py
if errorlevel 1 (
    echo ERROR: Evaluation failed!
    exit /b 1
)

echo.
echo ============================================================================
echo Pipeline Complete!
echo ============================================================================
echo.
echo Results:
echo   - Metrics:  models/evaluation_metrics.json
echo   - Plots:    plots/
echo   - Models:   models/
echo   - Data:     tables/
echo.
echo ============================================================================

pause
