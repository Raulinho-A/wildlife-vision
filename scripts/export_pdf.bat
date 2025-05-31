@echo off

:: ===============================================================
:: EXPORTADOR DE NOTEBOOKS A PDF PARA WINDOWS
:: ---------------------------------------------------------------
:: ⚠️ REQUISITOS:
:: 1. Tener instalado Pandoc → https://github.com/jgm/pandoc/releases/tag/3.7.0.2
:: 2. Tener instalado MiKTeX (para xelatex) → https://miktex.org/download
::    - Durante la instalación, activa "Install missing packages on-the-fly"
:: 3. Tener instalado nbconvert dentro del entorno virtual:
::    pip install nbconvert
:: ---------------------------------------------------------------
:: Este script intenta usar dependencias a nivel global.
:: Si no lo encuentra, utiliza la ruta definida en env.bat
:: ===============================================================

echo ---------- Exportando notebook a PDF ----------

:: Obtener ruta del script actual (por si lo ejecutas desde otro lado)
set SCRIPT_DIR=%~dp0

call "%SCRIPT_DIR%env.bat"

:: Verificar si pandoc está disponible
where pandoc >nul 2>&1
if errorlevel 1 (
    echo Pandoc no encontrado globalmente. Intentando con ruta definida...
    if exist "%PANDOC_PATH%\pandoc.exe" (
        set "PATH=%PATH%;%PANDOC_PATH%"
    ) else (
        echo Pandoc no encontrado en %PANDOC_PATH%
        echo Instala Pandoc desde https://github.com/jgm/pandoc/releases/tag/3.7.0.2
        pause
        exit /b
    )
) else (
    echo Pandoc ya disponible en el sistema.
)

:: Verificar si xelatex está disponible
where xelatex >nul 2>&1
if errorlevel 1 (
    echo xelatex no encontrado globalmente. Intentando con ruta definida...
    if exist "%XELATEX_PATH%\xelatex.exe" (
        set "PATH=%PATH%;%XELATEX_PATH%"
    ) else (
        echo xelatex no encontrado en %XELATEX_PATH%
        echo Instala MiKTeX desde https://miktex.org/download
        pause
        exit /b
    )
) else (
    echo xelatex ya está disponible en el sistema.
)

:: Exportar notebooks a PDF
jupyter-nbconvert --to pdf notebooks/1.0-raa-eda.ipynb --output-dir=reports/pdf
jupyter-nbconvert --to pdf notebooks/2.0-raa-train-model.ipynb --output-dir=reports/pdf

echo Exportación finalizada.
pause