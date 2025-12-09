@echo off
echo ========================================
echo 🚀 Démarrage de l'API RAG
echo ========================================
echo.

REM Activer l'environnement virtuel si présent
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

REM Lancer l'API
python src/api/main_api.py

pause



