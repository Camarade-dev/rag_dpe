@echo off
echo ========================================
echo Test de l'API Hugging Face Embeddings
echo ========================================
echo.

REM Définir votre clé API ici
set HUGGINGFACE_API_KEY=your_key_here

REM Lancer le test
cd /d "%~dp0"
python test/test_embedding_api.py

pause
