@echo off
echo --- Mise a jour de PIP ---
python -m pip install --upgrade pip

echo.
echo --- Installation des dependances (Mode Binaire Preferé) ---
pip install -r requirements.txt --prefer-binary

echo.
echo --- Installation terminee ! ---
pause