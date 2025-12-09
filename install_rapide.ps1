# Script d'installation rapide pour éviter les conflits de dépendances
# Exécutez : .\install_rapide.ps1

Write-Host "🚀 Installation rapide des dépendances RAG" -ForegroundColor Green

# 1. Mettre à jour pip
Write-Host "`n📦 Mise à jour de pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# 2. Installer les dépendances de base d'abord
Write-Host "`n📦 Installation des dépendances de base..." -ForegroundColor Yellow
pip install numpy pandas --only-binary :all:

# 3. Installer chromadb avec une version spécifique compatible
Write-Host "`n📦 Installation de chromadb..." -ForegroundColor Yellow
pip install "chromadb==0.4.24" "pydantic<2.0" "fastapi<0.100.0"

# 4. Installer llama-index packages (sans le meta-package)
Write-Host "`n📦 Installation de llama-index..." -ForegroundColor Yellow
pip install llama-index-core llama-index-llms-llama-cpp llama-index-embeddings-huggingface llama-index-vector-stores-chroma

# 5. Installer les autres dépendances
Write-Host "`n📦 Installation des autres dépendances..." -ForegroundColor Yellow
pip install sentence-transformers pypdf pymupdf

# 6. Installer llama-cpp-python depuis l'index spécial (avec wheels précompilés)
Write-Host "`n📦 Installation de llama-cpp-python (peut prendre du temps)..." -ForegroundColor Yellow
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

Write-Host "`n✅ Installation terminée !" -ForegroundColor Green
Write-Host "Vérifiez avec: pip list | Select-String -Pattern 'llama|chroma'" -ForegroundColor Cyan





