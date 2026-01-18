#!/bin/bash
# Script d'installation optimisé pour Render

set -e

echo "🔧 Installation optimisée pour Render (512MB RAM max)"

# Mettre à jour pip
pip install --upgrade pip setuptools wheel

# Installer les dépendances de base
echo "📦 Installation des dépendances de base..."
pip install --no-cache-dir --prefer-binary -r requirements_render_minimal.txt

# Installer llama-index packages SANS dépendances pour éviter torch
echo "📦 Installation des packages llama-index sans dépendances..."
pip install --no-deps llama-index-llms-huggingface || true
pip install --no-deps llama-index-embeddings-huggingface || true
pip install --no-deps llama-index-readers-file || true

# Installer SEULEMENT les dépendances nécessaires (sans torch)
echo "📦 Installation des dépendances minimales..."
pip install --no-cache-dir tokenizers==0.15.2

# Forcer la désinstallation de torch si installé
echo "🗑️  Nettoyage des packages lourds..."
pip uninstall -y torch torchvision torchaudio onnxruntime sentence-transformers transformers || true

# Nettoyer le cache
pip cache purge

echo "✅ Installation terminée"
