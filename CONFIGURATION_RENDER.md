# Configuration Render - Fichier de dépendances

## 🔍 Problème identifié

Votre commande de build utilise `requirements.txt` mais `render.yaml` utilise `requirements_render.txt`.

## 📋 Options de configuration

### Option 1 : Utiliser `requirements_render.txt` (RECOMMANDÉ)

**Dans le Dashboard Render → rag-api → Settings → Build Command :**
```bash
pip install --upgrade pip setuptools wheel && pip install --no-cache-dir --prefer-binary -r requirements_render.txt && pip uninstall -y torch torchvision torchaudio onnxruntime sentence-transformers transformers 2>/dev/null || true && pip cache purge
```

**Avantages :**
- ✅ Optimisé pour Render (512 MB RAM)
- ✅ Exclut `torch` et `sentence-transformers` (économise ~400 MB)
- ✅ Contient `requests` pour le wrapper personnalisé

### Option 2 : Utiliser `requirements.txt` (développement local)

**Si vous préférez continuer avec `requirements.txt` :**

Le fichier `requirements.txt` a été mis à jour pour inclure `requests>=2.31.0` (nécessaire pour le wrapper personnalisé).

**ATTENTION :** 
- ❌ `requirements.txt` inclut `sentence-transformers` (charge torch, ~400 MB RAM)
- ⚠️ Risque de dépasser la limite de 512 MB sur Render
- ✅ Fonctionne mais n'est pas optimal pour Render

## 🎯 Recommandation

**Utilisez `requirements_render.txt` sur Render** car :
1. Optimisé pour la limite de 512 MB
2. Exclut les packages lourds
3. Inclut `requests` pour le wrapper personnalisé

## 📝 Commandes de build recommandées

### Pour Render (production)
```bash
pip install --upgrade pip setuptools wheel
pip install --no-cache-dir --prefer-binary -r requirements_render.txt
pip uninstall -y torch torchvision torchaudio onnxruntime sentence-transformers transformers 2>/dev/null || true
pip cache purge
```

### Pour développement local
```bash
pip install -r requirements.txt
```

## ✅ Vérification

Après redéploiement, vérifiez dans les logs :
```
📦 Utilisation du wrapper personnalisé HuggingFaceTextEmbeddingsWrapper (nouvelle API router.huggingface.co)
✅ Embeddings via API Hugging Face (wrapper personnalisé, pas de modèle en mémoire, économise ~400 MB RAM)
```
