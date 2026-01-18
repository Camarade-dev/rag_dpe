# Guide de déploiement sur Render avec Solution 1 (API Embeddings)

## ✅ Solution 1 implémentée

La solution 1 utilise l'API Hugging Face pour les embeddings, ce qui évite d'installer `sentence-transformers` et `torch`, économisant ainsi ~400 MB de RAM.

## Fichiers modifiés

1. ✅ `query_engine.py` : Support de `USE_API_EMBEDDINGS=true`
2. ✅ `requirements_render.txt` : Version sans sentence-transformers
3. ✅ `render.yaml` : Configuration pour utiliser `requirements_render.txt`

## Configuration Render

### Variables d'environnement requises

Dans le service `rag-api` sur Render, configurez :

```
PYTHON_VERSION=3.12.0
PORT=8002
CHROMA_DB_PATH=/tmp/chroma_db
LLM_PROVIDER=huggingface
USE_API_EMBEDDINGS=true
HUGGINGFACE_API_KEY=votre_clé_api
HUGGINGFACE_MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1
```

### Build Command

Le `render.yaml` est configuré pour utiliser `requirements_render.txt` :
```bash
pip install --upgrade pip setuptools wheel && pip install --no-cache-dir --prefer-binary -r requirements_render.txt && pip cache purge
```

## Vérification

Après déploiement, vous devriez voir dans les logs :
```
✅ Embeddings via API Hugging Face (pas de modèle en mémoire, économise ~400 MB RAM)
```

## Dépannage

### Si "Out of memory" persiste

1. Vérifier que `requirements_render.txt` est utilisé (pas `requirements.txt`)
2. Vérifier que `USE_API_EMBEDDINGS=true` est bien configuré
3. Vérifier que `HUGGINGFACE_API_KEY` est configurée
4. Vérifier les logs pour voir quelle méthode d'embedding est utilisée

### Si HuggingFaceInferenceAPIEmbedding n'est pas disponible

Vérifier que `llama-index-embeddings-huggingface>=0.1.0` est bien installé dans `requirements_render.txt`.

## Développement local

Pour le développement local, utilisez `requirements.txt` qui inclut `sentence-transformers` :
```bash
pip install -r requirements.txt
# Ne pas définir USE_API_EMBEDDINGS ou le mettre à "false"
```

Cela permet d'utiliser les embeddings locaux pour éviter les appels API pendant le développement.
