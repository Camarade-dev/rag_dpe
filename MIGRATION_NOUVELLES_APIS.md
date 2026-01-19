# Migration vers les nouvelles APIs Hugging Face

## ✅ Modifications effectuées

### 1. **LLM : Migration vers `llama-index-llms-huggingface-api`**

**Ancienne classe (dépréciée)** :
```python
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
```

**Nouvelle classe (recommandée)** :
```python
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
```

**Changements dans `src/rag_core/query_engine.py`** :
- Import prioritaire de la nouvelle API avec fallback vers l'ancienne
- Gestion des paramètres `token` vs `api_key` (compatibilité avec les deux versions)
- Messages de log améliorés pour identifier quelle API est utilisée

### 2. **Embeddings : Migration vers `llama-index-embeddings-huggingface-api`**

**Ancienne classe (dépréciée, cause erreur 410 Gone)** :
```python
from llama_index.embeddings.huggingface import HuggingFaceInferenceAPIEmbedding
```

**Nouvelle classe (recommandée, corrige l'erreur 410)** :
```python
from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding
```

**Changements dans `src/rag_core/query_engine.py`** :
- Import prioritaire de la nouvelle API avec fallback vers l'ancienne
- Gestion des paramètres `api_key` vs `token` (compatibilité avec les deux versions)
- Meilleure gestion des erreurs avec messages explicites

### 3. **Dépendances mises à jour**

#### `requirements.txt` (développement local)
- ✅ Ajout de `llama-index-llms-huggingface-api>=0.1.0` (prioritaire)
- ✅ Ajout de `llama-index-embeddings-huggingface-api>=0.1.0` (prioritaire)
- ✅ Conservation des anciennes APIs comme fallback (dépréciées)

#### `requirements_render.txt` (déploiement Render)
- ✅ Ajout de `llama-index-llms-huggingface-api>=0.1.0` (prioritaire)
- ✅ Ajout de `llama-index-embeddings-huggingface-api>=0.1.0` (prioritaire)
- ✅ Conservation des anciennes APIs comme fallback (dépréciées)
- ✅ Exclusion explicite de `torch`, `sentence-transformers` (économise ~400 MB RAM)

## 🎯 Avantages

### Correction de l'erreur 410 Gone
- Les nouvelles classes utilisent l'API `text-embeddings` au lieu de `feature-extraction`
- Plus compatible avec les modèles Hugging Face récents
- Meilleure gestion des modèles d'embedding

### Optimisation mémoire pour Render
- Aucun modèle chargé en RAM (appels HTTP uniquement)
- Économie de ~400 MB en excluant `torch` et `sentence-transformers`
- Compatible avec la limite de 512 MB RAM de Render

### Meilleure maintenabilité
- Utilisation des APIs recommandées (non dépréciées)
- Plus de warnings de dépréciation
- Compatibilité ascendante avec fallback automatique

## 📋 Variables d'environnement nécessaires

### Sur Render (service `rag-api`)
- `USE_API_EMBEDDINGS=true` ✅ (obligatoire)
- `HUGGINGFACE_API_KEY=hf_...` ✅ (obligatoire)
- `HUGGINGFACE_MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1` ✅ (recommandé)
- `HUGGINGFACE_EMBEDDING_MODEL=intfloat/multilingual-e5-base` (optionnel, valeur par défaut)

## 🔍 Vérification après déploiement

Dans les logs Render, vous devriez voir :
```
📦 Utilisation de llama-index-llms-huggingface-api (nouvelle API)
📦 Utilisation de llama-index-embeddings-huggingface-api (nouvelle API)
📦 Modèle d'embedding: intfloat/multilingual-e5-base
✅ Embeddings via API Hugging Face (pas de modèle en mémoire, économise ~400 MB RAM)
```

Si vous voyez :
```
⚠️  Utilisation de llama-index-llms-huggingface (ancienne API, dépréciée)
⚠️  Utilisation de llama-index-embeddings-huggingface (ancienne API, dépréciée)
```

Cela signifie que les nouveaux packages ne sont pas installés. Vérifiez que `requirements_render.txt` inclut bien les nouvelles dépendances.

## 🚀 Prochaines étapes

1. **Redéployer l'API RAG sur Render**
2. **Vérifier les logs** pour confirmer l'utilisation des nouvelles APIs
3. **Tester une requête RAG** pour vérifier qu'il n'y a plus d'erreur 410 Gone

## 📝 Notes techniques

### Paramètres des classes

Les nouvelles classes peuvent accepter soit `api_key`, soit `token` comme paramètre. Le code gère automatiquement les deux cas :

```python
try:
    self.llm = HuggingFaceInferenceAPI(model_name=..., token=api_key, ...)
except TypeError:
    self.llm = HuggingFaceInferenceAPI(model_name=..., api_key=api_key, ...)
```

### Modèles d'embedding compatibles

- ✅ `intfloat/multilingual-e5-base` (par défaut) - fonctionne avec l'API text-embeddings
- ✅ `BAAI/bge-small-en-v1.5` (fallback) - alternative si le premier ne fonctionne pas
- ❌ `sentence-transformers/all-MiniLM-L6-v2` (retourne 410 Gone)
- ❌ `sentence-transformers/all-mpnet-base-v2` (retourne 410 Gone)
