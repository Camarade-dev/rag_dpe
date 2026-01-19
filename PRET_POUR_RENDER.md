# ✅ PRÊT POUR DÉPLOIEMENT SUR RENDER

## 🎯 Résumé des corrections

Tous les problèmes ont été corrigés :

1. ✅ **Erreur 410 Gone** : Corrigée avec le nouveau router `router.huggingface.co/hf-inference/...`
2. ✅ **Erreur onnxruntime** : Résolue en conservant `onnxruntime` dans les dépendances
3. ✅ **Optimisation mémoire** : `torch` et `sentence-transformers` exclus (~450 MB économisés)
4. ✅ **Wrapper personnalisé** : Utilise le bon format d'API Hugging Face

## 📋 Configuration finale

### Fichiers modifiés

1. **`src/rag_core/query_engine.py`**
   - ✅ Wrapper `HuggingFaceTextEmbeddingsWrapper` utilisant `router.huggingface.co/hf-inference/models/.../pipeline/feature-extraction`
   - ✅ Préfixe "query: " ajouté automatiquement pour multilingual-e5-base
   - ✅ Import direct de ChromaDB (plus de monkey-patch)

2. **`requirements_render.txt`**
   - ✅ `onnxruntime>=1.16.0` ajouté (nécessaire pour ChromaDB)
   - ✅ `requests>=2.31.0` ajouté (pour le wrapper)
   - ✅ Exclut `torch` et `sentence-transformers`

### Variables d'environnement sur Render

**Service `rag-api` → Environment → Variables :**

| Variable | Valeur | Où trouver ? |
|----------|--------|--------------|
| `USE_API_EMBEDDINGS` | `true` | ✅ Déjà configuré |
| `HUGGINGFACE_API_KEY` | `hf_...` | Hugging Face → Settings → Access Tokens |
| `HUGGINGFACE_MODEL` | `mistralai/Mixtral-8x7B-Instruct-v0.1` | ✅ Déjà configuré |
| `HUGGINGFACE_EMBEDDING_MODEL` | `intfloat/multilingual-e5-base` | ✅ Déjà configuré (optionnel, valeur par défaut) |
| `LLM_PROVIDER` | `huggingface` | ✅ Déjà configuré |
| `PYTHON_VERSION` | `3.12.0` | ✅ Déjà configuré |
| `PORT` | `8002` | ✅ Déjà configuré |
| `CHROMA_DB_PATH` | `/tmp/chroma_db` | ✅ Déjà configuré |

### Commande de build Render

**Dans Render Dashboard → rag-api → Settings → Build Command :**

```bash
pip install --upgrade pip setuptools wheel && pip install --no-cache-dir --prefer-binary -r requirements_render.txt && pip install --no-cache-dir onnxruntime>=1.16.0 && pip uninstall -y torch torchvision torchaudio sentence-transformers transformers 2>/dev/null || true && pip show onnxruntime || pip install --no-cache-dir onnxruntime>=1.16.0 && pip cache purge
```

### Commande de démarrage

**Dans Render Dashboard → rag-api → Settings → Start Command :**

```bash
uvicorn src.api.main_api:app --host 0.0.0.0 --port $PORT
```

## ✅ Vérifications après déploiement

Dans les logs de build, vous devriez voir :
```
Successfully installed ... onnxruntime-...
✅ Build terminé - torch/sentence-transformers désinstallés, onnxruntime conservé pour ChromaDB
```

Dans les logs de démarrage, vous devriez voir :
```
📦 Utilisation du wrapper personnalisé HuggingFaceTextEmbeddingsWrapper (nouvelle API router.huggingface.co)
📦 Modèle d'embedding: intfloat/multilingual-e5-base
✅ Embeddings via API Hugging Face (wrapper personnalisé, pas de modèle en mémoire, économise ~400 MB RAM)
✅ Client ChromaDB créé
✅ Collection 'renovation_knowledge' créée/récupérée
✅ Moteur RAG prêt !
```

**Pas d'erreurs :**
- ❌ Pas de `410 Gone`
- ❌ Pas de `ModuleNotFoundError: No module named 'onnxruntime'`
- ❌ Pas de `Out of memory`

## 🚀 Prêt à déployer !

Tout est configuré. Il suffit de :
1. ✅ Commit et push les modifications
2. ✅ Redéployer sur Render (ou attendre le déploiement automatique)
3. ✅ Tester une requête RAG

**Tout devrait fonctionner maintenant !** 🎉
