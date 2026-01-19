# Correction de l'erreur ChromaDB avec onnxruntime

## 🔍 Problème

ChromaDB essaie d'utiliser `DefaultEmbeddingFunction()` qui nécessite `onnxruntime`, mais on a désinstallé `onnxruntime` pour économiser de la mémoire (~400 MB).

L'erreur se produit lors de l'import de ChromaDB car il essaie d'initialiser l'embedding function par défaut dans la définition de la classe `Collection`.

## ✅ Solution implémentée

### Monkey-patch de `onnxruntime`

Avant d'importer ChromaDB, on crée un module mock pour `onnxruntime` dans `sys.modules`. Cela empêche ChromaDB de tenter d'importer le vrai `onnxruntime`.

### Code ajouté dans `query_engine.py` :

```python
# Créer un module mock pour onnxruntime AVANT que ChromaDB ne l'importe
mock_onnx = types.ModuleType('onnxruntime')
mock_onnx.InferenceSession = None
sys.modules['onnxruntime'] = mock_onnx
```

### Collection ChromaDB sans embedding function

Lors de la création de la collection, on spécifie explicitement `embedding_function=None` car on utilise les embeddings de llama-index :

```python
chroma_collection = db.get_or_create_collection(
    COLLECTION_NAME,
    embedding_function=None  # Pas d'embedding function, on utilise celle de llama-index
)
```

## 🎯 Résultat attendu

Après redéploiement, ChromaDB devrait s'importer sans erreur et utiliser les embeddings fournis par llama-index (via notre wrapper personnalisé).

## ⚠️ Alternative si le problème persiste

Si le monkey-patch ne fonctionne pas, il faudra installer `onnxruntime` (version CPU minimale) :

```bash
pip install onnxruntime
```

Mais cela ajoutera ~50-100 MB à la mémoire utilisée.
