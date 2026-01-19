# Correction de l'erreur 410 Gone avec wrapper personnalisé

## 🔧 Solution implémentée

Un wrapper personnalisé `HuggingFaceTextEmbeddingsWrapper` a été créé pour contourner le problème de l'erreur 410 Gone en utilisant directement `requests` avec les bons endpoints Hugging Face.

## 🎯 Fonctionnalités

1. **Utilisation du nouveau router Hugging Face** : `router.huggingface.co`
2. **Fallback automatique** : Si le nouveau router échoue, essaie l'ancien endpoint
3. **Gestion robuste des erreurs** : Gère différents formats de réponse
4. **Compatible avec llama-index** : Implémente `BaseEmbedding` de llama-index

## 📋 Changements dans le code

### Nouveau fichier/classe
- `HuggingFaceTextEmbeddingsWrapper` dans `src/rag_core/query_engine.py`
  - Utilise `requests` pour appeler directement l'API Hugging Face
  - Essaie d'abord `router.huggingface.co` (nouveau router)
  - Fallback vers `api-inference.huggingface.co` si nécessaire

### Priorité d'utilisation
1. **Wrapper personnalisé** (priorité) - utilise le nouveau router
2. Classes llama-index (fallback) - si le wrapper échoue

## 📦 Dépendances ajoutées

- `requests>=2.31.0` dans `requirements_render.txt`

## 🚀 Déploiement

Après redéploiement, vous devriez voir dans les logs :
```
📦 Utilisation du wrapper personnalisé HuggingFaceTextEmbeddingsWrapper (nouvelle API router.huggingface.co)
✅ Embeddings via API Hugging Face (wrapper personnalisé, pas de modèle en mémoire, économise ~400 MB RAM)
```

## ✅ Avantages

- ✅ Corrige l'erreur 410 Gone
- ✅ Utilise le nouveau router Hugging Face
- ✅ Pas de modèle chargé en RAM (économise ~400 MB)
- ✅ Compatible avec la limite de 512 MB de Render
- ✅ Fallback automatique en cas d'échec

## 🔍 Dépannage

Si l'erreur persiste :
1. Vérifiez que `requests` est installé
2. Vérifiez que `HUGGINGFACE_API_KEY` est correctement configurée
3. Vérifiez les logs pour voir quelle URL est utilisée
