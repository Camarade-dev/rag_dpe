# Instructions pour corriger la commande de build Render

## ⚠️ PROBLÈME ACTUEL

Le build command désinstalle `onnxruntime` car pip supprime automatiquement les dépendances inutilisées quand on désinstalle `transformers`/`sentence-transformers`.

## ✅ SOLUTION

Il faut réinstaller `onnxruntime` **AVANT** de désinstaller `transformers`/`sentence-transformers`, ou **APRÈS** leur désinstallation.

## 📋 Commande de build à utiliser

### Si vous utilisez Render Dashboard (configuration manuelle)

**Allez dans Render Dashboard → rag-api → Settings → Build Command**

**Copiez-collez cette commande EXACTEMENT :**

```bash
pip install --upgrade pip setuptools wheel && pip install --no-cache-dir --prefer-binary -r requirements_render.txt && pip install --no-cache-dir onnxruntime>=1.16.0 && pip uninstall -y torch torchvision torchaudio sentence-transformers transformers 2>/dev/null || true && pip show onnxruntime || pip install --no-cache-dir onnxruntime>=1.16.0 && pip cache purge
```

### Explication étape par étape

1. `pip install --upgrade pip setuptools wheel` - Met à jour pip
2. `pip install --no-cache-dir --prefer-binary -r requirements_render.txt` - Installe toutes les dépendances
3. `pip install --no-cache-dir onnxruntime>=1.16.0` - **Réinstalle explicitement onnxruntime** (important !)
4. `pip uninstall -y torch torchvision torchaudio sentence-transformers transformers` - Désinstalle les packages lourds
5. `pip show onnxruntime || pip install --no-cache-dir onnxruntime>=1.16.0` - Vérifie et réinstalle si nécessaire
6. `pip cache purge` - Nettoie le cache

### Si vous utilisez render.yaml (Blueprint)

Le fichier `render.yaml` a été mis à jour automatiquement. Il suffit de :
1. ✅ Commit et push les modifications
2. ✅ Redéployer depuis Render Dashboard (ou attendre le déploiement automatique)

## ✅ Vérification après déploiement

Dans les logs de build, vous devriez voir :
```
Successfully installed ... onnxruntime-...
✅ Build terminé - torch/sentence-transformers désinstallés, onnxruntime conservé pour ChromaDB
```

**PAS de ligne :**
```
Uninstalling onnxruntime-...
```

Dans les logs de démarrage, vous devriez voir :
```
✅ Client ChromaDB créé
✅ Collection 'renovation_knowledge' créée/récupérée
```

**PAS d'erreur :**
```
ModuleNotFoundError: No module named 'onnxruntime'
```
