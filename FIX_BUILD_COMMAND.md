# Correction de la commande de build Render

## 🔍 Problème identifié

Le build command désinstalle `onnxruntime` juste après l'avoir installé, ce qui cause l'erreur lors de l'import de ChromaDB.

**Dans les logs, on voit :**
1. ✅ `onnxruntime` est installé avec succès
2. ❌ Le build command le désinstalle : "Uninstalling onnxruntime-1.23.2"
3. ❌ ChromaDB ne peut plus l'importer : "ModuleNotFoundError: No module named 'onnxruntime'"

## ✅ Solution

Le fichier `render.yaml` a été mis à jour pour :
1. **Ne PAS désinstaller `onnxruntime`**
2. Vérifier qu'`onnxruntime` est installé après la désinstallation de torch
3. Le réinstaller si nécessaire

## 📋 Commandes de build corrigées

### Dans Render Dashboard (si vous n'utilisez pas render.yaml)

**Dans Render Dashboard → rag-api → Settings → Build Command :**

```bash
pip install --upgrade pip setuptools wheel && pip install --no-cache-dir --prefer-binary -r requirements_render.txt && pip uninstall -y torch torchvision torchaudio sentence-transformers transformers 2>/dev/null || true && pip show onnxruntime || pip install onnxruntime>=1.16.0 && pip cache purge
```

### Si vous utilisez render.yaml (recommandé)

Le fichier `render.yaml` a été mis à jour automatiquement. Assurez-vous de :
1. ✅ Avoir le fichier `render.yaml` à la racine du repo
2. ✅ Utiliser Render Blueprint pour déployer depuis `render.yaml`

## 🎯 Ce qui est désinstallé vs conservé

| Package | Action | Raison |
|---------|--------|--------|
| `torch` | ✅ Désinstallé | ~400 MB économisés |
| `torchvision` | ✅ Désinstallé | Dépendance de torch |
| `torchaudio` | ✅ Désinstallé | Dépendance de torch |
| `sentence-transformers` | ✅ Désinstallé | ~100 MB économisés, utilise torch |
| `transformers` | ✅ Désinstallé | Dépendance de sentence-transformers |
| `onnxruntime` | ✅ **CONSERVÉ** | Nécessaire pour ChromaDB (~50 MB) |

## ✅ Vérification après déploiement

Dans les logs, vous devriez voir :
```
✅ Build terminé - torch/sentence-transformers désinstallés, onnxruntime conservé pour ChromaDB
```

Et lors du démarrage :
```
✅ Client ChromaDB créé
✅ Collection 'renovation_knowledge' créée/récupérée
```

Pas d'erreur "ModuleNotFoundError: No module named 'onnxruntime'".
