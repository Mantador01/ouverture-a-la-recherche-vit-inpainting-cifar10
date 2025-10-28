
# Vision Transformer Project

## Description

Ce projet implémente un Vision Transformer (ViT) pour deux tâches principales :
1. **Classification d'images** : Entraînement sur le dataset CIFAR-10.
2. **Inpainting d'images** : Reconstruction d'images masquées.

Les scripts inclus permettent d'entraîner, d'évaluer, et d'effectuer des inférences avec des modèles ViT.

---

## Prérequis

Assurez-vous d'avoir Python (>=3.8) installé et exécutez la commande suivante pour installer les dépendances nécessaires :

```bash
pip install torch torchvision matplotlib seaborn numpy
```

---

## Fichiers inclus

- **`vit_inpainting_inference.py`** : Script d'inférence pour l'inpainting d'images avec un modèle pré-entraîné.
- **`VIT_2.py`** : Script d'entraînement d'un Vision Transformer pour l'inpainting.
- **`train_vit_cifar10.py`** : Script d'entraînement d'un modèle ViT pour la classification d'images sur CIFAR-10.
- **`ShowGraph.py`** : Génération de visualisations des cartes d'attention du modèle.

---

## Instructions d'utilisation

### 1. Préparation des données

Les scripts téléchargent automatiquement le dataset CIFAR-10 et le stockent dans un répertoire nommé `./data`.

### 2. Entraînement

#### Pour l'inpainting :
Exécutez la commande suivante pour entraîner un modèle Vision Transformer pour l'inpainting :
```bash
python VIT_2.py
```

#### Pour la classification :
Exécutez la commande suivante pour entraîner un modèle Vision Transformer pour la classification CIFAR-10 :
```bash
python train_vit_cifar10.py
```

### 3. Inférence

#### Inpainting :
Pour effectuer de l'inpainting avec un modèle pré-entraîné :
```bash
python vit_inpainting_inference.py
```

#### Visualisation des cartes d'attention :
Pour générer des visualisations des cartes d'attention :
```bash
python ShowGraph.py
```

---

## Résultats

1. Les résultats d'inpainting sont enregistrés dans un dossier `inpainting_results`.
2. Les visualisations des cartes d'attention sont sauvegardées dans les fichiers `attention_map.png` et `original_image.png`.

---

## Remarques

- Les modèles entraînés sont sauvegardés sous les noms `vit_inpainting.pth` et `vit_cifar10.pth`.
- Pour des performances optimales, utilisez un environnement avec support CUDA.
