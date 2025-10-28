# 🧠 Ouverture à la Recherche — Vision Transformer (CIFAR‑10)
**Classification & Inpainting par Vision Transformer — UCBL 2025**

Ce dépôt regroupe le **code**, les **documents** et les **supports** associés à un projet mené dans l’UE *Ouverture à la Recherche*.  
Objectif : explorer et implémenter un **Vision Transformer (ViT)** pour deux tâches sur **CIFAR‑10** :
- **Classification** d’images
- **Inpainting** (reconstruction d’images masquées)

> 📌 **Référence principale :** le **rapport final d’étude** décrit la méthodologie, les choix expérimentaux, l’analyse des résultats, les limites et perspectives. **Merci de vous y référer pour tous les détails scientifiques.**


---

## 📄 Documents (à lire en priorité)
- 📗 **Rapport final de l’étude** : `UE-INF1208M_Rapport_dépot_.pdf`
- 📘 **Cahier des charges (amont)** : `UE-INF1208M_CahierDesCharges_Cahier_des_charges_(2).pdf`
- 🎥 **Mini‑vidéo de vulgarisation** : `video.mp4`

---

## 📦 Arborescence & Rôles des fichiers

```
.
├── UE-INF1208M_CahierDesCharges_Cahier_des_charges_(2).pdf   # Cahier des charges initial
├── UE-INF1208M_Rapport_dépot_.pdf                             # Rapport final d’étude (source d’autorité)
├── train_vit_cifar10.py                                       # Entraînement ViT pour la classification CIFAR‑10
├── VIT_2.py                                                   # Entraînement ViT pour l’inpainting (masquage/reconstruction)
├── vit_inpainting_inference.py                                # Inférence d’inpainting (reconstruction d’images masquées)
├── ShowGraph.py                                               # Visualisations (ex. métriques, courbes, attention) selon implémentation
├── video.mp4                                                  # Mini‑vidéo (optionnelle)
└── README.md
```

---

## 🔧 Environnement (exemple)
- Python 3.10+
- PyTorch & TorchVision (versions compatibles CUDA/CPU selon votre machine)
- NumPy, Matplotlib
- (Optionnel) timm, scikit‑image, tqdm

**Installation (exemple minimal) :**
```bash
python -m venv .venv
source .venv/bin/activate    # (Windows: .venv\Scripts\activate)
pip install --upgrade pip
pip install torch torchvision  # ajoutez +cuXX si GPU CUDA
pip install numpy matplotlib tqdm
# si vos scripts l’utilisent :
pip install timm scikit-image
```

> **Dataset CIFAR‑10** : TorchVision sait le télécharger automatiquement au premier lancement si nécessaire.

---

## ▶️ Utilisation rapide

### 1) Classification (CIFAR‑10)
Entraînement d’un ViT de classification :
```bash
python train_vit_cifar10.py   --epochs 50   --batch-size 128   --lr 3e-4   --workers 4
```
- Les **hyperparamètres** (taille des patches, dimension du modèle, scheduler, poids de régularisation, etc.) sont à ajuster dans le script selon vos besoins.
- Les **sorties** (poids, courbes) sont enregistrées là où le script le prévoit.

### 2) Inpainting (apprentissage)
```bash
python VIT_2.py   --epochs 60   --batch-size 128   --mask-prob 0.4   --lr 1e-4
```
- Le script masque une partie des patches et apprend à **reconstruire** les contenus.  
- Ajustez la stratégie de **masquage** (aléatoire/structuré) et la **fonction de perte** dans le script si besoin.

### 3) Inpainting (inférence)
```bash
python vit_inpainting_inference.py   --weights path/to/checkpoint.pth   --input path/to/image_or_folder   --output outputs/
```
- Fournir un **checkpoint** entraîné et une **entrée** (image ou dossier).  
- Les reconstructions seront sauvegardées dans `--output`.

### 4) Visualisations
```bash
python ShowGraph.py   --logdir runs/exp1   --savefig figs/
```
- Selon l’implémentation, ce script peut tracer **courbes d’entraînement**, **matrices d’attention**, etc.

---

## 🧪 Reproductibilité & résultats
Les protocoles, métriques, comparaisons, courbes et résultats chiffrés sont **documentés dans le rapport final** :  
**`UE-INF1208M_Rapport_dépot_.pdf`** (à lire en priorité).

---

## 📝 Licence & citation
- Licence : à préciser selon vos besoins (MIT/BSD/Apache‑2.0, etc.).  
- Merci de citer l’UE **Ouverture à la Recherche — UCBL (2025)** et de référencer le **rapport final** dans toute réutilisation.

---

## 👥 Auteurs
Projet réalisé dans le cadre de l’UE **Ouverture à la Recherche** (UCBL, 2025) par l’équipe étudiante.  
Pour toute question technique, se référer d’abord au **rapport** puis aux **scripts** concernés.
