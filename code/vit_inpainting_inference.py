import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Vérifier si CUDA est disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Utilisation du dispositif : {device}")

# 1. Définir le Modèle Vision Transformer pour l'Inpainting

class VisionTransformerInpainting(nn.Module):
    def __init__(self, image_size=32, patch_size=4, dim=128, depth=6, heads=8, mlp_dim=256, dropout=0.1):
        super(VisionTransformerInpainting, self).__init__()

        assert image_size % patch_size == 0, "L'image doit être divisible par la taille des patchs."

        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size  # 3 canaux (RGB)

        self.patch_size = patch_size
        self.dim = dim
        self.num_patches = num_patches

        # Projection linéaire des patchs
        self.patch_to_embedding = nn.Linear(patch_dim, dim)

        # Embeddings de position
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))

        # Couches Transformer
        self.transformer = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout)
            for _ in range(depth)
        ])

        # Tête de reconstruction
        self.to_patch = nn.Linear(dim, patch_dim)

    def forward(self, x, mask):
        batch_size, num_patches, _, _, _ = x.size()

        # Aplatir les patchs
        x = x.view(batch_size, num_patches, -1)  # [batch_size, num_patches, patch_dim]

        # Embedding des patchs
        x = self.patch_to_embedding(x)  # [batch_size, num_patches, dim]

        # Appliquer le masque aux embeddings
        mask = mask.unsqueeze(-1).to(x.device)  # [batch_size, num_patches, 1]
        x = x * mask  # Mettre à zéro les embeddings des patchs masqués

        # Ajouter les embeddings de position
        x = x + self.pos_embedding[:, :num_patches, :]

        # Transformer nécessite [sequence_length, batch_size, dim]
        x = x.permute(1, 0, 2)  # [num_patches, batch_size, dim]

        # Passer par les couches Transformer
        for layer in self.transformer:
            x = layer(x)

        # Repasser à [batch_size, num_patches, dim]
        x = x.permute(1, 0, 2)

        # Prédire les patchs
        x = self.to_patch(x)  # [batch_size, num_patches, patch_dim]

        # Reshape pour obtenir les patchs
        x = x.view(batch_size, num_patches, 3, self.patch_size, self.patch_size)

        return x

# 2. Charger le Modèle Enregistré

# Initialiser le modèle avec la même architecture que lors de l'entraînement
model = VisionTransformerInpainting().to(device)
# Charger les poids du modèle
model.load_state_dict(torch.load('./Data/vit_inpainting.pth', map_location=device))
model.eval()
print("Modèle chargé avec succès.")

# 3. Préparer les Données pour l'Inférence

# Paramètres
mask_ratio = 0.25  # Doit être le même que celui utilisé lors de l'entraînement

# Fonction pour générer des masques aléatoires
def generate_random_mask(num_patches, mask_ratio=0.25):
    mask = torch.ones(num_patches)
    num_masked = int(mask_ratio * num_patches)
    mask_indices = torch.randperm(num_patches)[:num_masked]
    mask[mask_indices] = 0
    return mask  # Tensor de taille [num_patches], avec des 0 (masqué) et 1 (visible)

# Dataset personnalisé pour appliquer des masques aux images
class MaskedCIFAR10(datasets.CIFAR10):
    def __init__(self, mask_ratio=0.25, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_ratio = mask_ratio
        self.patch_size = 4
        self.num_patches = (32 // self.patch_size) ** 2

    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        mask = generate_random_mask(self.num_patches, self.mask_ratio)

        # Diviser l'image en patchs
        img_patches = img.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
        img_patches = img_patches.contiguous().view(3, -1, self.patch_size, self.patch_size)
        img_patches = img_patches.permute(1, 0, 2, 3)  # [num_patches, 3, patch_size, patch_size]

        # Appliquer le masque
        masked_patches = img_patches.clone()
        masked_patches[mask == 0] = 0  # Masquer les patchs

        return masked_patches, img_patches, mask

# Préparer le dataset et le dataloader
transform = transforms.ToTensor()
test_dataset = MaskedCIFAR10(root='./data', train=False, download=True, transform=transform, mask_ratio=mask_ratio)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

# 4. Inférence et Enregistrement des Images

# Fonction pour reconstruire l'image à partir des patchs
def reconstruct_image(patches, image_size=32, patch_size=4):
    num_patches_per_row = image_size // patch_size
    patches = patches.view(num_patches_per_row, num_patches_per_row, 3, patch_size, patch_size)
    patches = patches.permute(2, 0, 3, 1, 4).contiguous()
    patches = patches.view(3, image_size, image_size)
    return patches

# Dossier pour enregistrer les images
import os
output_dir = 'inpainting_results'
os.makedirs(output_dir, exist_ok=True)

# Traiter plusieurs images du jeu de test
model.eval()
with torch.no_grad():
    for idx, (masked_patches, original_patches, mask) in enumerate(test_loader):
        masked_patches = masked_patches.to(device)
        original_patches = original_patches.to(device)
        mask = mask.to(device)

        outputs = model(masked_patches, mask)

        # Reconstruction des images
        masked_img = reconstruct_image(masked_patches[0].cpu())
        original_img = reconstruct_image(original_patches[0].cpu())
        reconstructed_img = reconstruct_image(outputs[0].cpu())

        # Enregistrer les images individuelles
        def save_image(img_tensor, filename):
            img = np.transpose(img_tensor.numpy(), (1, 2, 0))
            img = np.clip(img, 0, 1)
            plt.imsave(filename, img)

        # Enregistrer les images
        save_image(original_img, os.path.join(output_dir, f'img_{idx}_original.png'))
        save_image(masked_img, os.path.join(output_dir, f'img_{idx}_masked.png'))
        save_image(reconstructed_img, os.path.join(output_dir, f'img_{idx}_reconstructed.png'))

        # Optionnel : Afficher une progression
        if (idx + 1) % 10 == 0:
            print(f'{idx + 1} images traitées')

        # Limiter le nombre d'images traitées pour cet exemple
        if idx >= 49:  # Traiter les 50 premières images
            break

print(f"Les images ont été enregistrées dans le dossier '{output_dir}'.")
