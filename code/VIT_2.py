import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Vérifier si CUDA est disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Utilisation du dispositif : {device}")

# 1. Préparer le Dataset avec des Patchs Masqués

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

# 2. Définir le Modèle Vision Transformer pour l'Inpainting

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

# 3. Préparer l'Entraînement

# Paramètres
mask_ratio = 0.25  # Pourcentage de patchs masqués

# Préparer les datasets et les dataloaders
transform = transforms.ToTensor()

train_dataset = MaskedCIFAR10(root='./data', train=True, download=True, transform=transform, mask_ratio=mask_ratio)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

test_dataset = MaskedCIFAR10(root='./data', train=False, download=True, transform=transform, mask_ratio=mask_ratio)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

# Initialiser le modèle
model = VisionTransformerInpainting().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 4. Boucle d'Entraînement

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for masked_patches, original_patches, mask in train_loader:
        masked_patches = masked_patches.to(device)
        original_patches = original_patches.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()

        # Forward
        outputs = model(masked_patches, mask)

        # Calculer la perte uniquement sur les patchs masqués
        loss = criterion(outputs[mask == 0], original_patches[mask == 0])

        # Backward et optimisation
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * masked_patches.size(0)

    epoch_loss = running_loss / len(train_dataset)
    print(f'Époque [{epoch+1}/{num_epochs}], Perte : {epoch_loss:.6f}')

# 5. Évaluation et Visualisation

# Fonction pour reconstruire l'image à partir des patchs
# Fonction pour reconstruire l'image à partir des patchs
def reconstruct_image(patches, image_size=32, patch_size=4):
    num_patches_per_row = image_size // patch_size  # Par exemple, 32 // 4 = 8
    # patches a la forme [num_patches, 3, patch_size, patch_size]
    patches = patches.view(num_patches_per_row, num_patches_per_row, 3, patch_size, patch_size)
    # Permuter pour obtenir [3, num_patches_per_row, patch_size, num_patches_per_row, patch_size]
    patches = patches.permute(2, 0, 3, 1, 4).contiguous()
    # Fusionner les dimensions pour obtenir [3, image_size, image_size]
    patches = patches.view(3, image_size, image_size)
    return patches


# Visualiser les résultats
model.eval()
with torch.no_grad():
    for masked_patches, original_patches, mask in test_loader:
        masked_patches = masked_patches.to(device)
        original_patches = original_patches.to(device)
        mask = mask.to(device)

        outputs = model(masked_patches, mask)

        # Sélectionner le premier exemple du batch
        idx = 0
        masked_img = reconstruct_image(masked_patches[idx].cpu())
        original_img = reconstruct_image(original_patches[idx].cpu())
        reconstructed_img = reconstruct_image(outputs[idx].cpu())

        # Afficher les images
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(np.transpose(original_img.numpy(), (1, 2, 0)))
        axs[0].set_title('Image Originale')
        axs[0].axis('off')

        axs[1].imshow(np.transpose(masked_img.numpy(), (1, 2, 0)))
        axs[1].set_title('Image avec Patchs Masqués')
        axs[1].axis('off')

        axs[2].imshow(np.transpose(reconstructed_img.numpy(), (1, 2, 0)))
        axs[2].set_title('Image Reconstituée')
        axs[2].axis('off')

        plt.show()
        break  # On ne visualise qu'un batch

# 6. Enregistrement du Modèle

# Enregistrer le modèle
torch.save(model.state_dict(), './Data/vit_inpainting.pth')
print("Modèle enregistré sous './Data/vit_inpainting.pth'")
