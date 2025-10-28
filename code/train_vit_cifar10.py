import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

# Vérifier si CUDA est disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Utilisation du dispositif : {device}")

# 1. Préparation du Dataset
# Définir les transformations avec data augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.ToTensor()
])

# Charger le dataset CIFAR-10
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# Créer les DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

# 2. Définition du Modèle Vision Transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, nhead, dim_feedforward, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=nhead, dropout=dropout)
        self.linear1 = nn.Linear(dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # src shape: [sequence_length, batch_size, dim]
        src2, attn_output_weights = self.self_attn(src, src, src)
        self.attn_output_weights = attn_output_weights  # Stocker les poids d'attention
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class VisionTransformer(nn.Module):
    def __init__(self, image_size=32, patch_size=4, num_classes=10, dim=64, depth=6, heads=8, mlp_dim=128, dropout=0.1):
        super(VisionTransformer, self).__init__()

        assert image_size % patch_size == 0, "L'image doit être divisible par la taille des patchs."

        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size  # 3 canaux (RGB)

        self.patch_size = patch_size
        self.dim = dim
        self.num_patches = num_patches

        # Projection linéaire des patchs
        self.patch_to_embedding = nn.Linear(patch_dim, dim)

        # Embeddings de position
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        # Token de classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Couches Transformer
        self.transformer = nn.ModuleList([
            TransformerEncoderLayer(dim=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout)
            for _ in range(depth)
        ])

        # Tête de classification
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)

        # Diviser l'image en patchs
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().reshape(batch_size, 3, -1, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 1, 3, 4)  # [batch_size, num_patches, 3, patch_size, patch_size]
        num_patches = x.size(1)

        # Aplatir les patchs
        x = x.reshape(batch_size, num_patches, -1)  # [batch_size, num_patches, patch_dim]

        # Embedding des patchs
        x = self.patch_to_embedding(x)  # [batch_size, num_patches, dim]

        # Ajouter le token de classification
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [batch_size, num_patches + 1, dim]

        # Ajouter les embeddings de position
        x = x + self.pos_embedding[:, :num_patches + 1, :]

        # Transformer nécessite [sequence_length, batch_size, dim]
        x = x.permute(1, 0, 2)  # [sequence_length, batch_size, dim]

        # Passer par les couches Transformer
        for layer in self.transformer:
            x = layer(x)

        # Extraire le token de classification
        x = x[0]  # [batch_size, dim]

        # Passer par la tête de classification
        x = self.mlp_head(x)  # [batch_size, num_classes]

        return x

# 3. Entraînement du Modèle
# Initialiser le modèle
model = VisionTransformer().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)

# Fonction pour calculer la précision
def calculate_accuracy(outputs, labels):
    _, predicted = outputs.max(1)
    total = labels.size(0)
    correct = predicted.eq(labels).sum().item()
    return 100. * correct / total

# Boucle d'entraînement
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zéro gradients
        optimizer.zero_grad()

        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward
        loss.backward()
        optimizer.step()

        # Statistiques
        running_loss += loss.item() * inputs.size(0)
        running_accuracy += calculate_accuracy(outputs, labels) * inputs.size(0)

    epoch_loss = running_loss / len(train_dataset)
    epoch_accuracy = running_accuracy / len(train_dataset)
    print(f'Époque [{epoch+1}/{num_epochs}], Perte : {epoch_loss:.4f}, Précision : {epoch_accuracy:.2f}%')

    # Validation
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0

    with torch.no_grad():
        for val_inputs, val_labels in test_loader:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            val_outputs = model(val_inputs)
            val_loss += criterion(val_outputs, val_labels).item() * val_inputs.size(0)
            val_accuracy += calculate_accuracy(val_outputs, val_labels) * val_inputs.size(0)

    val_epoch_loss = val_loss / len(test_dataset)
    val_epoch_accuracy = val_accuracy / len(test_dataset)
    print(f'Validation Perte : {val_epoch_loss:.4f}, Précision : {val_epoch_accuracy:.2f}%')

# 4. Visualisation des Cartes d'Attention
# Sélectionner une image du jeu de test
images, labels = next(iter(test_loader))
image = images[0:1].to(device)  # Prendre une seule image
label = labels[0]

# Passer l'image à travers le modèle
model.eval()
with torch.no_grad():
    output = model(image)

# Récupérer les poids d'attention du dernier bloc
attn_weights = model.transformer[-1].attn_output_weights  # [batch_size, sequence_length, sequence_length]

# Moyenne sur les têtes d'attention
attn_weights = attn_weights.mean(dim=0)  # [sequence_length, sequence_length]

# Visualiser les poids d'attention
plt.figure(figsize=(8, 6))
sns.heatmap(attn_weights.cpu(), cmap='viridis')
plt.title('Carte d\'attention moyenne sur les têtes')
plt.xlabel('Patchs')
plt.ylabel('Patchs')
plt.show()

# Afficher l'image originale
def show_image(img, title):
    img = img.cpu().numpy().transpose(1, 2, 0)
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

show_image(image[0], f'Image de test - Label : {label}')

# 5. Enregistrement du Modèle
# Enregistrer le modèle
torch.save(model.state_dict(), './Data/vit_cifar10.pth')
print("Modèle enregistré sous './Data/vit_cifar10.pth'")
