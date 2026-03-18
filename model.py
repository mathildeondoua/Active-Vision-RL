import torch
import torch.nn as nn
import torch.nn.functional as F

class GlimpseNetwork(nn.Module):
    """
    Ce réseau prend le patch et les coordonnées, et crée le vecteur intelligent g_t.
    """
    def __init__(self, patch_size=8, hidden_size=128):
        super(GlimpseNetwork, self).__init__()
        
        # Branche 1 : Traitement des pixels bruts
        self.patch_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(patch_size * patch_size, 64),
            nn.ReLU()
        )
        
        # Branche 2 : Traitement des coordonnées (x, y)
        self.loc_fc = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU()
        )
        
        # Fusion : On concatène et on mélange pour créer g_t
        self.combined_fc = nn.Sequential(
            nn.Linear(64 + 64, hidden_size),
            nn.ReLU()
        )

    def forward(self, patch, loc):
        # patch shape: (Batch, 1, 8, 8)
        # loc shape: (Batch, 2)
        
        phi_out = self.patch_fc(patch)
        loc_out = self.loc_fc(loc)
        
        # Concaténation le long de la dimension des caractéristiques (dim=1)
        merged = torch.cat([phi_out, loc_out], dim=1)
        
        # Sortie g_t
        g_t = self.combined_fc(merged)
        return g_t


class ClassificationHead(nn.Module):
    """
    L'Analyste : prend la mémoire récurrente (h_t) et sort les probabilités du chiffre.
    """
    def __init__(self, hidden_size=128, num_classes=10):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, h_t):
        # On ne met pas de Softmax explicite ici car la fonction de perte 
        # (CrossEntropyLoss) de PyTorch l'applique automatiquement en coulisses.
        logits = self.fc(h_t)
        return logits
