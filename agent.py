import torch
import torch.nn as nn
from torch.distributions import Categorical
from model import GlimpseNetwork, ClassificationHead # On importe ton code !

class RAMAgent(nn.Module):
    def __init__(self, patch_size=8, hidden_size=128, num_classes=10, num_actions=5):
        super(RAMAgent, self).__init__()
        
        # 1. Le Capteur Intelligent (g_t)
        self.sensor = GlimpseNetwork(patch_size, hidden_size)
        
        # 2. La Mémoire (h_t) - On utilise LSTMCell pour avancer pas à pas
        self.rnn = nn.LSTMCell(hidden_size, hidden_size)
        
        # 3. La Tête de Classification (L'Analyste)
        self.classifier = ClassificationHead(hidden_size, num_classes)
        
        # 4. La Tête RL (Le Pilote)
        # Sort les probabilités des 5 actions (Haut, Bas, Gauche, Droite, STOP)
        self.policy = nn.Sequential(
            nn.Linear(hidden_size, num_actions),
            nn.Softmax(dim=-1)
        )
        
        self.hidden_size = hidden_size

    def forward(self, patch, loc, h_t, c_t):
        """
        Un 'forward' correspond à un seul coup d'oeil (un step).
        """
        # 1. On fabrique g_t
        g_t = self.sensor(patch, loc)
        
        # 2. On met à jour la mémoire
        h_t, c_t = self.rnn(g_t, (h_t, c_t))
        
        # 3. Les deux têtes réfléchissent à partir de la nouvelle mémoire h_t
        class_logits = self.classifier(h_t)
        action_probs = self.policy(h_t)
        
        return class_logits, action_probs, h_t, c_t

    def get_action(self, action_probs):
        """
        Tire au sort l'action en respectant les probabilités (Exploration RL)
        et renvoie le log_prob (nécessaire pour la formule mathématique du RL).
        """
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

def compute_entropy(logits):
    """
    Calcule l'entropie d'une prédiction. 
    Plus le chiffre est proche de 0, plus le modèle est sûr de lui.
    """
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    # Formule de l'entropie de Shannon : H = - sum(p * log(p))
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy
