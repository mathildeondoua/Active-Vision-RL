import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GlimpseEnv(gym.Env):
    """
    Environnement personnalisé pour le modèle RAM (Prototype V1).
    L'agent navigue sur une image et observe un petit patch à chaque étape.
    """
    def __init__(self, images, labels, patch_size=8, step_size=2, max_steps=20):
        super(GlimpseEnv, self).__init__()
        
        # Données (numpy arrays de MNIST)
        self.images = images
        self.labels = labels
        
        # Paramètres
        self.patch_size = patch_size
        self.step_size = step_size
        self.max_steps = max_steps
        self.image_size = self.images.shape[-1] # Normalement 28 pour MNIST
        
        # Variables d'état interne
        self.current_image = None
        self.current_label = None
        self.x = 0
        self.y = 0
        self.step_count = 0
        
        # Espace d'action : 0=Haut, 1=Bas, 2=Gauche, 3=Droite, 4=STOP
        self.action_space = spaces.Discrete(5)
        
        # Espace d'observation : Un patch 8x8 en niveaux de gris (1 canal)
        self.observation_space = spaces.Dict({
            "patch": spaces.Box(low=0.0, high=1.0, shape=(1, self.patch_size, self.patch_size), dtype=np.float32),
            "loc": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        })

    def _get_patch(self):
        """Découpe l'image autour des coordonnées actuelles (x, y)."""
        half = self.patch_size // 2
        
        # On calcule les limites de la découpe
        x_min, x_max = self.x - half, self.x + half
        y_min, y_max = self.y - half, self.y + half
        
        # On gère les débordements (si l'agent regarde en dehors de l'image)
        # On pad (remplit) avec du noir (0)
        pad_top = max(0, -y_min)
        pad_bottom = max(0, y_max - self.image_size)
        pad_left = max(0, -x_min)
        pad_right = max(0, x_max - self.image_size)
        
        # On extrait la partie valide de l'image
        valid_y_min, valid_y_max = max(0, y_min), min(self.image_size, y_max)
        valid_x_min, valid_x_max = max(0, x_min), min(self.image_size, x_max)
        patch = self.current_image[valid_y_min:valid_y_max, valid_x_min:valid_x_max]
        
        # On applique le padding si nécessaire
        patch = np.pad(patch, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
        
        # Ajout de la dimension "canal" pour PyTorch (1, H, W)
        return np.expand_dims(patch, axis=0).astype(np.float32)

    def _get_obs(self):
        """Fonction utilitaire pour regrouper le patch et la localisation"""
        patch = self._get_patch()
        # On normalise les coordonnées entre -1 et 1 pour aider le réseau de neurones
        loc_x = (self.x / (self.image_size / 2.0)) - 1.0
        loc_y = (self.y / (self.image_size / 2.0)) - 1.0
        loc = np.array([loc_x, loc_y], dtype=np.float32)
        
        return {"patch": patch, "loc": loc}
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. Tirer une nouvelle image au hasard
        idx = self.np_random.integers(0, len(self.images))
        self.current_image = self.images[idx]
        self.current_label = self.labels[idx]
        
        # 2. Placer l'agent aléatoirement sur l'image
        self.x = self.np_random.integers(0, self.image_size)
        self.y = self.np_random.integers(0, self.image_size)
        self.step_count = 0
        
        # 3. On renvoie le dictionnaire
        obs = self._get_obs()
        return obs, {}
        
    def step(self, action):
        self.step_count += 1
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        # Mouvements
        if action == 0:   self.y -= self.step_size # Haut
        elif action == 1: self.y += self.step_size # Bas
        elif action == 2: self.x -= self.step_size # Gauche
        elif action == 3: self.x += self.step_size # Droite

        self.x = int(np.clip(self.x, 0, self.image_size))
        self.y = int(np.clip(self.y, 0, self.image_size))
        
        # Si mouvement, on applique une petite pénalité de temps
        if action in [0, 1, 2, 3]:
            reward = -0.05 
            
        # Action STOP
        elif action == 4:
            terminated = True
            info['guess_requested'] = True # Indique qu'il faut utiliser la tête de classification

        # Sécurité : fin de l'épisode si trop long
        if self.step_count >= self.max_steps:
            truncated = True

        obs = self._get_obs()

        return obs, reward, terminated, truncated, info
