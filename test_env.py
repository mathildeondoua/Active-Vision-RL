import torchvision
import matplotlib.pyplot as plt
import numpy as np
from env import GlimpseEnv # Assure-toi que ton fichier s'appelle bien env.py

def main():
    print("1. Téléchargement des données MNIST...")
    # On télécharge juste les données brutes pour le test
    mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True)
    
    # On convertit les 100 premières images en numpy array pour l'environnement
    images = mnist.data.numpy()[:100]
    labels = mnist.targets.numpy()[:100]

    print("2. Création de l'environnement Gym...")
    env = GlimpseEnv(images, labels, patch_size=8, step_size=4) # Pas de 4 pour bouger plus vite
    
    obs, info = env.reset()
    print(f"Image cible : Chiffre {env.current_label}")
    print(f"Position initiale : x={env.x}, y={env.y}")
    print(f"Forme de l'observation : {obs.shape} (Attendu: 1, 8, 8)")

    # Préparation de l'affichage matplotlib
    fig, axes = plt.subplots(1, 6, figsize=(15, 3))
    
    # Affichage de l'image complète (pour nous, les humains)
    axes[0].imshow(env.current_image, cmap='gray')
    axes[0].set_title(f"Vraie Image ({env.current_label})")
    axes[0].plot(env.x, env.y, 'r+', markersize=10) # Croix rouge sur la position
    
    # Affichage du premier Glimpse
    axes[1].imshow(obs[0], cmap='gray', vmin=0, vmax=255)
    axes[1].set_title("Glimpse Initial")

    actions_names = ["Haut", "Bas", "Gauche", "Droite", "STOP"]
    
    print("\n3. Début de l'exploration aléatoire :")
    for i in range(4): # On fait 4 actions pour remplir le graphique
        action = env.action_space.sample() # Action au hasard
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Étape {i+1} | Action choisie : {actions_names[action]} | Nouvelle position : x={env.x}, y={env.y}")
        
        # Affichage du nouveau Glimpse
        axes[i+2].imshow(obs[0], cmap='gray', vmin=0, vmax=255)
        axes[i+2].set_title(f"Action: {actions_names[action]}")
        
        if terminated:
            print("L'agent a décidé de s'arrêter (STOP) !")
            break

    plt.tight_layout()
    plt.savefig('test_result.png')
    print("\nSuccès ! Le résultat visuel a été sauvegardé dans 'test_result.png'")

if __name__ == "__main__":
    main()
