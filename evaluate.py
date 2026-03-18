import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import torchvision

from env import GlimpseEnv
from agent import RAMAgent

def evaluate_agent(agent, env, num_episodes=1000):
    print(f"Début de l'évaluation sur {num_episodes} images...")
    
    # On passe le modèle en mode évaluation (désactive le dropout, etc.)
    agent.eval()
    
    all_trues = []
    all_preds = []
    all_steps = []
    
    with torch.no_grad(): # On désactive le calcul des gradients pour aller plus vite
        for i in range(num_episodes):
            obs, _ = env.reset()
            h_t = torch.zeros(1, agent.hidden_size)
            c_t = torch.zeros(1, agent.hidden_size)
            
            done = False
            steps = 0
            
            while not done:
                patch = torch.tensor(obs['patch']).unsqueeze(0)
                loc = torch.tensor(obs['loc']).unsqueeze(0)
                
                class_logits, action_probs, h_t, c_t = agent(patch, loc, h_t, c_t)
                
                # En évaluation, on ne tire plus au hasard ! 
                # On prend l'action avec la plus haute probabilité (Exploitation pure)
                action = torch.argmax(action_probs[0]).item()
                
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                steps += 1
                
                # Si l'agent décide de s'arrêter (STOP) ou s'il atteint la limite de temps
                if action == 4 or done:
                    prediction = torch.argmax(class_logits, dim=-1).item()
                    all_preds.append(prediction)
                    all_trues.append(env.current_label)
                    all_steps.append(steps)
                    break

    # --- CALCUL DES MÉTRIQUES ---
    global_acc = accuracy_score(all_trues, all_preds)
    avg_steps = np.mean(all_steps)
    
    print("-" * 30)
    print(f"Précision Globale (Accuracy) : {global_acc * 100:.2f}%")
    print(f"Nombre moyen de Glimpses   : {avg_steps:.2f} étapes")
    print("-" * 30)

    # --- GÉNÉRATION DES GRAPHIQUES ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # 1. Matrice de Confusion
    cm = confusion_matrix(all_trues, all_preds, labels=range(10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
    axes[0].set_title("Matrice de Confusion")
    axes[0].set_xlabel("Prédiction de l'Agent")
    axes[0].set_ylabel("Vrai Chiffre")

    # 2. Distribution du nombre de Glimpses
    sns.histplot(all_steps, bins=range(1, env.max_steps + 2), discrete=True, ax=axes[1], color='coral')
    axes[1].set_title("Distribution des Glimpses avant décision")
    axes[1].set_xlabel("Nombre de Glimpses")
    axes[1].set_ylabel("Nombre d'images")

    # 3. Précision par Classe
    acc_per_class = cm.diagonal() / cm.sum(axis=1)
    sns.barplot(x=list(range(10)), y=acc_per_class, ax=axes[2], palette='viridis')
    axes[2].set_title("Précision par Chiffre (Accuracy per Class)")
    axes[2].set_xlabel("Chiffre")
    axes[2].set_ylabel("Précision")
    axes[2].set_ylim(0, 1.0)
    
    # Affichage des pourcentages sur les barres
    for i, v in enumerate(acc_per_class):
        axes[2].text(i, v + 0.02, f"{v*100:.0f}%", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('evaluation_results.png')
    plt.show()

# --- SCRIPT POUR LANCER L'ÉVALUATION DANS COLAB ---
if __name__ == "__main__":
    print("Chargement des données de TEST MNIST...")
    # Attention : on charge bien le dataset de TEST cette fois, pas celui d'entraînement !
    test_mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True)
    test_images = test_mnist.data.numpy()
    test_labels = test_mnist.targets.numpy()
    
    # On recrée un environnement de test
    test_env = GlimpseEnv(test_images, test_labels, patch_size=8, step_size=4, max_steps=10)
    
    # On instancie l'agent vierge
    agent = RAMAgent(patch_size=8, hidden_size=128, num_classes=10, num_actions=5)
    # On lui charge le cerveau entraîné
    agent.load_state_dict(torch.load('agent_weights.pth'))
    
    # On lance l'évaluation
    evaluate_agent(agent, test_env, num_episodes=1000)
