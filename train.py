import torch
import torch.optim as optim
import torch.nn as nn
from env import GlimpseEnv
from agent import RAMAgent, compute_entropy
import torchvision
import numpy as np

def main():
    print("1. Préparation des données et de l'Environnement...")
    mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True)
    images = mnist.data.numpy()[:10000] # On prend 10000 images
    labels = mnist.targets.numpy()[:10000]
    
    # RAPPEL : step_size=2 pour pouvoir se déplacer un peu, max_steps=20
    env = GlimpseEnv(images, labels, patch_size=8, step_size=2, max_steps=20)
    
    print("2. Initialisation de l'Agent et de l'Optimiseur...")
    agent = RAMAgent(patch_size=8, hidden_size=128, num_classes=10, num_actions=5)
    optimizer = optim.Adam(agent.parameters(), lr=1e-3)
    criterion_class = nn.CrossEntropyLoss() # Pour entraîner l'Analyste
    
    num_episodes = 15000
    
    print("3. Début de l'entraînement hybride !")
    for episode in range(num_episodes):
        obs, _ = env.reset()
        
        # On initialise les mémoires du LSTM à zéro au début de chaque image
        h_t = torch.zeros(1, agent.hidden_size)
        c_t = torch.zeros(1, agent.hidden_size)
        
        log_probs = []
        rewards = []
        class_losses = []
        
        # --- Calcul de l'entropie initiale (Tour à vide) ---
        patch_init = torch.tensor(obs['patch']).unsqueeze(0)
        loc_init = torch.tensor(obs['loc']).unsqueeze(0)
        with torch.no_grad():
            logits_init, _, _, _ = agent(patch_init, loc_init, h_t, c_t)
            last_entropy = compute_entropy(logits_init)
            
        # Initialisation des compteurs pour l'affichage
        steps = 0
        entropy_trajectory = [last_entropy.item()]
        
        done = False
        while not done:
            steps += 1 # On compte le déplacement
            
            # Conversion des observations pour PyTorch
            patch = torch.tensor(obs['patch']).unsqueeze(0)
            loc = torch.tensor(obs['loc']).unsqueeze(0)
            
            # 1. L'Agent réfléchit (CORRIGÉ : patch, loc)
            class_logits, action_probs, h_t, c_t = agent(patch, loc, h_t, c_t)
            action, log_prob = agent.get_action(action_probs[0])
            next_obs, env_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 2. Suivi de l'entropie pour les logs (CORRIGÉ)
            current_entropy = compute_entropy(class_logits)
            entropy_trajectory.append(current_entropy.item())
            
            # 3. --- LA NOUVELLE RÉCOMPENSE PURE ---
            if action < 4 and not truncated: 
                # Les mouvements ne rapportent rien d'autre que la pénalité de temps
                # pour l'encourager à être rapide.
                reward = env_reward 
            else:
                # L'heure de vérité.
                prediction = torch.argmax(class_logits, dim=-1).item()
                true_label = env.current_label
                if prediction == true_label:
                    reward = 1.0  # Jackpot
                else:
                    reward = -1.0 # Echec
                
                # Le réseau de classification s'entraîne SEULEMENT à la fin
                true_label_tensor = torch.tensor([env.current_label], dtype=torch.long)
                loss_c = criterion_class(class_logits, true_label_tensor)
                class_losses.append(loss_c)
            
            log_probs.append(log_prob)
            rewards.append(reward)
            obs = next_obs
            
        # --- MISE À JOUR DU CERVEAU (Apprentissage à la fin de l'image) ---
        R = sum(rewards)
        
        # Formule RL classique : REINFORCE (Policy Gradient)
        policy_loss = []
        for lp in log_probs:
            policy_loss.append(-lp * R) 
        policy_loss = torch.stack(policy_loss).sum()
        
        # L'Analyste (Supervisé) + Le Pilote (RL)
        total_loss = sum(class_losses) + policy_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # --- Affichage détaillé des logs ---
        if (episode + 1) % 500 == 0:
            traj_str = " -> ".join([f"{e:.2f}" for e in entropy_trajectory])
            print(f"Épisode {episode+1}/{num_episodes} | Étapes: {steps} | R Totale: {R:.2f} | Action Finale: {action}")
            print(f"   Trajectoire Entropie : [{traj_str}]")
            print("-" * 60)
            
    torch.save(agent.state_dict(), 'agent_weights.pth')
    print("Entraînement terminé !")

if __name__ == "__main__":
    main()
