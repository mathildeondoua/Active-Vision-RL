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
    images = mnist.data.numpy()[:10000] # On prend 1000 images pour tester vite
    labels = mnist.targets.numpy()[:10000]
    
    env = GlimpseEnv(images, labels, patch_size=8, step_size=4, max_steps=20)
    
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
        
        # --- NOUVEAU : Calcul de l'entropie initiale (Tour à vide) ---
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
            
            # Conversion des observations pour PyTorch (ajout dimension Batch)
            patch = torch.tensor(obs['patch']).unsqueeze(0)
            loc = torch.tensor(obs['loc']).unsqueeze(0)
            
            # 1. L'Agent réfléchit
            class_logits, action_probs, h_t, c_t = agent(patch, loc, h_t, c_t)
            
            # 2. Le Pilote choisit une action (Exploration RL)
            action, log_prob = agent.get_action(action_probs[0])
            
            # 3. L'environnement réagit au mouvement
            next_obs, env_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # --- CALCUL DE LA RÉCOMPENSE INTELLIGENTE ---
            current_entropy = compute_entropy(class_logits)
            entropy_trajectory.append(current_entropy.item())
            
            ENTROPY_SCALE = 5
            
            if action < 4 and not truncated: 
                # C'est un mouvement ! 
                entropy_drop = (last_entropy - current_entropy).item()
                reward = entropy_drop * ENTROPY_SCALE
            else:
                # C'est l'action STOP ! L'agent rend sa copie.
                prediction = torch.argmax(class_logits, dim=-1).item()
                true_label = env.current_label
                if prediction == true_label:
                    reward = +2.0 # Jackpot !
                else:
                    reward = -1.0 # Mauvaise réponse
                
                # CORRECTION CRUCIALE : On calcule l'erreur du classifieur 
                # UNIQUEMENT sur sa décision finale !
                true_label_tensor = torch.tensor([env.current_label], dtype=torch.long)
                loss_c = criterion_class(class_logits, true_label_tensor)
                class_losses.append(loss_c)
            
            # On stocke pour l'apprentissage de fin d'épisode
            log_probs.append(log_prob)
            rewards.append(reward)
            
            last_entropy = current_entropy
            obs = next_obs
            
        # --- MISE À JOUR DU CERVEAU (Apprentissage à la fin de l'image) ---
        # Calcul du rendement total de l'épisode (Sont-ils de bons choix ?)
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
        
        # --- NOUVEAU : Affichage détaillé des logs ---
        if (episode + 1) % 50 == 0:
            traj_str = " -> ".join([f"{e:.2f}" for e in entropy_trajectory])
            print(f"Épisode {episode+1}/{num_episodes} | Étapes: {steps} | R Totale: {R:.2f} | Action Finale: {action}")
            print(f"   Trajectoire Entropie : [{traj_str}]")
            print("-" * 60)
            
    torch.save(agent.state_dict(), 'agent_weights.pth')
    print("Entraînement terminé !")

if __name__ == "__main__":
    main()
