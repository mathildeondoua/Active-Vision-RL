import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torchvision
from env import GlimpseEnv
from agent import RAMAgent, compute_entropy

def debug_viz():
    # 1. Setup
    mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True)
    env = GlimpseEnv(mnist.data.numpy(), mnist.targets.numpy(), patch_size=8, step_size=4, max_steps=20)
    agent = RAMAgent(patch_size=8, hidden_size=128, num_classes=10, num_actions=5)
    
    try:
        agent.load_state_dict(torch.load('agent_weights.pth', map_location='cpu'))
        print("Poids chargés avec succès.")
    except:
        print("Attention : Aucun poids trouvé, visualisation d'un agent aléatoire.")

    agent.eval()
    obs, _ = env.reset()
    h_t, c_t = torch.zeros(1, 128), torch.zeros(1, 128)
    
    history = {'pos': [(env.x, env.y)], 'patches': [obs['patch'][0][0]], 'ents': []}

    # Simulation
    for _ in range(10):
        with torch.no_grad():
            p, l = torch.tensor(obs['patch']).unsqueeze(0), torch.tensor(obs['loc']).unsqueeze(0)
            logits, probs, h_t, c_t = agent(p, l, h_t, c_t)
            action = torch.argmax(probs[0]).item()
            history['ents'].append(compute_entropy(logits).item())
            
        obs, _, done, trunc, _ = env.step(action)
        history['pos'].append((env.x, env.y))
        history['patches'].append(obs['patch'][0][0])
        if done or trunc or action == 4: break

    # Plot
    n = len(history['patches'])
    fig, axes = plt.subplots(1, n + 1, figsize=(3*n, 3))
    axes[0].imshow(env.current_image, cmap='gray')
    for i, (px, py) in enumerate(history['pos']):
        rect = patches.Rectangle((px-4, py-4), 8, 8, linewidth=1, edgecolor='red', facecolor='none')
        axes[0].add_patch(rect)
        axes[0].text(px, py, str(i), color='red', fontsize=8)
    
    for i in range(n):
        axes[i+1].imshow(history['patches'][i], cmap='gray', vmin=0, vmax=1)
        axes[i+1].set_title(f"T={i}")
    
    plt.savefig('debug_path.png')
    print("Image sauvegardée : debug_path.png")

if __name__ == "__main__":
    debug_viz()
