# agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import QuadrupedActorModel, QuadrupedCriticModel
from physics_env.config import DEBUG_RL_AGENT
import torch.nn.functional as F
from collections import deque
import random
import time
import os

class QuadrupedAgent:
    """
    Wrapper de haut niveau qui couple un **Acteur** (politique π_θ) et un **Critique Duel**
    (Q_ϕ & V_ϕ).
    Caractéristiques principales
    ------------ 
      • *Boucle d'apprentissage*  
        1. L'acteur produit π_θ(a | s) et sélectionne les actions (ε-greedy).  
        2. Le critique produit Q(s,·) et V(s) → Cible TD  
           *td* = r + γ maxₐ′ Q(s′, a′).  
        3. Pertes  
           – **Acteur** : −log π_θ · Avantage  (A = Q−V)  − β H[π]  
           – **Critique**: MSE(Q(s,a), td)  
        4. Deux optimiseurs Adam indépendants mettent à jour θ et ϕ.
    """
    def __init__(self, device,state_size, action_size, gamma, learning_rate, load_model=False, load_path=None):
        """
        Initialisation de l'agent
        :param state_size: Taille du vecteur d'état
        :param action_size: Nombre d'actions possibles
        :param gamma: Facteur d'actualisation pour les récompenses futures
        :param learning_rate: Taux d'apprentissage
        :param load_model: Si True, charge un modèle existant
        :param load_path: Chemin vers le modèle à charger
        """

        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.entropy_coeff = 0.01
        self.value_loss_coeff = 0.15
        self.invalid_action_loss_coeff = 15
        self.policy_loss_coeff = 0.8
        self.reward_norm_coeff = 4.0
        self.target_match_loss_coeff = 0.2
        self.critic_loss_coeff = 0.015

        # Utilisation du modèle Transformer qui attend une séquence d'inputs
        self.actor_model = QuadrupedActorModel(input_dim=state_size, output_dim=action_size).to(device)
        self.critic_model = QuadrupedCriticModel(input_dim=state_size, output_dim=action_size).to(device)
        self.optimizer = optim.Adam(self.actor_model.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=self.learning_rate * 0.1)
        self.memory = deque(maxlen=1_000)  # Buffer de replay

        if load_model:
            self.load(load_path)
        
    def load(self, load_path):
        """
        Charge un modèle sauvegardé
        """
        if not isinstance(load_path, str):
            raise TypeError(f"[AGENT] load_path doit être une chaîne de caractères (reçu: {type(load_path)})")
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"[AGENT] Le fichier {load_path} n'existe pas")
        
        try:
            checkpoint = torch.load(load_path)
            self.actor_model.load_state_dict(checkpoint['actor_state_dict'])
            self.critic_model.load_state_dict(checkpoint['critic_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            print(f"[AGENT] Modèle chargé avec succès: {load_path}")
        except Exception as e:
            raise RuntimeError(f"[AGENT] Erreur lors du chargement du modèle: {str(e)}")

    def get_action(self, state, epsilon=0.0):
        """
        Sélectionne une action selon la politique epsilon-greedy.
        Ici, 'state' est une séquence de vecteurs (shape: [n, 106]).

        Retourne l'action choisie et une éventuelle pénalité si l'action était invalide.
        """
        if not isinstance(state, (list, np.ndarray)):
            raise TypeError(f"[AGENT] state doit être une liste ou un numpy array (reçu: {type(state)})")
    
        if np.random.rand() < epsilon:
            action_probs = np.random.rand(self.action_size)
            chosen_actions = [1 if prob > 0.5 else 0 for prob in action_probs]
            if DEBUG_RL_AGENT:
                print(f"[AGENT] Action choisie aléatoirement parmi les actions valides (epsilon-greedy)")
        else:
            # Convertir les arrays numpy en tenseurs PyTorch
            state_tensors = [torch.from_numpy(s).float().to(self.device) for s in state]
            state_tensor = torch.stack(state_tensors).unsqueeze(0)
            
            self.actor_model.eval()
            with torch.no_grad():
                action_probs = self.actor_model(state_tensor).cpu().numpy().flatten()
            
            chosen_actions = [1 if prob > 0.5 else 0 for prob in action_probs]
            if DEBUG_RL_AGENT:
                print(f"[AGENT] Action choisie par le modèle actuel en train d'être entrainé (epsilon-greedy)")
                    
        return chosen_actions, action_probs

    def remember(self, state, action_probs, reward, done, next_state):
        """
        Stocke une transition dans la mémoire de replay, cette transition sera utilisée pour l'entrainement du model
        """
        # Stocker la séquence d'états, l'indice d'action choisie, le masque d'action valide et la récompense finale
        self.memory.append((state, action_probs, reward, done, next_state))

    def train_model(self, batch_size=32):
        """
        Une étape d'optimisation sur un mini-batch.

        Workflow
        --------
            1.  Échantillonne `batch_size` transitions du buffer de replay choisi
                (court = on-policy, long = off-policy).  
            2.  Calcule
                    π_θ(a|s)                        # Réseau acteur
                    Q_ϕ(s, ·), V_ϕ(s)               # Réseau critique 
                    Q_target(s′, ·)                 # Réseau critique cible pour TD  
                    td_target = r + γ·maxₐ′ Q_target(s′, a′)  
                    advantage = Q(s,a) − V(s)  
            3.  Pertes  
                    critic_loss = Huber(Q(s,a), td_target)  
                    actor_loss  = −E[log π(a|s) · advantage] − β entropy  
            4.  Rétropropager et mettre à jour les deux optimiseurs.
            5.  Mettre à jour le réseau cible avec un lissage de Polyak.
        """
        if len(self.memory) < batch_size:
            if DEBUG_RL_AGENT:
                print('Pas assez de données pour entraîner:', len(self.memory))
            return {
                'reward_norm_mean': None,
                'invalid_action_loss': None,
                'value_loss': None,
                'policy_loss': None,
                'entropy': None,
                'total_loss': None
            }

        # Échantillonner les transitions et décompresser, y compris les target_vectors
        batch = random.sample(self.memory, batch_size)
        state, action_probs, rewards, dones, next_state = zip(*batch)

        # Convertir les actions et récompenses en tenseurs
        action_probs_tensor = torch.tensor([a for a in action_probs], device=self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float, device=self.device)
        states_tensor = torch.stack([torch.from_numpy(s).float() for s in state])
        next_states_tensor = torch.stack([torch.from_numpy(s).float() for s in next_state])

        # Passage en avant à travers le réseau
        action_probs = self.actor_model(states_tensor)
        q_values, state_values = self.critic_model(states_tensor)       # Q(s,*), V(s) => (batch_size, num_actions), (batch_size, 1)
        
        # Calcul des valeurs des états suivants en utilisant le réseau cible pour la stabilité
        with torch.no_grad():
            q_next, _ = self.critic_model(next_states_tensor) # Q_target(s',*) => (batch_size, num_actions)
            next_state_values = q_next.max(dim=1).values      # max_a' Q_target(s',a') => (batch_size, 1)
        
        # Obtenir la Q-value pour l'action choisie
        chosen_action_q_values = q_values.gather(1, action_probs_tensor.unsqueeze(1)).squeeze(1) # Q(s,a) => (batch_size, 1)

        # Calculer la cible TD et l'avantage
        td_targets = rewards_tensor + self.gamma * next_state_values * (1 - dones_tensor)     # td_target = r + γ·maxₐ′ Q_target(s′, a′)
        advantages = chosen_action_q_values - state_values.squeeze(1).detach()                # A = Q - V Positif signifie que l'action est meilleure que prévu.

        if DEBUG_RL_AGENT:
            print(f'[AGENT] state_values, mean : {state_values.mean()}, max : {state_values.max()}, min : {state_values.min()}')
            print(f'[AGENT] rewards, mean : {rewards_tensor.mean()}, max : {rewards_tensor.max()}, min : {rewards_tensor.min()}')
            print(f'[AGENT] next_state_values, mean : {next_state_values.mean()}, max : {next_state_values.max()}, min : {next_state_values.min()}')
            print(f'[AGENT] advantages, mean : {advantages.mean()}, max : {advantages.max()}, min : {advantages.min()}')

        # Perte du critique: MSE entre les Q-values prédites et les cibles TD
        critic_loss = F.mse_loss(chosen_action_q_values, td_targets.detach())
        
        # Perte de l'état: MSE entre les valeurs d'état prédites et les récompenses normalisées
        state_value_loss = F.mse_loss(state_values.squeeze(), td_targets.detach())
        
        # Combinaison des pertes du critique
        total_critic_loss = critic_loss * self.critic_loss_coeff + self.value_loss_coeff * state_value_loss

        # Perte de l'acteur: utiliser l'avantage pour guider la politique
        # On veut maximiser log(π(a|s)) * avantage, donc minimiser le négatif
        log_probs = torch.log(action_probs.clamp(min=1e-8))
        # Récupérer les log-probs pour les actions choisies
        action_log_probs = log_probs.gather(1, action_probs_tensor.unsqueeze(1)).squeeze(1)
        # Politique guidée par l'avantage
        policy_loss = -(action_log_probs * advantages.detach()).mean()
        
        # Bonus d'entropie: encourager l'exploration
        entropy = -torch.sum(action_probs * log_probs, dim=1).mean()
        
        # Perte totale de l'acteur: combinaison pondérée des composantes
        total_actor_loss = (
            policy_loss * self.policy_loss_coeff
            - entropy * self.entropy_coeff  # Le négatif car on veut maximiser l'entropie
        )

        # Étape d'optimisation pour le critique
        self.critic_optimizer.zero_grad()
        total_critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()
        
        # Étape d'optimisation pour l'acteur
        self.optimizer.zero_grad()
        total_actor_loss.backward()
        self.optimizer.step()

        # Métriques pour le suivi
        metrics = {
            'reward_norm_mean': rewards_tensor.mean().item(),
            'critic_loss': critic_loss.item() * self.critic_loss_coeff,
            'state_value_loss': state_value_loss.item() * self.value_loss_coeff,
            'policy_loss': policy_loss.item() * self.policy_loss_coeff,
            'entropy': entropy.item() * self.entropy_coeff,
            'total_actor_loss': total_actor_loss.item(),
            'total_critic_loss': total_critic_loss.item()
        }

        return metrics