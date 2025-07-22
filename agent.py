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
        self.critic_model = QuadrupedCriticModel(input_dim=state_size).to(device)
        self.critic_target = QuadrupedCriticModel(input_dim=state_size).to(device)
        self.critic_target.load_state_dict(self.critic_model.state_dict())
        self.optimizer = optim.Adam(self.actor_model.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=self.learning_rate * 0.1)
        self.memory = deque(maxlen=1_000)  # Buffer de replay
        self.polyak_tau = 0.995

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
        Sélectionne une action continue selon la politique (tanh ∈ [‑1, 1]).
        Retourne (shoulder_actions, elbow_actions, actions_pred) où actions_pred est le vecteur complet (8,).
        """
        if not isinstance(state, (list, np.ndarray)):
            raise TypeError(f"[AGENT] state doit être une liste ou un numpy array (reçu: {type(state)})")
    
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        self.actor_model.eval()
        with torch.no_grad():
            actions_pred = self.actor_model(state_tensor)
        actions_pred_np = actions_pred.cpu().numpy()
        shoulder_actions = actions_pred_np[:4]
        elbow_actions = actions_pred_np[4:]
        if DEBUG_RL_AGENT:
            print(f"[AGENT] shoulder_actions : {shoulder_actions}")
            print(f"[AGENT] elbow_actions : {elbow_actions}")
        return shoulder_actions, elbow_actions, actions_pred

    def remember(self, state, action_probs, reward, done, next_state):
        """
        Stocke une transition dans la mémoire de replay, cette transition sera utilisée pour l'entrainement du model
        """
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
                'critic_loss': None,
                'actor_loss': None,
                'entropy': None,
                'total_loss': None
            }

        batch = random.sample(self.memory, batch_size)
        state, action_probs, rewards, dones, next_state = zip(*batch)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)
        states_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        next_states_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        action_tensor = torch.stack(action_probs).to(self.device)

        # Critic forward
        state_values = self.critic_model(states_tensor).squeeze(1)
        next_state_values = self.critic_target(next_states_tensor).squeeze(1).detach()

        # TD target et advantage
        td_targets = rewards_tensor + self.gamma * next_state_values * (1 - dones_tensor)
        advantages = td_targets - state_values.detach()

        critic_loss = F.mse_loss(state_values, td_targets)

        # Actor: log-prob d'une gaussienne diag centrée sur l'action prédite
        mu = self.actor_model(states_tensor)
        log_std = torch.zeros_like(mu)  # std fixée à 1 pour chaque action
        std = log_std.clamp(-4, 2).exp() + 1e-6
        dist = torch.distributions.Normal(mu, std)
        logp_actions = dist.log_prob(action_tensor).sum(-1)
        actor_loss = -(advantages * logp_actions).mean()
        entropy = dist.entropy().sum(-1).mean()

        # Optim Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()

        # Optim Actor
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        # Polyak update
        with torch.no_grad():
            for param, target_param in zip(self.critic_model.parameters(), self.critic_target.parameters()):
                target_param.data.mul_(self.polyak_tau)
                target_param.data.add_((1 - self.polyak_tau) * param.data)

        metrics = {
            'reward_norm_mean': rewards_tensor.mean().item(),
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'entropy': entropy.item(),
            'total_loss': actor_loss.item() + critic_loss.item()
        }
        return metrics