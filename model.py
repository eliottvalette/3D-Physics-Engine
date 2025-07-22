# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuadrupedActorModel(nn.Module):
    """
    Calcule la politique π_θ(a | s).

    Le réseau construit un vecteur de caractéristiques partagé h = shared_layers(state),
    puis produit des logits pour toutes les combinaisons d'actions possibles.

    La sortie est une distribution catégorielle sur toutes les combinaisons d'actions possibles.
    """
    def __init__(self, input_dim, output_dim, dim_feedforward=512):
        super().__init__()

        self.seq_1 = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.GELU(),
            nn.LayerNorm(dim_feedforward)
        )

        self.seq_2 = nn.Sequential(
            nn.Linear(dim_feedforward, dim_feedforward),
            nn.GELU(),
            nn.LayerNorm(dim_feedforward)
        )

        self.seq_3 = nn.Sequential(
            nn.Linear(dim_feedforward, dim_feedforward),
            nn.GELU(),
            nn.LayerNorm(dim_feedforward)
        )
        self.action_head = nn.Sequential(
            nn.Linear(dim_feedforward, dim_feedforward),
            nn.GELU(),
            nn.LayerNorm(dim_feedforward),
            nn.Linear(dim_feedforward, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.seq_1(x)
        x = self.seq_2(x)
        x = self.seq_3(x)
        action_probs = self.action_head(x)

        return action_probs
    
class QuadrupedCriticModel(nn.Module):
    """
    Réseau Q duel pour les actions composites :
        • branche partagée  → h
        • tête V(s)         → (batch,1)
        • tête A(s,a)       → (batch, num_actions) - une pour chaque action
        • Q(s,a)=V+A-mean(A)
    """
    def __init__(self, input_dim, dim_feedforward=512):
        super().__init__()

        self.seq_1 = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.GELU(),
            nn.LayerNorm(dim_feedforward)
        )

        self.seq_2 = nn.Sequential(
            nn.Linear(dim_feedforward, dim_feedforward),
            nn.GELU(),
            nn.LayerNorm(dim_feedforward)
        )

        self.seq_3 = nn.Sequential(
            nn.Linear(dim_feedforward, dim_feedforward),
            nn.GELU(),
            nn.LayerNorm(dim_feedforward)
        )

        self.V_head = nn.Sequential(
            nn.Linear(dim_feedforward, dim_feedforward),
            nn.GELU(),
            nn.LayerNorm(dim_feedforward),
            nn.Linear(dim_feedforward, 1)
        )

    def forward(self, x):
        """
        Here V(s) estimates the value of the state, it's an estimation of how much the situation is favorable (in terms of future expected rewards)
        """
        x = self.seq_1(x)
        x = self.seq_2(x)
        x = self.seq_3(x)

        V = self.V_head(x)

        return V




