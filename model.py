# poker_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import random as rd


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
        
        # Tête de sortie pour prédire les probabilités d'action.
        # Elle transforme la représentation finale en un vecteur de probabilité de taille output_dim .
        self.action_head = nn.Sequential(
            nn.Linear(dim_feedforward, dim_feedforward),
            nn.GELU(),
            nn.LayerNorm(dim_feedforward),
            nn.Linear(dim_feedforward, output_dim),
            nn.Softmax(dim=-1)
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
    def __init__(self, input_dim, output_dim, nhead=4, num_layers=4, dim_feedforward=512):
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

        # Tête de sortie pour prédire les probabilités d'action.
        # Elle transforme la représentation finale (64 dimensions) en un vecteur de probabilité de taille output_dim (ici 5).
        self.V_head = nn.Sequential(
            nn.Linear(dim_feedforward, dim_feedforward),
            nn.GELU(),
            nn.LayerNorm(dim_feedforward),
            nn.Linear(dim_feedforward, 1)
        )

        self.A_head = nn.Sequential(
            nn.Linear(dim_feedforward, dim_feedforward),
            nn.GELU(),
            nn.LayerNorm(dim_feedforward),
            nn.Linear(dim_feedforward, output_dim)
        )

    def forward(self, x):
        """
        Here V(s) estimates the value of the state, it's an estimation of how much the situation is favorable (in terms of future expected rewards)
        A(s,a) estimates the advantage of each action combination, it's an estimation of how much each action combination is favorable compared to the other combinations.

        So Q(s,a) is a function that estimates the future expected rewards of action combination a in state s.
        To do so, it takes that current value of the state V(s), add the advantage of the action combination A(s,a) to it and substract the mean of the advantages to normalize it. 
        """
        x = self.seq_1(x)
        x = self.seq_2(x)
        x = self.seq_3(x)

        V = self.V_head(x)

        A = self.A_head(x)

        # 7. Calcul de la Q-value :
        #    Q(s,a)=V+A-mean(A)
        Q = V + A - A.mean(dim=-1, keepdim=True)

        return Q, V




