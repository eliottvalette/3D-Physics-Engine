# train.py
import os
import gc
import numpy as np
import random as rd
import pygame
import torch
import time
import copy
import traceback
from visualization import DataCollector
from agent import QuadrupedAgent
from physics_env.quadruped_env import QuadrupedEnv
from typing import List, Tuple
import json
from physics_env.config import EPISODES, EPS_DECAY, START_EPS, EPS_MIN, DEBUG_RL_TRAIN, SAVE_INTERVAL, PLOT_INTERVAL, MAX_STEPS
from helpers_rl import save_models

def run_episode(env: QuadrupedEnv, agent: QuadrupedAgent, epsilon: float, rendering: bool, episode: int, render_every: int, data_collector: DataCollector) -> Tuple[List[float], List[dict]]:
    """
    Exécute un épisode complet du jeu de quadruped.
    
    Args:
        env (QuadrupedEnv): L'environnement de jeu
        epsilon (float): Paramètre d'exploration
        rendering (bool): Active/désactive le rendu graphique
        episode (int): Numéro de l'épisode en cours
        render_every (int): Fréquence de mise à jour du rendu
        data_collector (DataCollector): Collecteur de données pour la visualisation

    Returns:
        Tuple[List[float], List[dict]]: Récompenses finales et métriques d'entraînement
    """

    env.quadruped.reset()

    if DEBUG_RL_TRAIN:
        print(f"[TRAIN] Début de la main")

    #### Boucle principale du jeu ####
    for step in range(MAX_STEPS):
        # Récupération du joueur actuel et mise à jour des boutons   
        state = env.get_state()

        # Prédiction avec une inférence classique du modèle
        chosen_actions, action_probs = agent.get_action(state = state, epsilon = epsilon)
        
        if DEBUG_RL_TRAIN:
            print(f"[TRAIN] state : {state}")
            print(f"[TRAIN] action_probs : {action_probs}")
            print(f"[TRAIN] chosen_actions : {chosen_actions}")
        
        review_state = env.get_state()
        if state.tolist() != review_state.tolist():
            raise ValueError(f"[TRAIN] state != review_state => {state.tolist()} != {review_state.tolist()}")

        # Exécuter l'action dans l'environnement
        next_state, reward = env.step(chosen_actions)

        # Stocker l'expérience
        done = step == MAX_STEPS - 1
        agent.remember(state, action_probs, reward, done, next_state)

        data_collector.add_state(state)
    
        # Rendu graphique si activé
        if rendering and (step % render_every == 0):
            env.render(reward)

    print(f"\n[TRAIN] === Résultats de l'épisode [{episode + 1}/{EPISODES}] ===")

    metrics_list = []
    metrics = agent.train_model()
    metrics_list.append(metrics)

    # Sauvegarde des données
    data_collector.add_metrics(metrics_list)
    data_collector.save_episode(episode)

def main_training_loop(agent: QuadrupedAgent, episodes: int, rendering: bool, render_every: int):
    """
    Boucle principale d'entraînement des agents.
    
    Args:
        agent (QuadrupedAgent): Agent à entraîner
        episodes (int): Nombre total d'épisodes d'entraînement
        rendering (bool): Active/désactive le rendu graphique
        render_every (int): Fréquence de mise à jour du rendu graphique
    """
    # Initialisation des historiques et de l'environnement
    env = QuadrupedEnv(rendering=rendering)
    
    # Configuration du collecteur de données
    data_collector = DataCollector(
        save_interval=SAVE_INTERVAL,
        plot_interval=PLOT_INTERVAL,
        start_epsilon=START_EPS,
        epsilon_decay=EPS_DECAY,
        epsilon_min=EPS_MIN
    )
    
    try:
        for episode in range(episodes):
            start_time = time.time()

            # Décroissance d'epsilon
            epsilon = np.clip(START_EPS * EPS_DECAY ** episode, EPS_MIN, START_EPS)
            epsilon = np.clip(START_EPS * EPS_DECAY ** episode, EPS_MIN, START_EPS)
            
            # Exécuter l'épisode et obtenir les résultats incluant les métriques
            run_episode(env, agent, epsilon, rendering, episode, render_every, data_collector)
            
            # Afficher les informations de l'épisode
            print(f"\n[TRAIN] Episode [{episode + 1}/{episodes}]")
            print(f"[TRAIN] Randomness: {epsilon*100:.3f}%")
            print(f"[TRAIN] Time taken: {time.time() - start_time:.2f} seconds")
            
        # Save models at end of training
        if episode == episodes - 1:
            save_models(agent, episode)
            print("[TRAIN] Generating visualization...")
            data_collector.force_visualization()

    except Exception as e:
        print(f"[TRAIN] An error occurred: {e}")
        save_models(agent, episode)
        print("[TRAIN] Generating visualization...")
        data_collector.force_visualization()
        
    finally:
        save_models(agent, episode)
        print("[TRAIN] Generating visualization...")
        data_collector.force_visualization()
