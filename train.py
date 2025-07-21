# poker_train_expresso.py
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

def run_episode(env: QuadrupedEnv, epsilon: float, rendering: bool, episode: int, render_every: int, data_collector: DataCollector) -> Tuple[List[float], List[dict]]:
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

    env.reset()
    
    experiences = []

    if DEBUG_RL_TRAIN:
        print(f"[TRAIN] Début de la main")

    #### Boucle principale du jeu ####
    for step in range(MAX_STEPS):
        # Récupération du joueur actuel et mise à jour des boutons   
        current_state = env.get_state()

        # Prédiction avec une inférence classique du modèle
        chosen_action, action_mask, action_probs = agent.get_action(state = current_state, valid_actions = valid_actions, target_vector = target_vector, epsilon = epsilon)

        if DEBUG_RL_TRAIN:
            print(f"[TRAIN] state : {state}")
            print(f"[TRAIN] current_player.name : {current_player.name}, current_phase : {env.current_phase}")
            print(f"[TRAIN] current_player_bet : {current_player.current_player_bet}, current_maximum_bet : {env.current_maximum_bet}, stack : {current_player.stack}")
        
        review_state = env.get_state(seat_position = current_player.seat_position)
        if state.tolist() != review_state.tolist():
            raise ValueError(f"[TRAIN] state != review_state => {state.tolist()} != {review_state.tolist()}")

        # Exécuter l'action dans l'environnement
        next_state = env.step(chosen_action)
        
        # Mise à jour de la séquence : on ajoute le nouvel état à la fin
        if env.current_phase != GamePhase.SHOWDOWN:
            # garde les 10 derniers états
            state_seq[current_player.name] = state_seq[current_player.name][-10:]
            current_state_seq = state_seq[current_player.name].copy() # avant action
            # garde les 10 derniers états
            state_seq[current_player.name] = state_seq[current_player.name][-10:]
            current_state_seq = state_seq[current_player.name].copy() # avant action
            state_seq[current_player.name].append(next_state)
            next_state_seq = state_seq[current_player.name][-10:].copy() # après action
            next_state_seq = state_seq[current_player.name][-10:].copy() # après action
            
            # Sauve l'exp dans un json. On ne stocke pas le state durant le showdown car on le fait plus tard et cela créerait un double compte
            current_state = player_state_seq[-1]


            state_info = {
                "player": current_player.name,
                "phase": env.current_phase.value,
                "action": chosen_action.value,
                "final_stacks": env.final_stacks,
                "num_active_players": len(players_that_can_play),
                "state_vector": current_state.tolist(),
                "target_vector": target_vector.tolist(),
                }
            data_collector.add_state(state_info)
            
            # Buffer experience for later reward assignment
            experiences.append((current_player.agent, current_state_seq.copy(), action_to_idx[chosen_action], action_mask, target_vector, env.current_phase, phi_value, next_state_seq.copy()))
            experiences.append((current_player.agent, current_state_seq.copy(), action_to_idx[chosen_action], action_mask, target_vector, env.current_phase, phi_value, next_state_seq.copy()))
        
        else : # Cas spécifique au joueur qui déclenche le showdown par son action
            # Stocker l'expérience pour l'entrainement du modèle: on enregistre une copie de la séquence courante
            previous_player_state_seq = state_seq[current_player.name][-10:].copy()
            previous_player_state_seq = state_seq[current_player.name][-10:].copy()
            penultimate_state = previous_player_state_seq[-1]
            final_state = env.get_final_state(penultimate_state, env.final_stacks)
            
            # On crée la séquence d'états pour l'expérience finale mais elle ne sera pas utilisée pour l'entrainement du modèle car on est sur done = True
            next_state_seq = state_seq[current_player.name][-10:].copy()
            
            # On crée la séquence d'états pour l'expérience finale mais elle ne sera pas utilisée pour l'entrainement du modèle car on est sur done = True
            next_state_seq = state_seq[current_player.name][-10:].copy()

            # Stocker l'expérience
            experiences.append((current_player.agent, previous_player_state_seq.copy(), action_to_idx[chosen_action], action_mask, target_vector, env.current_phase, phi_value, next_state_seq.copy()))
            experiences.append((current_player.agent, previous_player_state_seq.copy(), action_to_idx[chosen_action], action_mask, target_vector, env.current_phase, phi_value, next_state_seq.copy()))
        
        # Rendu graphique si activé
        handle_rendering(env, rendering, episode, render_every)

    # Calcul des récompenses finales en utilisant les stacks capturées pré-reset
    print(f"\n[TRAIN] === Résultats de l'épisode [{episode + 1}/{EPISODES}] ===")
    # Attribution des récompenses finales
    for player in env.players:
        # ---- Pour la collecte et l'affichage des métriques ----
        # Récupération de l'état final
        player_state_seq = state_seq[player.name]
        penultimate_state = player_state_seq[-1]
        final_state = env.get_final_state(penultimate_state, env.final_stacks)
        # Stocker l'expérience finale pour la collecte des métriques
        current_player_name = player.name
        
        state_info = {
            "player": current_player_name,
            "phase": GamePhase.SHOWDOWN.value,
            "action": None,
            "stack_changes": env.net_stack_changes,
            "final_stacks": env.final_stacks,
            "num_active_players": len(players_that_can_play),
            "state_vector": final_state.tolist(),
            "target_vector": target_vector.tolist(),
        }
        data_collector.add_state(state_info)
    
    # --------------------------------------------------
    #  Construction des récompenses densifiées via φ
    # --------------------------------------------------
    agent_exp_indices = {} # {agent_1 : [0, 3, 6], agent_2 : [1, 4, 7], agent_3 : [2, 5]}
    for idx, experience in enumerate(experiences):
        agent = experience[0]
        agent_exp_indices.setdefault(agent, []).append(idx)
    
    for idx, experience in enumerate(experiences):
        (agent, state_sequence, action_idx, valid_mask,
         target_vector, phase, current_phi_value,
         next_state_seq) = experience # ← 8 éléments
        indices = agent_exp_indices[agent] # [0, 3, 6]
        game_action_index = indices.index(idx)
        if DEBUG_RL_TRAIN:
            print(f"[TRAIN] agent={agent.name}, action_idx={action_idx}, state_sequence length: {len(state_sequence)} and length of state_sequence[0]: {len(state_sequence[0])}")
        is_final = (game_action_index == len(indices) - 1) # C'est la dernière action du joueur dans l'épisode ?
        next_phi_value = experiences[indices[game_action_index + 1]][6] if not is_final else current_phi_value # On récupère la valeur de φ(s') pour la prochaine action (ssi ce n'est pas la dernière action de la main pour ce joueur)
    agent_exp_indices = {} # {agent_1 : [0, 3, 6], agent_2 : [1, 4, 7], agent_3 : [2, 5]}
    for idx, experience in enumerate(experiences):
        agent = experience[0]
        agent_exp_indices.setdefault(agent, []).append(idx)
    
    for idx, experience in enumerate(experiences):
        (agent, state_sequence, action_idx, valid_mask,
         target_vector, phase, current_phi_value,
         next_state_seq) = experience # ← 8 éléments
        indices = agent_exp_indices[agent] # [0, 3, 6]
        game_action_index = indices.index(idx)
        if DEBUG_RL_TRAIN:
            print(f"[TRAIN] agent={agent.name}, action_idx={action_idx}, state_sequence length: {len(state_sequence)} and length of state_sequence[0]: {len(state_sequence[0])}")
        is_final = (game_action_index == len(indices) - 1) # C'est la dernière action du joueur dans l'épisode ?
        next_phi_value = experiences[indices[game_action_index + 1]][6] if not is_final else current_phi_value # On récupère la valeur de φ(s') pour la prochaine action (ssi ce n'est pas la dernière action de la main pour ce joueur)

        R_env = env.net_stack_changes[agent.name] if is_final else 0.0 # Récompense de l'environnement (Variation de stack (ssi c'est la dernière action de la main pour ce joueur)
        reward = R_env + agent.gamma * next_phi_value - current_phi_value # Récompense totale (R_env + γ·φ(s') - φ(s)) (cf. theorie)
        if DEBUG_RL_TRAIN:
            print(f"[TRAIN] agent={agent.name}, action_idx={action_idx}, R_env={R_env:.3f}, next_phi_value={next_phi_value:.3f}, current_phi_value={current_phi_value:.3f}, reward={reward:.3f}")
        done = is_final 
        if DEBUG_RL_TRAIN:
            print(f"[TRAIN] agent={agent.name}, action_idx={action_idx}, next_state_seq length: {len(next_state_seq)} and length of next_state_seq[0]: {len(next_state_seq[0])}")
        R_env = env.net_stack_changes[agent.name] if is_final else 0.0 # Récompense de l'environnement (Variation de stack (ssi c'est la dernière action de la main pour ce joueur)
        reward = R_env + agent.gamma * next_phi_value - current_phi_value # Récompense totale (R_env + γ·φ(s') - φ(s)) (cf. theorie)
        if DEBUG_RL_TRAIN:
            print(f"[TRAIN] agent={agent.name}, action_idx={action_idx}, R_env={R_env:.3f}, next_phi_value={next_phi_value:.3f}, current_phi_value={current_phi_value:.3f}, reward={reward:.3f}")
        done = is_final 
        if DEBUG_RL_TRAIN:
            print(f"[TRAIN] agent={agent.name}, action_idx={action_idx}, next_state_seq length: {len(next_state_seq)} and length of next_state_seq[0]: {len(next_state_seq[0])}")

        if DEBUG_RL_TRAIN:
            print(f"[TRAIN] agent={agent.name}, reward={reward:.3f} (R_env={R_env:.3f}, Δφ={(agent.gamma*next_phi_value - current_phi_value):.3f})")
            print(f"[TRAIN] agent={agent.name}, reward={reward:.3f} (R_env={R_env:.3f}, Δφ={(agent.gamma*next_phi_value - current_phi_value):.3f})")

        agent.remember(
            state_seq         = state_sequence,
            action_index      = action_idx,
            valid_action_mask = valid_mask,
            reward            = reward,
            target_vector     = target_vector,
            done              = done,
            next_state_seq    = next_state_seq
        )

    metrics_list = []
    for player in env.players:
        metrics = player.agent.train_model()
        metrics_list.append(metrics)

    # Sauvegarde des données
    data_collector.add_metrics(metrics_list)
    data_collector.save_episode(episode)

    # Gestion du rendu graphique final
    if rendering and (episode % render_every == 0):
        handle_final_rendering(env)

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
            run_episode(env, epsilon, rendering, episode, render_every, data_collector)
            
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
