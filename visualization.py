# visualization.py
import matplotlib
matplotlib.use('Agg')  # Use Agg backend that doesn't require GUI
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import seaborn as sns
import os
import json
import pandas as pd
from datetime import datetime
from physics_env.config import DEBUG_RL_VIZ, START_EPS, EPS_DECAY, EPS_MIN, PLOT_INTERVAL, SAVE_INTERVAL

PLAYERS = ['Player_0', 'Player_1', 'Player_2']

class DataCollector:
    def __init__(self, save_interval, plot_interval, start_epsilon, epsilon_decay, epsilon_min, output_dir="viz_json"):
        """
        Initialise le collecteur de données.
        
        Args:
            save_interval (int): Nombre d'épisodes à regrouper avant de sauvegarder
            plot_interval (int): Intervalle pour le tracé
            output_dir (str): Répertoire pour enregistrer les fichiers JSON
        """
        self.save_interval = save_interval
        self.plot_interval = plot_interval
        self.output_dir = output_dir
        self.start_epsilon = start_epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.current_episode_states = []
        self.batch_episode_states = [] # Contient un liste de current_episode_states qui seront ajouté toutes les save_interval dans le fichier json
        self.batch_episode_metrics = [] # Contient les métriques d'entraînement pour chaque épisode
        
        # Créer le répertoire pour les JSON s'il n'existe pas
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Créer le dossier visualizations si il n'existe pas
        if not os.path.exists("visualizations"):
            os.makedirs("visualizations")

        # Créer le dossier daté pour les visualisations
        self.viz_dir = "visualizations" # On save dans le dossier visualizations, pas le peine de créer un dossier daté
        if not os.path.exists(self.viz_dir):
            os.makedirs(self.viz_dir)

        # Supprimer les fichiers JSON existants dans le répertoire
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))

        self.visualizer = Visualizer(
            output_dir=output_dir, 
            viz_dir=self.viz_dir,
            start_epsilon=start_epsilon, 
            epsilon_decay=epsilon_decay, 
            epsilon_min=epsilon_min,
            plot_interval=plot_interval, 
            save_interval=save_interval
        )
    
    def add_state(self, state):
        """
        Ajoute un état à l'épisode courant.
        """

        self.current_episode_states.append(state.copy())  # Use copy to ensure no reference issues
    
    def add_metrics(self, episode_metrics):
        """
        Ajoute les métriques d'un épisode au batch.
        
        Args:
            episode_metrics (list): Liste des métriques pour chaque agent dans l'épisode
        """
        self.batch_episode_metrics.append(episode_metrics)
    
    def save_episode(self, episode_num):
        """
        Sauvegarde les états et métriques de l'épisode courant dans des fichiers JSON.
        
        Args:
            episode_num (int): Numéro de l'épisode
        """
        # Ajouter l'épisode courant au batch
        self.batch_episode_states.append(self.current_episode_states)
        
        # Vérifier si on a atteint l'intervalle de sauvegarde
        if len(self.batch_episode_states) >= self.save_interval:
            # Sauvegarder les états
            states_filename = os.path.join(self.output_dir, "episodes_states.json")
            
            # Créer le fichier avec un dictionnaire vide s'il n'existe pas
            if not os.path.exists(states_filename):
                with open(states_filename, 'w') as f:
                    f.write("{}")
            
            # Ajouter les nouveaux épisodes au fichier JSON
            for i, episode_states in enumerate(self.batch_episode_states):
                episode_idx = episode_num - len(self.batch_episode_states) + i + 1
                self._append_to_json(states_filename, str(episode_idx), episode_states)
            
            # Sauvegarder les métriques
            metrics_filename = os.path.join(self.output_dir, "metrics_history.json")
            
            # Créer le fichier avec un dictionnaire vide s'il n'existe pas
            if not os.path.exists(metrics_filename):
                with open(metrics_filename, 'w') as f:
                    f.write("{}")
            
            # Ajouter les nouvelles métriques au fichier JSON
            for i, episode_metrics in enumerate(self.batch_episode_metrics):
                episode_idx = episode_num - len(self.batch_episode_metrics) + i + 1
                self._append_to_json(metrics_filename, str(episode_idx), episode_metrics)

            # Reset batches
            self.batch_episode_states = []
            self.batch_episode_metrics = []
        
        # Reset current episode states
        self.current_episode_states = []

        if episode_num % self.plot_interval == self.plot_interval - 1:
            # Load Jsons
            states_path = os.path.join(self.output_dir, "episodes_states.json")
            metrics_path = os.path.join(self.output_dir, "metrics_history.json")
            
            with open(states_path, 'r') as f:
                states_data = json.load(f)
            with open(metrics_path, 'r') as f:
                metrics_data = json.load(f)
            self.visualizer.plot_progress(states_data, metrics_data, dpi = 250)
            
            # Supprimer les données et libérer de la mémoire
            del states_data, metrics_data
            plt.close('all')

        if episode_num % (self.plot_interval * 3) == (self.plot_interval * 3) - 1:
            # Load Jsons
            states_path = os.path.join(self.output_dir, "episodes_states.json")
            metrics_path = os.path.join(self.output_dir, "metrics_history.json")
            
            with open(states_path, 'r') as f:
                states_data = json.load(f)
            with open(metrics_path, 'r') as f:
                metrics_data = json.load(f)
            self.visualizer.plot_metrics(metrics_data)

            # Supprimer les données et libérer de la mémoire
            del states_data, metrics_data
            plt.close('all')
    
    def _append_to_json(self, file_path, key, data):
        """
        Ajoute une nouvelle entrée à un fichier JSON existant en modifiant le fichier en place.
        
        Args:
            file_path (str): Chemin du fichier JSON
            key (str): Clé de la nouvelle entrée
            data: Données à ajouter
        """
        with open(file_path, 'r+') as f:
            f.seek(0)
            content = f.read().strip()
            # On part du principe que le contenu est un dict valide
            # Supprimer le dernier caractère (qui doit être "}")
            f.seek(0, os.SEEK_END)
            pos = f.tell() - 1
            f.seek(pos)
            last_char = f.read(1)
            if last_char != '}':
                raise ValueError(f"Fichier JSON mal formé: {file_path}")
            
            # Vérifier si le dict est vide
            if content == "{}":
                new_content = f'"{key}": {json.dumps(data)}'
            else:
                new_content = f', "{key}": {json.dumps(data)}'
            
            # Réécriture : on tronque le dernier "}" et on y ajoute notre contenu + "}"
            f.seek(pos)
            f.truncate()
            f.write(new_content + "}")
    
    def force_visualization(self):
        """
        Force la génération de toutes les visualisations
        """
        # On les json une seule fois
        states_path = os.path.join(self.output_dir, "episodes_states.json")
        metrics_path = os.path.join(self.output_dir, "metrics_history.json")
        
        with open(states_path, 'r') as f:
            states_data = json.load(f)
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)

        self.visualizer.plot_progress(states_data, metrics_data)
        self.visualizer.plot_metrics(metrics_data)
        self.visualizer.plot_analytics(states_data)
        self.visualizer.plot_heatmaps_by_players(states_data)
        self.visualizer.plot_heatmaps_by_position(states_data)
        self.visualizer.plot_stack_sum(states_data)

        # Supprimer les données et libérer de la mémoire
        del states_data, metrics_data
        plt.close('all')

class Visualizer:
    """
    Visualise les données collectées dans le répertoire viz_json
    """
    def __init__(self, start_epsilon, epsilon_decay, epsilon_min, plot_interval, save_interval, output_dir="viz_json", viz_dir=None):
        self.output_dir = output_dir
        self.viz_dir = viz_dir or os.path.join("visualizations", datetime.now().strftime("%Y-%m-%d_%Hh-%Mm-%Ss"))
        self.start_epsilon = start_epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.plot_interval = plot_interval
        self.save_interval = save_interval

        # Create all necessary directories
        for directory in [self.output_dir, self.viz_dir]:
            os.makedirs(directory, exist_ok=True)

        # Définition des couleurs pour chaque action
        self.action_colors = {
            'fold': '#780000',     # Rouge Sang
            'check': '#C1121F',    # Rouge
            'call': '#FDF0D5',     # Beige
            'raise': '#669BBC',    # Bleu
            'all-in': '#003049'    # Bleu Nuit
        }

    def plot_progress(self, states_data, metrics_data, dpi=500):
        """
        Génère les visualisations à partir des données JSON enregistrées
        """        
        # Créer une figure avec 6 sous-graphiques (2x3)
        fig = plt.figure(figsize=(25, 20))
        
        # Définir une palette de couleurs pastel
        pastel_colors = ['#003049', '#006DAA', '#D62828', '#F77F00', '#FCBF49', '#EAE2B7']
        
        # 1. Average mbb/hand par agent
        ax1 = plt.subplot(2, 3, 1)
        window = self.plot_interval * 3 
        agents = PLAYERS
        mbb_data = {agent: [] for agent in agents}
        for episode_num, episode in states_data.items():
            for state in episode:
                if state["phase"] == "showdown":            
                    # Pour chaque agent, ajouter son stack change en mbb
                    for agent in agents:
                        if agent in state["stack_changes"]:
                            stack_change = state["stack_changes"][agent]
                            mbb_data[agent].append(stack_change * 1000)  # Conversion en mbb
                        else:
                            raise Exception(f"Agent {agent} n'a pas de showdown dans les metrics même vide")
                    break

        # Tracer les moyennes mobiles pour chaque agent
        for i, (agent, data) in enumerate(mbb_data.items()):
            rolling_avg = pd.Series(data).rolling(window=window, min_periods=1).mean()
            ax1.plot(rolling_avg, label=agent, color=pastel_colors[i], linewidth=3)
        
        ax1.set_title('Average mbb/hand par agent')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('mbb/hand')
        ax1.legend()
        ax1.set_ylim(-30000, 30000)
        
        ax1.set_facecolor('#F0F0F0')  # Fond légèrement gris
        ax1.grid(True, alpha=0.3)
        
        # 2. Fréquence des actions par agent
        ax3 = plt.subplot(2, 3, 2)
        action_freq = {agent: {
            'fold': 0, 
            'check': 0, 
            'call': 0, 
            'raise': 0, 
            'all-in': 0
        } for agent in agents}
        
        for episode in states_data.values():
            for state in episode:
                if state["action"]:
                    action = state["action"].lower()
                    # Normaliser les actions de raise pour le comptage
                    if action.startswith('raise-'):
                        action = 'raise'
                    action_freq[state["player"]][action] += 1
        
        x = np.arange(len(agents))
        width = 0.15
        actions = ['fold', 'check', 'call', 'raise', 'all-in']
        
        for i, action in enumerate(actions):
            values = [action_freq[agent][action] for agent in agents]
            total_actions = [sum(action_freq[agent].values()) for agent in agents]
            frequencies = [v/t if t > 0 else 0 for v, t in zip(values, total_actions)]
            bars = ax3.bar(x + i*width, frequencies, width, 
                          label=action, 
                          color=self.action_colors[action])
            
            # Ajouter les pourcentages au-dessus des barres
            for bar, freq in zip(bars, frequencies):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{freq*100:.1f}%',
                        ha='center', va='bottom', rotation=0,
                        fontsize=8)
        
        ax3.set_title('Fréquence des actions par agent')
        ax3.set_xticks(x + width*2)
        ax3.set_xticklabels(agents)
        ax3.legend()
        ax3.set_ylim(0, 1)  # Fixer la limite y à 1
        
        ax3.set_facecolor('#F0F0F0')
        ax3.grid(True, alpha=0.3)

        # 3. Fréquence des actions par agent et par phase de jeu
        ax2 = plt.subplot(2, 3, 3)
        phase_action_freq = {agent: {
            phase: {
                'fold': 0, 
                'check': 0, 
                'call': 0, 
                'raise': 0, 
                'all-in': 0
            } for phase in ['preflop', 'flop', 'turn', 'river', 'showdown']
        } for agent in agents}

        # Compter les actions par phase pour chaque agent
        for episode in states_data.values():
            for state in episode:
                if state["action"] and state["phase"].lower() != 'showdown':
                    action = state["action"].lower()
                    # Normaliser les actions de raise pour le comptage
                    if action.startswith('raise-'):
                        action = 'raise'
                    phase = state["phase"].lower()
                    phase_action_freq[state["player"]][phase][action] += 1

        # Créer le graphique empilé pour chaque phase
        phases = ['preflop', 'flop', 'turn', 'river']
        actions = ['fold', 'check', 'call', 'raise', 'all-in']
        x = np.arange(len(agents))
        width = 0.2  # Largeur des barres

        for p_idx, phase in enumerate(phases):
            bottom = np.zeros(len(agents))
            for a_idx, action in enumerate(actions):
                values = []
                for agent in agents:
                    total_actions = sum(phase_action_freq[agent][phase].values())
                    freq = phase_action_freq[agent][phase][action] / total_actions if total_actions > 0 else 0
                    values.append(freq)
                
                bars = ax2.bar(x + p_idx * width - width * 1.5, values, width, 
                        bottom=bottom, 
                        label=f'{action} ({phase})' if p_idx == 0 else "",
                        color=self.action_colors[action],
                        alpha=0.7)
                
                # Ajouter les pourcentages pour les barres > 5%
                for idx, (val, b) in enumerate(zip(values, bars)):
                    if val > 0.05:  # Seulement pour les valeurs > 5%
                        ax2.text(
                            b.get_x() + b.get_width()/2.,
                            bottom[idx] + val/2,
                            f'{val*100:.0f}%',
                            ha='center',
                            va='center',
                            fontsize=6,
                            color='black',
                            rotation=90
                        )
                bottom += np.array(values)  # Mettre à jour bottom avec un numpy array

        ax2.set_title('Fréquence des actions par agent et par phase de jeu')
        ax2.set_xticks(x)
        ax2.set_xticklabels(agents)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.set_ylim(0, 1)

        ax2.set_facecolor('#F0F0F0')
        ax2.grid(True, alpha=0.3)

        # 4. Evolution de epsilon avec nouvelle couleur
        ax4 = plt.subplot(2, 3, 4)
        episodes = sorted([int(k) for k in metrics_data.keys()])
        epsilon_values = [np.clip(self.start_epsilon * self.epsilon_decay ** episode, self.epsilon_min, self.start_epsilon) for episode in episodes]
        ax4.plot(episodes, epsilon_values, color='#2E86AB', linewidth=2)  # Bleu foncé pour epsilon
        ax4.set_title('Evolution de Epsilon')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Epsilon')
        ax4.set_ylim(0, 1)
        
        ax4.set_facecolor('#F0F0F0')
        ax4.grid(True, alpha=0.3)

        # 6. Winrate par agent
        ax6 = plt.subplot(2, 3, 6)

        # Collecter et trier les agents
        window = self.plot_interval * 3

        # Préparer les données pour chaque agent
        agent_results = {agent: [] for agent in agents}

        for episode_num, episode in states_data.items():
            # Trouver le dernier état showdown qui contient les stack_changes finaux
            showdown_states = [s for s in episode if s["phase"].lower() == "showdown"]
                
            last_showdown_state = showdown_states[-1]
            
            # Déterminer les gagnants basés sur les stack_changes
            stack_changes = last_showdown_state["stack_changes"]
                
            # Trouver le gain maximum
            max_gain = max(stack_changes.values())
            winners = [player for player, gain in stack_changes.items() if gain == max_gain]
            players = PLAYERS
            
            # Distribuer les résultats (1 pour victoire, 0 pour défaite)
            win_share = 1.0 / len(winners) if winners else 0
            for agent in agents:
                if agent in players:
                    result = win_share if agent in winners else 0
                    agent_results[agent].append(result)

        # Tracer les courbes de winrate pour chaque agent
        for i, (agent, results) in enumerate(agent_results.items()):
            # Convertir en pandas Series pour gérer les valeurs manquantes
            series = pd.Series(results)
            # Calculer la moyenne mobile en ignorant les valeurs manquantes
            rolling_winrate = series.rolling(window=window, min_periods=1).mean()
            
            ax6.plot(rolling_winrate, 
                     label=f"{agent}", 
                     color=pastel_colors[i],
                     linewidth=2)

        ax6.set_title('Winrate par agent (moyenne mobile sur 1000 épisodes)')
        ax6.set_xlabel('Episode')
        ax6.set_ylabel('Winrate')
        ax6.legend()

        ax6.set_facecolor('#F0F0F0')
        ax6.grid(True, alpha=0.3)

        # Style global mis à jour
        plt.rcParams.update({
            'figure.facecolor': '#FFFFFF',
            'axes.facecolor': '#F8F9FA',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.labelsize': 10,
            'axes.titlesize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'lines.linewidth': 2,
            'font.family': 'sans-serif',
            'axes.spines.top': False,
            'axes.spines.right': False
        })
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'Poker_progress.jpg'), dpi=dpi, bbox_inches='tight')
        plt.close()
    
    def plot_metrics(self, metrics_data, dpi=500):
        """
        Génère des visualisations des métriques d'entraînement à partir du fichier metrics_history.json
        """
        # Créer une figure avec 9 sous-graphiques (3x3)
        fig = plt.figure(figsize=(24, 20))
        
        # Définir une palette de couleurs pastel
        pastel_colors = ['#003049', '#006DAA', '#D62828', '#F77F00', '#FCBF49', '#EAE2B7']
        
        # Extraire les agents uniques
        agents = PLAYERS

        # Métriques spécifiques à tracer basées sur l'output de train_model
        metrics_to_plot = [
            ('reward_norm_mean', 'Reward Normalisée Moyenne', None, None),
            ('invalid_action_loss', 'Somme des probabilités d\'action sur les actions invalides', None, None),
            ('critic_loss', 'Critic Loss (MSE entre Q-Value et TD-Target)', None, None),
            ('state_value_loss', 'State Value Loss (MSE entre V(s) et TD-Target)', None, None),
            ('policy_loss', 'Policy Loss (Log-Prob * Advantage)', None, None),
            ('target_match_loss', 'Target Match Loss (KL avec MCCFR)', None, None),
            ('entropy', 'Entropie', None, None),
            ('total_actor_loss', 'Total Actor Loss', None, None),
            ('total_critic_loss', 'Total Critic Loss', None, None)
        ]

        # Créer un subplot pour chaque métrique
        for idx, (metric_name, display_name, y_max, y_min) in enumerate(metrics_to_plot):
            ax = plt.subplot(3, 3, idx + 1)
            
            # Préparer les données pour chaque agent
            for agent_idx, agent in enumerate(agents):
                episodes = []
                values = []
                
                for episode_num, episode_metrics in metrics_data.items():
                    # Check if the metric exists and is not None before adding it
                    if metric_name in episode_metrics[agent_idx] and episode_metrics[agent_idx][metric_name] is not None:
                        episodes.append(int(episode_num))
                        values.append(float(episode_metrics[agent_idx][metric_name]))
                
                # Calculer la moyenne mobile
                window = self.plot_interval * 3
                if len(values) > 0:
                    rolling_avg = pd.Series(values).rolling(window=window, min_periods=1).mean()
                    ax.plot(episodes, rolling_avg, 
                           label=agent, 
                           color=pastel_colors[agent_idx % len(pastel_colors)],
                           linewidth=2)

            ax.set_title(f'Evolution de {display_name}')
            ax.set_xlabel('Episode')
            ax.set_ylabel(display_name)
            ax.legend()
            ax.set_ylim(y_min, y_max)
            # Style du subplot
            ax.set_facecolor('#F0F0F0')
            ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # Style global
        plt.rcParams.update({
            'figure.facecolor': '#FFFFFF',
            'axes.facecolor': '#F8F9FA',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.labelsize': 10,
            'axes.titlesize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'lines.linewidth': 2,
            'font.family': 'sans-serif',
            'axes.spines.top': False,
            'axes.spines.right': False
        })

        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'Poker_metrics.jpg'), dpi=dpi, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    visualizer = Visualizer(start_epsilon=START_EPS, epsilon_decay=EPS_DECAY, epsilon_min=EPS_MIN, plot_interval=PLOT_INTERVAL, save_interval=SAVE_INTERVAL)
    # On les json une seule fois
    states_path = os.path.join(visualizer.output_dir, "episodes_states.json")
    metrics_path = os.path.join(visualizer.output_dir, "metrics_history.json")
    
    with open(states_path, 'r') as f:
        states_data = json.load(f)
    with open(metrics_path, 'r') as f:
        metrics_data = json.load(f)

    visualizer.plot_progress(states_data, metrics_data)
    visualizer.plot_metrics(metrics_data)

    plt.close('all')