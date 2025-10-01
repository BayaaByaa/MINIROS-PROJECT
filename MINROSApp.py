import streamlit as st
import pandas as pd
from io import BytesIO
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from copy import deepcopy
from operator import attrgetter
import os
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

st.markdown(
    """
    <h1 style='text-align: center; color: black;'>Optimisation de la planification de la production </h1>
    """,
    unsafe_allow_html=True
)
DATAS='EVERYTHING.xlsx'


#__________________DIAGRAMME DE GANTT_________________________
def calculer_temps_total_T(ordonnancement, P, S):
    temps_machines = {}
    
    for machine, sequence in ordonnancement.items():
        temps_total = 0
        tache_precedente = None
        
        for tache in sequence:
            if tache_precedente is not None:
                # Setup est le même pour toutes les machines
                temps_total += S[tache_precedente][tache]
            # Temps d'exécution est le même sur toutes les machines
            temps_total += P[tache][0]
            tache_precedente = tache
        
        temps_machines[machine] = temps_total / 60  # Conversion en minutes
    
    return temps_machines
def calculer_temps_total(ordonnancement, P, S):
                temps_machines = {}
                
                for machine, sequence in ordonnancement.items():
                    temps_total = 0
                    tache_precedente = None
                    
                    for tache in sequence:
                        if tache_precedente is not None:
                            # Ajouter le temps de setup entre la tâche précédente et la tâche actuelle
                            temps_total += S[machine][tache_precedente][tache]
                        # Ajouter le temps d'exécution de la tâche actuelle
                        temps_total += P[tache][machine]
                        tache_precedente = tache
                    
                    temps_machines[machine] = temps_total/ 60
                
                return temps_machines
def plot_gantt_T(S, P, schedule, articles, machines):
    # Adapter les dimensions si nécessaire
    if len(P.shape) == 1 or P.shape[1] == 1:
        # Si P est un vecteur (temps identiques sur toutes machines)
        n_tasks = len(P)
        n_machines = len(machines)
        P_adapted = np.tile(P.reshape(-1, 1), (1, n_machines))
    else:
        n_tasks, n_machines = P.shape
        P_adapted = P
    
    # Si S est 2D (N×N), la convertir en structure attendue (une matrice par machine)
    if len(S.shape) == 2:
        S_adapted = np.tile(S[np.newaxis, :, :], (n_machines, 1, 1))
    else:
        S_adapted = S
    
    # Calcul de la taille adaptative
    figsize = calculate_figure_size(n_machines, n_tasks)
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calcul des temps totaux
    machine_times = {machine_idx: 0 for machine_idx in range(n_machines)}
    
    # Paramètres visuels adaptatifs
    bar_height = max(5, 8 - n_machines * 0.2)  # Hauteur des barres
    font_size = max(6, 15)      # Taille de police
    
    # Inversion de l'ordre pour avoir la première machine en haut
    machine_order = list(range(n_machines))[::-1]
    
    # Calcul préalable des temps
    for machine_idx in machine_order:
        if machine_idx in schedule:
            prev_task = None
            current_time = 0
            for task_idx in schedule[machine_idx]:
                setup_time = S_adapted[machine_idx][prev_task][task_idx] if prev_task is not None else 0
                proc_time = P_adapted[task_idx][machine_idx]
                current_time += setup_time + proc_time
                prev_task = task_idx
            machine_times[machine_idx] = current_time
    
    # Dessin du Gantt
    for machine_idx in machine_order:
        if machine_idx not in schedule:
            continue
            
        prev_task = None
        current_time = 0
        for task_idx in schedule[machine_idx]:
            setup_time = S_adapted[machine_idx][prev_task][task_idx] if prev_task is not None else 0
            proc_time = P_adapted[task_idx][machine_idx]
            start_time = current_time + setup_time
            
            y_pos = 10 * (n_machines - 1 - machine_idx)
            
            # Dessin de la barre
            ax.broken_barh([(start_time, proc_time)], (y_pos - bar_height/2, bar_height),
                         facecolors='#FFCB90', edgecolors='black', alpha=0.7)
            
            # Texte adapté
            label = articles[task_idx]
            if len(label) > 10 and n_tasks > 10:  # Raccourcir si beaucoup de tâches
                label = label[:8] + "..."
            ax.text(start_time + proc_time/2, y_pos, label,
                   ha='center', va='center', fontsize=font_size, color='black')
            
            current_time = start_time + proc_time
            prev_task = task_idx
        
    # Configuration des axes
    ax.set_yticks([10 * i for i in range(n_machines)])
    ax.set_yticklabels(machines[::-1], fontsize=font_size+1)
    ax.set_xlabel("Temps", fontsize=font_size+1)
    
    # Titre adaptatif
    title = f"Diagramme de Gantt"
    ax.set_title(title, fontsize=font_size+2, pad=15)
    
    ax.grid(False)
    
    # Ajustement automatique des limites
    max_time = max(machine_times.values()) * 1.2 if machine_times else 100
    ax.set_xlim(0, max_time)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Affichage des données synthétiques
    st.markdown("<h2 style='text-align: center;'>Synthèse</h2>", unsafe_allow_html=True)
    # Calcul du makespan (temps total maximum)
    makespan = max(machine_times.values()) if machine_times else 0
    
    cols = st.columns(3)
    with cols[0]:
        st.metric("Nombre de machines", n_machines)
    with cols[1]:
        st.metric("Nombre d'articles", n_tasks)
    with cols[2]:
        # Conversion du makespan en heures
        makespan_hours = round(makespan / 60, 2)
        st.metric("Temps total de la cellule", f"{makespan_hours} heures")
        
    
    # Tableau détaillé
    st.write("**Détail par machine:**")
    detail_data = {
        "Machine": machines,
        "Nombre d'articles affectés": [len(schedule.get(i, [])) for i in range(n_machines)],
        "Temps total machine (h)": [round(machine_times[i] / 60, 2) for i in range(n_machines)]
    }
    st.dataframe(detail_data, hide_index=True, use_container_width=True)
def calculate_figure_size(n_machines, n_tasks):
    # Taille de base + ajustement en fonction du nombre d'éléments
    base_width = 12
    base_height = 6
    
    # Ajustement de la hauteur en fonction du nombre de machines
    adjusted_height = base_height + n_machines * 0.5
    
    # Ajustement de la largeur en fonction du nombre de tâches
    adjusted_width = base_width + n_tasks * 0.2
    
    # Limites min/max
    fig_width = max(10, min(adjusted_width, 20))
    fig_height = max(6, min(adjusted_height, 15))
    
    return (fig_width, fig_height)

def plot_gantt(S, P, schedule, articles, machines):
    n_tasks, n_machines = P.shape
    
    # Calcul de la taille adaptative
    figsize = calculate_figure_size(n_machines, n_tasks)
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calcul des temps totaux
    machine_times = {machine_idx: 0 for machine_idx in range(n_machines)}
    
    # Paramètres visuels adaptatifs
    bar_height = max(5, 8 - n_machines * 0.2)  # Hauteur des barres
    font_size = max(6, 15)      # Taille de police
    
    # Inversion de l'ordre pour avoir la première machine en haut
    machine_order = list(range(n_machines))[::-1]
    
    # Calcul préalable des temps
    for machine_idx in machine_order:
        if machine_idx in schedule:
            prev_task = None
            current_time = 0
            for task_idx in schedule[machine_idx]:
                setup_time = S[machine_idx][prev_task][task_idx] if prev_task is not None else 0
                proc_time = P[task_idx][machine_idx]
                current_time += setup_time + proc_time
                prev_task = task_idx
            machine_times[machine_idx] = current_time
    
    # Dessin du Gantt
    for machine_idx in machine_order:
        if machine_idx not in schedule:
            continue
            
        prev_task = None
        current_time = 0
        for task_idx in schedule[machine_idx]:
            setup_time = S[machine_idx][prev_task][task_idx] if prev_task is not None else 0
            proc_time = P[task_idx][machine_idx]
            start_time = current_time + setup_time
            
            y_pos = 10 * (n_machines - 1 - machine_idx)
            
            # Dessin de la barre
            ax.broken_barh([(start_time, proc_time)], (y_pos - bar_height/2, bar_height),
                         facecolors='#FFCB90', edgecolors='black', alpha=0.7)
            
            # Texte adapté
            label = articles[task_idx]
            if len(label) > 10 and n_tasks > 10:  # Raccourcir si beaucoup de tâches
                label = label[:8] + "..."
            ax.text(start_time + proc_time/2, y_pos, label,
                   ha='center', va='center', fontsize=font_size, color='black')
            
            current_time = start_time + proc_time
            prev_task = task_idx
        

    # Configuration des axes
    ax.set_yticks([10 * i for i in range(n_machines)])
    ax.set_yticklabels(machines[::-1], fontsize=font_size+1)
    ax.set_xlabel("Temps", fontsize=font_size+1)
    
    # Titre adaptatif
    title = f"Diagramme de Gantt"
    ax.set_title(title, fontsize=font_size+2, pad=15)
    
    ax.grid(False)
    
    # Ajustement automatique des limites
    max_time = max(machine_times.values()) * 1.2 if machine_times else 100
    ax.set_xlim(0, max_time)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Affichage des données synthétiques
    st.markdown("<h2 style='text-align: center;'>Synthèse</h2>", unsafe_allow_html=True)
    # Calcul du makespan (temps total maximum)
    makespan = max(machine_times.values()) if machine_times else 0
    
    cols = st.columns(3)
    with cols[0]:
        st.metric("Nombre de machines", n_machines)
    with cols[1]:
        st.metric("Nombre d'articles", n_tasks)
    with cols[2]:
        # Conversion du makespan en heures
        makespan_hours = round(makespan / 60, 2)
        st.metric("Temps total de la cellule", f"{makespan_hours} heures")
        
    
    # Tableau détaillé
    st.write("**Détail par machine:**")
    detail_data = {
        "Machine": machines,
        "Nombre d'articles' affectés": [len(schedule.get(i, [])) for i in range(n_machines)],
        "Temps total machine (h)": [round(machine_times[i] / 60, 2) for i in range(n_machines)]
    }
    st.dataframe(detail_data, hide_index=True, use_container_width=True)
#_________________PROBLEME D'ORDRE_____________________________
def get_completion_times(schedule, P, S, nb_machines=2):
    n = len(P)
    completion_times = [0] * n
    machine_disponible = [0] * nb_machines
    taches_precedentes = [-1] * nb_machines

    for machine in range(nb_machines):
        for task in schedule[machine]:
            prec = taches_precedentes[machine]
            setup_time = 0 if prec == -1 else S[prec][task]
            start_time = machine_disponible[machine] + setup_time
            end_time = start_time + P[task]

            completion_times[task] = end_time
            machine_disponible[machine] = end_time
            taches_precedentes[machine] = task

    return completion_times
def GH_dynamique_disponibilite(P, S, M, D):

    n_tasks = P.shape[0]
    n_machines = P.shape[1]
    

    # --- Initialisation ---
    # Utiliser un set pour retirer efficacement les tâches planifiées
    unscheduled_tasks = set(range(n_tasks))
   
    # Calculer une seule fois les temps max pour la règle LPT
    max_time_per_task = np.max(P * M, axis=1)

    schedule = {m: [] for m in range(n_machines)}
    machine_completion_time = np.zeros(n_machines)
    task_completion = {}
    task_assignment = {}
    
    # --- Boucle principale ---
    while unscheduled_tasks:
        
        # Trouver les tâches dont la date de disponibilité est passée
        # par rapport au moment où les machines se libèrent.
        min_machine_ready_time = min(machine_completion_time) if len(task_completion) < n_machines else np.min(machine_completion_time)
        
        eligible_tasks = [t for t in unscheduled_tasks if D[t] <= min_machine_ready_time]

        # S'il n'y a aucune tâche disponible, on doit "sauter dans le temps".
        # On avance le temps jusqu'à la prochaine date de disponibilité.
        if not eligible_tasks:
            earliest_future_availability = min(D[t] for t in unscheduled_tasks)
            current_time = earliest_future_availability
            eligible_tasks = [t for t in unscheduled_tasks if D[t] <= current_time]
        
        # Appliquer la règle LPT : choisir la tâche la plus longue parmi les éligibles
        best_task_to_schedule = max(eligible_tasks, key=lambda task: max_time_per_task[task])

        # Maintenant, trouver la meilleure machine pour CETTE tâche
        best_machine = -1
        best_completion_time = float('inf')

        for machine in range(n_machines):
            if M[best_task_to_schedule, machine] == 1:
                last_task_on_machine = schedule[machine][-1] if schedule[machine] else None
                setup_time = 0 if last_task_on_machine is None else S[machine, last_task_on_machine, best_task_to_schedule]
                
                machine_ready_time = machine_completion_time[machine] + setup_time
                task_availability_time = D[best_task_to_schedule]
                
                start_time = max(machine_ready_time, task_availability_time)
                completion_time = start_time + P[best_task_to_schedule, machine]
                
                if completion_time < best_completion_time:
                    best_completion_time = completion_time
                    best_machine = machine
        
        # --- Mise à jour de l'état ---
        if best_machine != -1:
            # Assigner la tâche
            schedule[best_machine].append(best_task_to_schedule)
            machine_completion_time[best_machine] = best_completion_time
            task_assignment[best_task_to_schedule] = best_machine
            task_completion[best_task_to_schedule] = best_completion_time
            
            # Retirer la tâche de la liste d'attente
            unscheduled_tasks.remove(best_task_to_schedule)
        else:
            # Cas d'erreur : aucune machine trouvée, la boucle pourrait être infinie
            raise RuntimeError(f"Impossible d'assigner la tâche {best_task_to_schedule}. Vérifiez la matrice M.")

    # --- Finalisation ---
    cmax = np.max(machine_completion_time)


    sorted_tasks = sorted(task_completion.items(), key=lambda x: x[1])
    output_order = [task for task, _ in sorted_tasks]
    return schedule, cmax, task_assignment, output_order, task_completion  
#_________________MODELES P2|SETUPCMAX________________________
def H1(P, S, nb_machines):
    n = len(P)
    setup_start = [0] * n
    machine_disponible = [0] * nb_machines
    taches_precedentes = [-1] * nb_machines

    schedule = {m: [] for m in range(nb_machines)}
    ci = {m: 0 for m in range(nb_machines)}

    ordre = sorted(range(n), key=lambda t: P[t], reverse=True)

    for task in ordre:
        best_machine = None
        best_ci = float('inf')

        for machine in range(nb_machines):
            prec = taches_precedentes[machine]
            setup_time = setup_start[task] if prec == -1 else S[prec][task]
            new_ci = machine_disponible[machine] + setup_time + P[task]

            if new_ci < best_ci:
                best_ci = new_ci
                best_machine = machine

        schedule[best_machine].append(task)
        machine_disponible[best_machine] = best_ci
        taches_precedentes[best_machine] = task
        ci[best_machine] = best_ci

    cmax = int(max(ci.values()))
    
    return schedule, cmax
def H2(P, S, nb_machines): 
    n = len(P)
    setup_start = [0] * n  # Valeur par défaut : aucun setup initial
    taches_restantes = set(range(n))
    machine_disponible = [0] * nb_machines
    taches_precedentes = [-1] * nb_machines
    machine_assignments = []

    schedule = {m: [] for m in range(nb_machines)}  # Dictionnaire pour l'assignation des tâches
    ci = {m: 0 for m in range(nb_machines)}  # Dictionnaire pour le temps de fin de chaque machine

    # Étape 1 : assigner les deux plus longues tâches initialement
    taches_longues = sorted(taches_restantes, key=lambda t: P[t], reverse=True)[:nb_machines]

    for i in range(nb_machines):
        t = taches_longues[i]
        fin = machine_disponible[i] + setup_start[t] + P[t]
        machine_disponible[i] = fin
        taches_precedentes[i] = t
        taches_restantes.remove(t)
        schedule[i].append(t)  # Tâche assignée à la machine avec son temps de fin

    # Étape 2 : heuristique gloutonne avec setup
    while taches_restantes:
        meilleures_options = []

        for machine in range(nb_machines):
            for t in taches_restantes:
                prec = taches_precedentes[machine]
                setup_time = 0 if prec == -1 else S[prec][t]
                dispo = machine_disponible[machine]
                fin = dispo + setup_time + P[t]
                meilleures_options.append((fin, t, machine, setup_time))

        # Choix optimal
        meilleures_options.sort()
        temps_fin, tache_choisie, machine_choisie, setup_time = meilleures_options[0]

        machine_disponible[machine_choisie] = temps_fin
        taches_precedentes[machine_choisie] = tache_choisie
        taches_restantes.remove(tache_choisie)
        schedule[machine_choisie].append(tache_choisie)  # Ajouter la tâche à la machine

    # Calcul du Cmax
    cmax = int(max(machine_disponible))
    

    return schedule, cmax

def amelioration_locale_T(schedule, P, S):
    improved_schedule = {}
    improved_ci = {}
    improved_completion = {}

    for machine, tasks in schedule.items():
        order = []
        task_c = {}

        for task in tasks:
            best_order = None
            best_cmax = float('inf')
            best_completion = {}

            for i in range(len(order) + 1):
                temp_order = order[:i] + [task] + order[i:]
                current_time = 0.0
                temp_completion = {}

                for j, t in enumerate(temp_order):
                    if j == 0:
                        setup = 0.0
                    else:
                        prev = temp_order[j - 1]
                        setup = float(S[prev, t])
                    current_time += setup + float(P[t])
                    temp_completion[t] = current_time

                cmax = max(temp_completion.values())
                if cmax < best_cmax:
                    best_cmax = cmax
                    best_order = temp_order
                    best_completion = temp_completion

            order = best_order
            task_c = best_completion

        improved_schedule[machine] = order
        improved_ci[machine] = max(task_c.values()) if task_c else 0.0
        improved_completion.update(task_c)

    # Convertir total_cmax en float simple si c’est un array
    total_cmax = max(improved_ci.values()) if improved_ci else 0.0
    if isinstance(total_cmax, np.ndarray):
        total_cmax = total_cmax.item()

    return improved_schedule, total_cmax, improved_completion

def heuristique3(P, S):
    """
    Applique l'heuristique 3 pour le problème d'ordonnancement avec setups = 5
    
    P : liste des temps de traitement des tâches (indexés de 0 à n-1)
    S : matrice des setups S[i][j]
    
    Retourne : (schedule, Cmax)
    où schedule est un dictionnaire {machine_id: liste_des_tâches}
    et Cmax est le makespan
    """
    # Étape 1: Construction des sous-ensembles compatibles
    def construire_sous_ensembles(P, S):
        n = len(P)
        graphe = {i: set() for i in range(n)}
        
        for i in range(n):
            for j in range(i + 1, n):
                if S[i][j] == 5 and S[j][i] == 5:
                    graphe[i].add(j)
                    graphe[j].add(i)
        
        visites = [False] * n
        ensembles = []

        def dfs(u, composante):
            visites[u] = True
            composante.append(u)
            for v in graphe[u]:
                if not visites[v]:
                    dfs(v, composante)

        for i in range(n):
            if not visites[i]:
                composante = []
                dfs(i, composante)
                ensembles.append(composante)
        
        return ensembles
    
    # Étape 2: Application de LPT sur l'ensemble global des tâches compatibles
    def lpt_setup_5(P, taches):
        taches_triees = sorted(taches, key=lambda x: -P[x])
        
        machine1 = [taches_triees[0]]
        machine2 = [taches_triees[1]] if len(taches_triees) > 1 else []
        
        t1 = P[taches_triees[0]]
        t2 = P[taches_triees[1]] if len(taches_triees) > 1 else 0
        
        prec1 = taches_triees[0]
        prec2 = taches_triees[1] if len(taches_triees) > 1 else None
        
        reste = taches_triees[2:] if len(taches_triees) > 2 else []
        
        while reste:
            meilleur_C = float('inf')
            meilleur_j = -1
            meilleure_machine = None
            
            for j in reste:
                C1 = t1 + 5 + P[j]
                C2 = t2 + 5 + P[j]
                if C1 < meilleur_C:
                    meilleur_C = C1
                    meilleur_j = j
                    meilleure_machine = 1
                if C2 < meilleur_C:
                    meilleur_C = C2
                    meilleur_j = j
                    meilleure_machine = 2
            
            if meilleure_machine == 1:
                machine1.append(meilleur_j)
                t1 += 5 + P[meilleur_j]
                prec1 = meilleur_j
            else:
                machine2.append(meilleur_j)
                t2 += 5 + P[meilleur_j]
                prec2 = meilleur_j
            
            reste.remove(meilleur_j)
        
        Cmaxx = max(t1, t2)
        Cmax=float(Cmaxx)
        
        schedule = {0: machine1, 1: machine2}  # Format demandé
        return (schedule, Cmax)
    
    # Exécution de l'heuristique
    sous_ensembles = construire_sous_ensembles(P, S)
    
    # Regrouper toutes les tâches des sous-ensembles
    toutes_les_taches = set()
    for groupe in sous_ensembles:
        toutes_les_taches.update(groupe)
    toutes_les_taches = list(toutes_les_taches)
    
    # Appliquer LPT et retourner le schedule et Cmax
    return lpt_setup_5(P, toutes_les_taches)


#________________________MODELES RMPM|SETUP|CMAX______________________
# Définition de l'heuristique GH1
def GH1(P, S, M):
    n_tasks = P.shape[0]
    n_machines = P.shape[1]
    
    # Étape 1 : Générer l'ordre initial basé sur le temps maximal sur les machines éligibles
    max_time_per_task = np.zeros(n_tasks)
    for task in range(n_tasks):
        eligible_machines = np.where(M[task] == 1)[0]
        if len(eligible_machines) > 0:
            max_time_per_task[task] = np.max(P[task, eligible_machines])
        else:
            max_time_per_task[task] = 0
    ordre = np.argsort(-max_time_per_task).tolist()
    
    # Étape 2 : Appliquer l'heuristique GH1
    schedule = {m: [] for m in range(n_machines)}
    ci = {m: 0 for m in range(n_machines)}
    task_completion = {}
    task_assignment = {}
    
    for task in ordre:
        best_machine = None
        best_ci = float('inf')
        
        for machine in range(n_machines):
            if M[task, machine] == 1:
                setup_time = 0 if not schedule[machine] else S[machine, schedule[machine][-1], task]
                new_ci = ci[machine] + setup_time + P[task, machine]
                
                if new_ci < best_ci:
                    best_ci = new_ci
                    best_machine = machine
        
        if best_machine is not None:
            schedule[best_machine].append(task)
            ci[best_machine] = best_ci
            task_assignment[task] = best_machine
            task_completion[task] = best_ci
    
    # Ordre de sortie selon les temps d'achèvement croissants
    sorted_tasks = sorted(task_completion.items(), key=lambda x: x[1])
    output_order = [task for task, _ in sorted_tasks]
    
    cmax = max(ci.values())
    return schedule, cmax, task_assignment, output_order, task_completion, ordre
def GH2(P,S,M):
    num_tasks = len(P)
    num_machines = len(P[0])
    T = list(range(num_tasks))
    machines = list(range(num_machines))
    G = set(T)

    Vi = {i: {j for j in T if M[j][i]} for i in machines}
    Vi_copy = {i: Vi[i].copy() for i in machines}

    last_task_on_machine = {i: None for i in machines}
    schedule = {i: [] for i in machines}  # Liste simple des tâches
    end_time = {j: None for j in T}
    current_time = {i: 0.0 for i in machines}

    t = 0.0  # Horloge globale

    def compute_Rij(i, j, Vi_i):
        return sum(S[i][k][j] + P[j][i] for k in Vi_i) / len(Vi_i)

    while G:
        progress = False

        for i in machines:
            if current_time[i] > t:
                continue  # Machine occupée

            available_tasks = Vi_copy[i].intersection(G)
            if not available_tasks:
                continue

            R_values = {j: compute_Rij(i, j, available_tasks) for j in available_tasks}
            j_prime = min(R_values, key=R_values.get)

            k = last_task_on_machine[i]
            setup = S[i][k][j_prime] if k is not None else 0
            start = max(current_time[i], t)
            processing = P[j_prime][i]
            finish = start + setup + processing

            schedule[i].append(j_prime)
            current_time[i] = finish
            end_time[j_prime] = finish
            last_task_on_machine[i] = j_prime
            Vi_copy[i].remove(j_prime)
            G.remove(j_prime)

            progress = True

        if not progress:
            future_times = [current_time[i] for i in machines if Vi_copy[i].intersection(G)]
            if future_times:
                t = min(future_times)
            else:
                break
    if any(v is None for v in end_time.values()):
        raise ValueError("Certaines tâches n'ont pas été planifiées :", [j for j, v in end_time.items() if v is None])

    Cmax = max(end_time.values())
    return schedule, Cmax
def amelioration_locale(schedule, P, S):
    improved_schedule = {}
    improved_ci = {}
    improved_completion = {}

    for machine, tasks in schedule.items():
        order = []
        task_c = {}

        for task in tasks:
            best_order = None
            best_cmax = float('inf')
            best_completion = {}

            for i in range(len(order) + 1):
                temp_order = order[:i] + [task] + order[i:]
                current_time = 0
                temp_completion = {}

                for j, t in enumerate(temp_order):
                    if j == 0:
                        setup = 0
                    else:
                        setup = S[machine, temp_order[j - 1], t]
                    current_time += setup + P[t, machine]
                    temp_completion[t] = current_time

                cmax = max(temp_completion.values())
                if cmax < best_cmax:
                    best_cmax = cmax
                    best_order = temp_order
                    best_completion = temp_completion

            order = best_order
            task_c = best_completion

        improved_schedule[machine] = order
        improved_ci[machine] = max(task_c.values()) if task_c else 0
        improved_completion.update(task_c)

    total_cmax = max(improved_ci.values()) if improved_ci else 0
    return improved_schedule, total_cmax, improved_completion
### Fonctions pour le Recuit Simulé ###
def calculate_makespan(schedule, P, S):
    """Calcule le makespan d'un planning donné"""
    ci = {m: 0 for m in schedule.keys()}
    for machine, tasks in schedule.items():
        for i, task in enumerate(tasks):
            if i == 0:
                setup = 0
            else:
                setup = S[machine, tasks[i-1], task]
            ci[machine] += setup + P[task, machine]
    return max(ci.values())

def neighbor_solution(current_schedule):
    """Génère une solution voisine"""
    new_schedule = {m: current_schedule[m].copy() for m in current_schedule.keys()}
    machines = [m for m in new_schedule.keys() if len(new_schedule[m]) > 0]
    
    if random.random() < 0.5 and len(machines) >= 1:  # Swap
        m = random.choice(machines)
        if len(new_schedule[m]) >= 2:
            i, j = random.sample(range(len(new_schedule[m])), 2)
            new_schedule[m][i], new_schedule[m][j] = new_schedule[m][j], new_schedule[m][i]
    else:  # Déplacement
        m1 = random.choice(machines)
        task_idx = random.randint(0, len(new_schedule[m1])-1)
        task = new_schedule[m1].pop(task_idx)
        
        eligible_machines = [m for m in range(M.shape[1]) if M[task, m] == 1]
        m2 = random.choice(eligible_machines)
        insert_pos = random.randint(0, len(new_schedule[m2]))
        new_schedule[m2].insert(insert_pos, task)
    
    return new_schedule

def simulated_annealing(initial_schedule, P, S, M, 
                        initial_temp=1000, cooling_rate=0.95, 
                        min_temp=1, iterations_per_temp=100):
    
    current_schedule = {m: initial_schedule[m].copy() for m in initial_schedule.keys()}
    current_cost = calculate_makespan(current_schedule, P, S)
    best_schedule = {m: current_schedule[m].copy() for m in current_schedule.keys()}
    best_cost = current_cost
    
    # Historique pour la visualisation
    history = []
    temp_history = []
    best_history = []
    
    temp = initial_temp
    iteration = 0
    
    while temp > min_temp:
        for _ in range(iterations_per_temp):
            # Générer une solution voisine
            neighbor = neighbor_solution(current_schedule)
            neighbor_cost = calculate_makespan(neighbor, P, S)
            
            # Calculer la différence de coût
            cost_diff = neighbor_cost - current_cost
            
            # Accepter ou rejeter la solution
            if cost_diff < 0 or random.random() < math.exp(-cost_diff / temp):
                current_schedule = {m: neighbor[m].copy() for m in neighbor.keys()}
                current_cost = neighbor_cost
                
                # Mettre à jour la meilleure solution
                if current_cost < best_cost:
                    best_schedule = {m: current_schedule[m].copy() for m in current_schedule.keys()}
                    best_cost = current_cost
            
            # Enregistrement des données pour la visualisation
            history.append(current_cost)
            best_history.append(best_cost)
            temp_history.append(temp)
            iteration += 1
        
        # Refroidissement
        temp *= cooling_rate
    
    return best_schedule, best_cost, history, best_history, temp_history


### 1. Classes et fonctions de base ###
class Chromosome:
    def __init__(self, schedule=None):
        self.schedule = schedule if schedule else {}
        self.fitness = float('inf')
    
    def evaluate(self, P, S, M):
        ci = {m: 0 for m in self.schedule.keys()}
        for machine, tasks in self.schedule.items():
            for i, task in enumerate(tasks):
                if i == 0:
                    setup = 0
                else:
                    setup = S[machine, tasks[i-1], task]
                ci[machine] += setup + P[task, machine]
        self.fitness = max(ci.values())
        return self.fitness

def initialize_population(pop_size, P, S, M, schedule):
    population = []
    
    # Ajouter la solution GH1
    population.append(Chromosome(schedule))
    
    n_tasks = P.shape[0]
    n_machines = M.shape[1]
    
    for _ in range(pop_size - 1):
        new_schedule = {m: [] for m in range(n_machines)}
        tasks = list(range(n_tasks))
        random.shuffle(tasks)
        
        for task in tasks:
            eligible_machines = [m for m in range(n_machines) if M[task, m] == 1]
            if eligible_machines:
                m = random.choice(eligible_machines)
                new_schedule[m].append(task)
        
        population.append(Chromosome(new_schedule))
    
    return population

### 2. Opérateurs génétiques ###
def crossover(parent1, parent2, P, S, M):
    child_schedule = {}
    machines = list(parent1.schedule.keys())
    
    for m in machines:
        if random.random() < 0.5:
            child_schedule[m] = parent1.schedule[m].copy()
        else:
            child_schedule[m] = parent2.schedule[m].copy()
    
    all_tasks = set(range(P.shape[0]))
    present_tasks = set(task for tasks in child_schedule.values() for task in tasks)
    
    missing_tasks = all_tasks - present_tasks
    for task in missing_tasks:
        eligible_machines = [m for m in range(M.shape[1]) if M[task, m] == 1]
        if eligible_machines:
            m = random.choice(eligible_machines)
            child_schedule[m].append(task)
    
    task_counts = {}
    for tasks in child_schedule.values():
        for task in tasks:
            task_counts[task] = task_counts.get(task, 0) + 1
    
    for task, count in task_counts.items():
        if count > 1:
            for m in child_schedule:
                if task in child_schedule[m]:
                    child_schedule[m] = [t for t in child_schedule[m] if t != task]
                    child_schedule[m].append(task)
                    break
    
    return Chromosome(child_schedule)

def mutate(chromosome, P, S, M, mutation_rate=0.1):
    mutated = deepcopy(chromosome)
    
    for m in mutated.schedule:
        if len(mutated.schedule[m]) > 1 and random.random() < mutation_rate:
            i, j = random.sample(range(len(mutated.schedule[m])), 2)
            mutated.schedule[m][i], mutated.schedule[m][j] = mutated.schedule[m][j], mutated.schedule[m][i]
    
    if random.random() < mutation_rate:
        task = random.randint(0, P.shape[0]-1)
        current_machine = None
        for m in mutated.schedule:
            if task in mutated.schedule[m]:
                current_machine = m
                break
        
        if current_machine is not None:
            eligible_machines = [m for m in range(M.shape[1]) if M[task, m] == 1 and m != current_machine]
            if eligible_machines:
                new_machine = random.choice(eligible_machines)
                mutated.schedule[current_machine].remove(task)
                insert_pos = random.randint(0, len(mutated.schedule[new_machine]))
                mutated.schedule[new_machine].insert(insert_pos, task)
    
    return mutated

def selection(population, tournament_size=3):
    selected = []
    for _ in range(len(population)):
        contestants = random.sample(population, tournament_size)
        winner = min(contestants, key=attrgetter('fitness'))
        selected.append(deepcopy(winner))
    return selected

### 3. Algorithme génétique principal ###
def genetic_algorithm(P, S, M, schedule, pop_size=50, generations=100, 
                      crossover_rate=0.8, mutation_rate=0.1, elite_size=2):
    population = initialize_population(pop_size, P, S, M, schedule)
    for ind in population:
        ind.evaluate(P, S, M)
    
    best_fitness_history = []
    
    for gen in range(generations):
        population.sort(key=attrgetter('fitness'))
        best_fitness = population[0].fitness
        best_fitness_history.append(best_fitness)
        
        new_population = population[:elite_size]
        selected = selection(population)
        
        while len(new_population) < pop_size:
            parent1, parent2 = random.sample(selected, 2)
            
            if random.random() < crossover_rate:
                child = crossover(parent1, parent2, P, S, M)
            else:
                child = deepcopy(random.choice([parent1, parent2]))
            
            child = mutate(child, P, S, M, mutation_rate)
            child.evaluate(P, S, M)
            new_population.append(child)
        
        population = new_population
    
    population.sort(key=attrgetter('fitness'))
    return population[0], best_fitness_history


#-----------------PARAMETRES METAHEURISTIQUES
POP_SIZE = 50
GENERATIONS = 100
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1
ELITE_SIZE = 2

INITIAL_TEMP=1000
COOLING_RATE=0.95
MIN_TEMP=1
ITERATIONS_PER_TEMP=100


# Définir les options de choix
options = ["Ordonnancement synchronisé","Ordonnancement par cellule","Aide","À propos"]
st.sidebar.image("MINIROS.png",  width=200)
#st.sidebar.image("USTHB.png.png",  width=100)
# Demander à l'utilisateur de choisir une option
choix = st.sidebar.radio("Choisir:", options)



###CARACTERISATION 1 ###
if choix == "Aide":
    st.markdown("""
        ### Instructions d'utilisation :

        1. **Choisir** l'option *Outils d'ordonnancement* dans le menu latéral  
        2. **Importer le fichier Excel du Work order** contenant Les 'ARTICLES' planifiés, et leur 'QUANTITE'.
        3. **Sélectionner** les machines disponibles par cellule, et par atelier  
        4. **Lancer l’optimisation** en cliquant sur le bouton 'START', en veillant à ce qu'au moins une machine par cellule soit sélectionnée.  
        5. **Consulter les ordres de fabrication** affichés automatiquement pour chaque cellule, sous forme de Gantt avec une synthèse de l'opération.  
        6. **Optionnel:** Explorer d'autres solutions avec *Afficher les différents ordonnancements possibles*  
        7. **Exporter l'ordonnancement optimisé** au format `.xlsx` pour usage externe.
        """)  
if choix == "À propos":
    # 📷 Charger les images (remplace par les chemins de tes fichiers)
    image1 = Image.open("USTHB.png.png")  # ou .jpg
    image2 = Image.open("MINIROS.png")

    # 🖼️ Afficher les images côte à côte
    col0, col1, col_space, col2, col3 = st.columns([1, 1.5, 2, 1.5, 1])
    with col1:
        st.image(image1, width=100)
    with col2:
        st.image(image2, width=190)
    st.write('')
    # 📝 Introduction
    st.markdown("""
    Bienvenue sur cette application dédiée à l'optimisation de la planification, réalisée au sein de **MINIROS**.  
    Notre travail porte sur l'optimisation de l’ordonnancement au niveau des ateliers **ROULEAUX** et **PINCEAUX**.

    Cette application a été développée par : **BASLIMANE Baya** & **BELAMINE Meriem**

    Spécialité: Recherche Opérationnelle - Modèles et Méthodes pour l’Ingénierie et la Recherche

    **Contacts :**
    - baya.baslimane@gmail.com  
    - myriamblmn@gmail.com
    """)
  
if choix == "Ordonnancement par cellule":
    st.header('Work order du jour:')
    file = st.file_uploader(':orange[IMPORTER le WO du jour]', type=['xlsx'])
    # Vérifier si un fichier a été téléchargé
    if file is not None:
        # Lire le fichier CSV en utilisant Pandas
        wo = pd.read_excel(file)
        st.write("WO importé:")
        st.write(wo)  
        st.header('Machines disponibles:')

        # Créer deux colonnes
        col1, col2 = st.columns(2)

        # ----- COLONNE GAUCHE : Cellule Découpe Manchons -----
        with col1:
            #st.subheader('Cellule Découpe Manchons:')
            st.subheader('Atelier Rouleau:')


            # Initialisation des clés pour découpe
            for key in ["select_all_DECOUPE", "option_a", "option_b", "option_c", "option_d"]:
                if key not in st.session_state:
                    st.session_state[key] = False
            for key in ["select_all_T", "option_j", "option_k"]:
                if key not in st.session_state:
                    st.session_state[key] = False

            def select_all_decoupe():
                st.session_state.option_a = True
                st.session_state.option_b = True
                st.session_state.option_c = True
                st.session_state.option_d = True
            def select_all_T():
                st.session_state.option_j = True
                st.session_state.option_k = True

            st.write("**Cellule Découpe Manchons**")    
            


            st.checkbox("Tout sélectionner", key="select_all_DECOUPE", on_change=select_all_decoupe)
            st.checkbox("Machine Edward Jackson", key="option_a")
            st.checkbox("Machine 3 Lames", key="option_b")
            st.checkbox("Machine 4 Lames 1", key="option_c")
            st.checkbox("Machine 4 Lames 2", key="option_d")
            st.write("**Cellule Thermofusion**")
            st.checkbox("Tout sélectionner", key="select_all_T", on_change=select_all_T)
            st.checkbox("Machine thermofusion 1", key="option_j")
            st.checkbox("Machine thermofusion 2", key="option_k")

            selected_DECOUPE = []
            if st.session_state.option_a:
                selected_DECOUPE.append("EDJ")
            if st.session_state.option_b:
                selected_DECOUPE.append("3L")
            if st.session_state.option_c:
                selected_DECOUPE.append("4L1")
            if st.session_state.option_d:
                selected_DECOUPE.append("4L2")
            selected_T = []
            if st.session_state.option_j:
                selected_T.append("Machine thermo 1")
            if st.session_state.option_k:
                selected_T.append("Machine thermo 2")


        # ----- COLONNE DROITE : Cellule Préparation Têtes de Pinceaux -----
        with col2:
            #st.subheader('Cellule Préparation Têtes de Pinceaux:')
            st.subheader('Atelier Pinceaux:')

            # Initialisation des clés pour pinceaux
            for key in ["select_all_PNX", "option_e", "option_f", "option_g", "option_h", "option_i"]:
                if key not in st.session_state:
                    st.session_state[key] = False

            def select_all_pnx():
                st.session_state.option_e = True
                st.session_state.option_f = True
                st.session_state.option_g = True
                st.session_state.option_h = True
                st.session_state.option_i = True
            st.write('**Cellule Préparation Têtes de Pinceaux**')

            st.checkbox("Tout sélectionner", key="select_all_PNX", on_change=select_all_pnx)
            st.checkbox("Table Manuelle 1", key="option_e")
            st.checkbox("Table Manuelle 2", key="option_f")
            st.checkbox("Table Manuelle 3", key="option_g")
            st.checkbox("Machine Automatique Plat", key="option_h")
            st.checkbox("Machine Automatique Rond", key="option_i")

            selected_PNX = []
            if st.session_state.option_e:
                selected_PNX.append("TM1")
            if st.session_state.option_f:
                selected_PNX.append("TM2")
            if st.session_state.option_g:
                selected_PNX.append("TM3")
            if st.session_state.option_h:
                selected_PNX.append("MAP")
            if st.session_state.option_i:
                selected_PNX.append("MAR")
    



        
        #________________EXTRACTION FROM EXCEL________________________
        articles = wo['ARTICLE'].tolist()
        quantites = wo['QUANTITE'].tolist()
        #-----------------EXTRACTION THERMO----------------------------
        liste_thermo= pd.read_excel(DATAS, sheet_name='THERMOListe', index_col=0)
        P_thermo = pd.read_excel(DATAS, sheet_name='THERMOP', index_col=0, header=None)
        S_thermo= pd.read_excel(DATAS, sheet_name='THERMOS', index_col=0)
        corr=pd.read_excel(DATAS, sheet_name='THERMOcorr')
        coef=pd.read_excel(DATAS, sheet_name='THERMOcoef')

        #-----------------EXTRACTION DECOUPE----------------------------------
        liste_decoupe= pd.read_excel(DATAS, sheet_name='DECOUPEListe', index_col=0)
        P_decoupe = pd.read_excel(DATAS, sheet_name='DECOUPEP', index_col=0)
        M_decoupe = pd.read_excel(DATAS, sheet_name='DECOUPEM', index_col=0)
        EDJo = pd.read_excel(DATAS, sheet_name='DECOUPEEDJ', index_col=0)
        L3o = pd.read_excel(DATAS, sheet_name='DECOUPE3L', index_col=0)
        L4o = pd.read_excel(DATAS, sheet_name='DECOUPE4L', index_col=0)

        #-------------------------EXTRACTION PNX----------------------------
        liste_pnx = pd.read_excel(DATAS, sheet_name='PNXListe', index_col=0)
        TMo = pd.read_excel(DATAS, sheet_name='PNXTM', index_col=0)
        MAPo = pd.read_excel(DATAS, sheet_name='PNXPLAT', index_col=0)
        MARo = pd.read_excel(DATAS, sheet_name='PNXROND', index_col=0)
        P_pnx = pd.read_excel(DATAS, sheet_name='PNXP', index_col=0)
        M_pnx = pd.read_excel(DATAS, sheet_name='PNXM', index_col=0)

        #----------------------ARTICLES?????????????????
        def chercher_articles_et_quantites_par_feuille(fichier_excel, articles, quantites, feuilles=["DECOUPEListe", "THERMOcorr", "PNXListe"]):
            # Charger le fichier Excel
            xls = pd.ExcelFile(fichier_excel)
            
            # Construire un dictionnaire article -> quantité
            dict_articles_quantites = dict(zip(articles, quantites))
            
            # Fonction interne pour trouver les articles dans une feuille
            def trouver_articles(df, articles):
                trouvés = set()
                for col in df.columns:
                    for val in df[col].dropna().astype(str):
                        if val in articles:
                            trouvés.add(val)
                return list(trouvés)

            résultats_articles = {}
            résultats_quantites = {}

            for feuille in feuilles:
                df = xls.parse(feuille)
                art_trouves = trouver_articles(df, articles)
                résultats_articles[feuille] = art_trouves
                # Récupérer la quantité correspondant à chaque article trouvé
                quant_trouvees = [dict_articles_quantites[art] for art in art_trouves]
                résultats_quantites[feuille] = quant_trouvees
            
            return (résultats_articles["DECOUPEListe"], résultats_quantites["DECOUPEListe"],
                    résultats_articles["THERMOcorr"], résultats_quantites["THERMOcorr"],
                    résultats_articles["PNXListe"], résultats_quantites["PNXListe"])

        decoupe, decoupe_q, thermo, thermo_q, pnx, pnx_q = chercher_articles_et_quantites_par_feuille(DATAS, articles, quantites)
        # Création des listes d'index simples
        liste_decoupe = list(range(len(decoupe)))
        liste_thermo = list(range(len(thermo)))
        liste_pnx = list(range(len(pnx)))

        #_________________CONSTRUCTION DES MATRICES________________________
        #-----------------        THERMO           ------------------------
        thermo_q_array = np.array(thermo_q)
        sf_list = []
        coefs = []

        for produit in thermo:
            ligne_corresp = corr[corr['article'] == produit]
            ligne_coef = coef[coef['article'] == produit]

            if not ligne_corresp.empty and not ligne_coef.empty:
                sf = ligne_corresp['sf'].iloc[0]
                coef_val = ligne_coef['coef'].iloc[0]  # Remplacez 'coef' par le vrai nom de colonne dans coef
                sf_list.append(sf)
                coefs.append(coef_val)
            else:
                st.write(f"Produit inconnu ou coefficient manquant : {produit}")
        COEFS = np.array(coefs)
        P_thermo1= P_thermo.loc[sf_list]
        S_thermo1 = S_thermo.loc[sf_list,sf_list]
        S_THERMO = np.array(S_thermo1.values)
        P_thermo_iter = np.array(P_thermo1.values)
        P_THERMO = P_thermo_iter * thermo_q_array[:, np.newaxis] / COEFS[:, np.newaxis]

        #-----------------        DECOUPE          ------------------------
        # Extraire la sous-matrice
        decoupe_q_array = np.array(decoupe_q)
        M_decoupe1 = M_decoupe.loc[decoupe,selected_DECOUPE]
        P_decoupe1 = P_decoupe.loc[decoupe,selected_DECOUPE]
        # Dictionnaire de correspondance entre noms et matrices sources
        matrices_source_DECOUPE = {
            'EDJ': EDJo,
            '3L': L3o,
            '4L1': L4o,
            '4L2': L4o
        }

        # Création de la liste des sous-matrices
        matrices_list_DECOUPE = [matrices_source_DECOUPE[item].loc[decoupe, decoupe].values for item in selected_DECOUPE]

        # Conversion en matrice NumPy
        S_DECOUPE = np.array(matrices_list_DECOUPE)
        M_DECOUPE= np.array( M_decoupe1.values)
        P_decoupe_iter = np.array(P_decoupe1.values)
        # Reshape du vecteur en colonne pour un broadcasting ligne par ligne
        P_DECOUPE = P_decoupe_iter * decoupe_q_array[:, np.newaxis]

        #-----------------        PNX              ------------------------
        pnx_q_array = np.array(pnx_q)
        M_pnx1 = M_pnx.loc[pnx,selected_PNX]
        P_pnx1 = P_pnx.loc[pnx,selected_PNX]
        # Dictionnaire de correspondance entre noms et matrices sources
        matrices_source_PNX = {
            'TM1': TMo,
            'TM2': TMo,
            'TM3': TMo,
            'MAP': MAPo,
            'MAR': MARo      }
        matrices_list_PNX = [matrices_source_PNX[item].loc[pnx,pnx].values for item in selected_PNX]
        S_PNX = np.array(matrices_list_PNX)
        M_PNX = np.array(M_pnx1 .values)
        P_pnx_iter = np.array(P_pnx1.values)
        P_PNX = P_pnx_iter * pnx_q_array[:, np.newaxis]
        valid_selection = (
            len(selected_DECOUPE) > 0 and
            len(selected_T) > 0 and
            len(selected_PNX) > 0
        )

        # Affichage du bouton Start
        start_button = st.button("Start", disabled=not valid_selection)
        if start_button:
            st.success("Lancement de la moulinette...")
            #________________________APPEL DES FONCTIONS______________________
            #-----------------------------THERMO---------------------
            #HEURISTIQUE1
            nmt=len(selected_T)
            schedule1_T_O, cmax1_T_O = H1(P_THERMO, S_THERMO, nmt)
            schedule1_T, cmax1_T,*_=amelioration_locale_T(schedule1_T_O, P_THERMO, S_THERMO)

            #HEURISTIQUE2
            schedule2_T_O, cmax2_T_O = H2(P_THERMO, S_THERMO, nmt)
            schedule2_T, cmax2_T,*_=amelioration_locale_T(schedule2_T_O, P_THERMO, S_THERMO)
            #HEURISTIQUE3
            #schedule3_T, cmax3_T = heuristique3(P_THERMO, S_THERMO)
            #BEST ONE
            solutions_heuristiques_T = {
                "Heuristique1": (schedule1_T, cmax1_T),
                "Heuristique2": (schedule2_T, cmax2_T)}
            best_label_T = min(solutions_heuristiques_T, key=lambda k: solutions_heuristiques_T[k][1])
            worst_label_T = max(solutions_heuristiques_T, key=lambda k: solutions_heuristiques_T[k][1])
            best_schedule_T, best_cmax_T = solutions_heuristiques_T[best_label_T]
            worst_schedule_T, worst_cmax_T = solutions_heuristiques_T[worst_label_T]
            
            # Trie les solutions par Cmax croissant
            sorted_solutions_T= sorted(solutions_heuristiques_T.items(), key=lambda x: x[1][1])

            # Récupère les 3 meilleures solutions
            top_3_solutions_T= sorted_solutions_T[:2]

            # Crée deux listes : une pour les schedules, une pour les Cmax
            top_3_schedules_PNX = [sol[1][0] for sol in top_3_solutions_T]
            top_3_cmax_PNX = [sol[1][1] for sol in top_3_solutions_T]

            #------------------------DECOUPE-------------------------
            M=M_DECOUPE
            #HEURISTIQUE1
            schedule1_DECOUPE, cmax1_DECOUPE,*_ = GH1(P=P_DECOUPE, S=S_DECOUPE, M=M_DECOUPE)   
            #HEURISTIQUE1 + AMELIORATION
            better_schedule1_DECOUPE, better_cmax1_DECOUPE, _ = amelioration_locale(schedule1_DECOUPE, P=P_DECOUPE, S=S_DECOUPE)
            #HEURISTIQUE2
            #schedule2_DECOUPE, cmax2_DECOUPE = GH2(P=P_DECOUPE, S=S_DECOUPE, M=M_DECOUPE)
            #HEURISTIQUE2 + AMELIORATION
            #better_schedule2_DECOUPE, better_cmax2_DECOUPE, _ = amelioration_locale(schedule2_DECOUPE, P=P_DECOUPE, S=S_DECOUPE)      
            #Best Cmax pour solution innitiale
            best_cmax_DECOUPE = min(cmax1_DECOUPE,better_cmax1_DECOUPE)
            schedule_map_DECOUPE = {
                cmax1_DECOUPE: schedule1_DECOUPE,
                better_cmax1_DECOUPE: better_schedule1_DECOUPE
            }
            best_schedule_DECOUPE = schedule_map_DECOUPE[best_cmax_DECOUPE]      
            #AG
            AG_schedule_DECOUPE, history_AG_DECOUPE = genetic_algorithm(
            P=P_DECOUPE, S=S_DECOUPE, M=M_DECOUPE,
            schedule=best_schedule_DECOUPE,
            pop_size=POP_SIZE,
            generations=GENERATIONS,
            crossover_rate=CROSSOVER_RATE,
            mutation_rate=MUTATION_RATE,
            elite_size=ELITE_SIZE)
            #RS
            RS_schedule_DECOUPE, best_cmax_RS_DECOUPE, history_RS_DECOUPE, best_history, temp_history = simulated_annealing(
            best_schedule_DECOUPE, P=P_DECOUPE, S=S_DECOUPE, M=M_DECOUPE,
            initial_temp=INITIAL_TEMP, 
            cooling_rate=COOLING_RATE,
            min_temp=MIN_TEMP,
            iterations_per_temp=ITERATIONS_PER_TEMP)
            #BEST ONE
            solutions_heuristiques_DECOUPE = {
                "Heuristique1": (schedule1_DECOUPE, cmax1_DECOUPE),
                "Heuristique1 Améliorée": (better_schedule1_DECOUPE, better_cmax1_DECOUPE),
                "Algorithme Génétique":(RS_schedule_DECOUPE,best_cmax_RS_DECOUPE),
                "Recuit Simulé": (RS_schedule_DECOUPE,best_cmax_RS_DECOUPE)}
            best_label_DECOUPE = min(solutions_heuristiques_DECOUPE, key=lambda k: solutions_heuristiques_DECOUPE[k][1])
            worst_label_DECOUPE = max(solutions_heuristiques_DECOUPE, key=lambda k: solutions_heuristiques_DECOUPE[k][1])
            best_schedule_DECOUPE, best_cmax_DECOUPE = solutions_heuristiques_DECOUPE[best_label_DECOUPE]
            worst_schedule_DECOUPE, worst_cmax_DECOUPE = solutions_heuristiques_DECOUPE[worst_label_DECOUPE]
            # Trie les solutions par Cmax croissant
            sorted_solutions_DECOUPE = sorted(solutions_heuristiques_DECOUPE.items(), key=lambda x: x[1][1])

            # Récupère les 3 meilleures solutions
            top_3_solutions_DECOUPE = sorted_solutions_DECOUPE[:2]
            # Crée deux listes : une pour les schedules, une pour les Cmax
            top_3_schedules_DECOUPE = [sol[1][0] for sol in top_3_solutions_DECOUPE]
            top_3_cmax_DECOUPE = [sol[1][1] for sol in top_3_solutions_DECOUPE]

            #-----------------------------------PNX-----------------------------------------
            M=M_PNX
            #HEURISTIQUE1
            schedule1_PNX, cmax1_PNX,*_ = GH1(P=P_PNX, S=S_PNX, M=M_PNX)   
            #HEURISTIQUE1 + AMELIORATION
            better_schedule1_PNX, better_cmax1_PNX, _ = amelioration_locale(schedule1_PNX, P=P_PNX, S=S_PNX)       
            #HEURISTIQUE2
            #schedule2_PNX, cmax2_PNX = GH2(P=P_PNX, S=S_PNX, M=M_PNX)
            #HEURISTIQUE2 + AMELIORATION
            #better_schedule2_PNX, better_cmax2_PNX, _ = amelioration_locale(schedule2_PNX, P=P_PNX, S=S_PNX)
            #Best Cmax pour solution innitiale
            best_cmax_PNX = min(cmax1_PNX,better_cmax1_PNX)
            schedule_map_PNX = {
                cmax1_PNX: schedule1_PNX,
                better_cmax1_PNX: better_schedule1_PNX
            }
            best_schedule_PNX = schedule_map_PNX[best_cmax_PNX]
            #AG
            AG_schedule_PNX, history_AG_PNX = genetic_algorithm(
            P=P_PNX, S=S_PNX, M=M_PNX,
            schedule=best_schedule_PNX,
            pop_size=POP_SIZE,
            generations=GENERATIONS,
            crossover_rate=CROSSOVER_RATE,
            mutation_rate=MUTATION_RATE,
            elite_size=ELITE_SIZE)
            #RS
            RS_schedule_PNX, best_cmax_RS_PNX, history_RS_PNX, best_history, temp_history = simulated_annealing(
            best_schedule_PNX, P=P_PNX, S=S_PNX, M=M_PNX,
            initial_temp=INITIAL_TEMP, 
            cooling_rate=COOLING_RATE,
            min_temp=MIN_TEMP,
            iterations_per_temp=ITERATIONS_PER_TEMP)
            #BEST ONE
            solutions_heuristiques_PNX = {
                "Heuristique1": (schedule1_PNX, cmax1_PNX),
                "Heuristique1 Améliorée": (better_schedule1_PNX, better_cmax1_PNX),
                "Algorithme Génétique":(RS_schedule_PNX,best_cmax_RS_PNX),
                "Recuit Simulé": (RS_schedule_PNX,best_cmax_RS_PNX)}
            best_label_PNX = min(solutions_heuristiques_PNX, key=lambda k: solutions_heuristiques_PNX[k][1])
            worst_label_PNX = max(solutions_heuristiques_PNX, key=lambda k: solutions_heuristiques_PNX[k][1])
            best_schedule_PNX, best_cmax_PNX = solutions_heuristiques_PNX[best_label_PNX]
            worst_schedule_PNX, worst_cmax_PNX = solutions_heuristiques_PNX[worst_label_PNX]
            # Trie les solutions par Cmax croissant
            sorted_solutions_PNX = sorted(solutions_heuristiques_PNX.items(), key=lambda x: x[1][1])

            # Récupère les 3 meilleures solutions
            top_3_solutions_PNX= sorted_solutions_PNX[:2]

            # Crée deux listes : une pour les schedules, une pour les Cmax
            top_3_schedules_PNX = [sol[1][0] for sol in top_3_solutions_PNX]
            top_3_cmax_PNX = [sol[1][1] for sol in top_3_solutions_PNX]
            # Création des lignes pour chaque ordonnancement
            def build_schedule_dict(best_schedule, articles, machines):
                data = []
                for machine_id, article_indices in best_schedule.items():
                    for ordre, idx in enumerate(article_indices, start=1):
                        data.append({
                            "ARTICLE": articles[idx],
                            "Machine": machines[machine_id],
                            "Ordre": ordre
                        })
                return pd.DataFrame(data)
            #MACHINES
            
            machine_NOM_DECOUPE= {
                    'EDJ': "Machine Découpe EDJ",
                    '3L': "Machine Découpe 3L",
                    '4L1': "Machine Découpe 4L1",
                    '4L2': "Machine Découpe 4L2"
                }
            machine_NOM_PNX = {
                    'TM1': "Table Manuelle 1",
                    'TM2': "Table Manuelle 2",
                    'TM3': "Table Manuelle 3",
                    'MAP': "Machine Plat",
                    'MAR': "Machine Rond"
                }
            
            selected_names_DECOUPE = [machine_NOM_DECOUPE[machine] for machine in selected_DECOUPE]
            selected_names_PNX = [machine_NOM_PNX[machine] for machine in selected_PNX]
        
            df_thermo = build_schedule_dict(best_schedule_T, thermo, selected_T)
            df_pnx = build_schedule_dict(best_schedule_PNX, pnx, selected_names_PNX)
            df_decoupe = build_schedule_dict(best_schedule_DECOUPE, decoupe, selected_names_DECOUPE)

            # Fusion des deux ordonnancements
            df_merge = pd.concat([df_pnx, df_decoupe,df_thermo])

            # Fusion avec le fichier Excel de base
            df_final = wo.merge(df_merge, on="ARTICLE", how="left")
            # 1. Afficher le DataFrame
            st.subheader("📊 ORDONNANCEMENT")
            st.dataframe(df_final)

            # 2. Convertir en fichier Excel en mémoire
            def convert_df_to_excel(df):
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='Ordonnance')
                output.seek(0)
                return output

            # 3. Bouton de téléchargement
            excel_file = convert_df_to_excel(df_final)

            st.download_button(
                label="📥 Télécharger en Excel",
                data=excel_file,
                file_name="WO_RECOLTE_ORDONNANCE.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            #st.header('Cellule Thermo ')
            #MT=['M1','M2']
            #plot_gantt(S_THERMO, P_THERMO, best_schedule_T, thermo, MT)
            st.header('Cellule Thermofusion ')
            n_T=len(selected_T)
            with st.expander("Afficher les differents ordonnancements possibles"):
                               # Pré-calcul des temps machines pour toutes les solutions
                all_temps_machines_T = []
                for _, (schedule, _) in top_3_solutions_T:
                    all_temps_machines_T.append(calculer_temps_total_T(schedule, P_THERMO, S_THERMO))

                # Affichage
                for idx, ((label, (schedule, cmax)), temps_machines) in enumerate(zip(top_3_solutions_T, all_temps_machines_T), start=1):
                    st.write(f"## 💡 Solution {idx}:")
                    for machine in range(n_T):
                        tasks = schedule[machine]
                        noms_articles = [thermo[i] for i in tasks]
                        machine_time = temps_machines[machine]
                        st.write(f"**{selected_T[machine]}**: {noms_articles} (**{machine_time:.2f}** heures)")
                    st.success(f"**Cmax**: {cmax / 60:.2f} heures\n")
            plot_gantt_T(S_THERMO, P_THERMO, schedule1_T,thermo, selected_T)
            st.header('Cellule Découpe Manchons ')
            n_DECOUPE=len(selected_DECOUPE)
            with st.expander("Afficher les differents ordonnancements possibles"):
                # Pré-calcul des temps machines pour toutes les solutions
                all_temps_machines_DECOUPE = []
                for _, (schedule, _) in top_3_solutions_DECOUPE:
                    all_temps_machines_DECOUPE.append(calculer_temps_total(schedule, P_DECOUPE, S_DECOUPE))

                # Affichage
                for idx, ((label, (schedule, cmax)), temps_machines) in enumerate(zip(top_3_solutions_DECOUPE, all_temps_machines_DECOUPE), start=1):
                    st.write(f"#💡 Solution {idx}:")
                    for machine in range(n_DECOUPE):
                        tasks = schedule[machine]
                        noms_articles = [decoupe[i] for i in tasks]
                        machine_time = temps_machines[machine]
                        st.write(f"**{selected_DECOUPE[machine]}**: {noms_articles} (**{machine_time:.2f}** heures)")
                    st.success(f"**Cmax**: {cmax / 60:.2f} heures\n")
                


            plot_gantt(S_DECOUPE, P_DECOUPE, best_schedule_DECOUPE, decoupe, selected_DECOUPE)
            st.header('Cellule Préparation Têtes de Pinceaux ')
            n_PNX=len(selected_PNX)
            with st.expander("Afficher les differents ordonnancement possible pour la cellule PNX"):
                # Pré-calcul des temps machines pour toutes les solutions
                all_temps_machines_PNX = []
                for _, (schedule, _) in top_3_solutions_PNX:
                    all_temps_machines_PNX.append(calculer_temps_total(schedule, P_PNX, S_PNX))

                # Affichage
                for idx, ((label, (schedule, cmax)), temps_machines) in enumerate(zip(top_3_solutions_PNX, all_temps_machines_PNX), start=1):
                    st.write(f"#💡 Solution{idx}:")
                    for machine in range(n_PNX):
                        tasks = schedule[machine]
                        noms_articles = [pnx[i] for i in tasks]
                        machine_time = temps_machines[machine]
                        st.write(f"**{selected_PNX[machine]}**: {noms_articles} (**{machine_time:.2f}** heures)")
                    st.success(f"**Cmax**: {cmax / 60:.2f} heures\n")
            plot_gantt(S_PNX, P_PNX, best_schedule_PNX, pnx, selected_PNX)
            

    else:
        st.write("Aucun fichier Excel n'a été téléchargé.")
if choix == "Ordonnancement synchronisé":
    st.header('Work order du jour:')
    file = st.file_uploader(':orange[IMPORTER le WO du jour]', type=['xlsx'])
    # Vérifier si un fichier a été téléchargé
    if file is not None:
        # Lire le fichier CSV en utilisant Pandas
        wo = pd.read_excel(file)
        st.write("WO importé:")
        st.write(wo)  
        st.header('Machines disponibles:')

        # Créer deux colonnes
        col1, col2 = st.columns(2)

        # ----- COLONNE GAUCHE : Cellule Découpe Manchons -----
        with col1:
            #st.subheader('Cellule Découpe Manchons:')
            st.subheader('Atelier Rouleau:')


            # Initialisation des clés pour découpe
            for key in ["select_all_DECOUPE", "option_a", "option_b", "option_c", "option_d"]:
                if key not in st.session_state:
                    st.session_state[key] = False
            for key in ["select_all_T", "option_j", "option_k"]:
                if key not in st.session_state:
                    st.session_state[key] = False

            def select_all_decoupe():
                st.session_state.option_a = True
                st.session_state.option_b = True
                st.session_state.option_c = True
                st.session_state.option_d = True
            def select_all_T():
                st.session_state.option_j = True
                st.session_state.option_k = True

            st.write("**Cellule Découpe Manchons**")    
            


            st.checkbox("Tout sélectionner", key="select_all_DECOUPE", on_change=select_all_decoupe)
            st.checkbox("Machine Edward Jackson", key="option_a")
            st.checkbox("Machine 3 Lames", key="option_b")
            st.checkbox("Machine 4 Lames 1", key="option_c")
            st.checkbox("Machine 4 Lames 2", key="option_d")
            st.write("**Cellule Thermofusion**")
            st.checkbox("Tout sélectionner", key="select_all_T", on_change=select_all_T)
            st.checkbox("Machine thermofusion 1", key="option_j")
            st.checkbox("Machine thermofusion 2", key="option_k")

            selected_DECOUPE = []
            if st.session_state.option_a:
                selected_DECOUPE.append("EDJ")
            if st.session_state.option_b:
                selected_DECOUPE.append("3L")
            if st.session_state.option_c:
                selected_DECOUPE.append("4L1")
            if st.session_state.option_d:
                selected_DECOUPE.append("4L2")
            selected_T = []
            if st.session_state.option_j:
                selected_T.append("Machine thermo 1")
            if st.session_state.option_k:
                selected_T.append("Machine thermo 2")


        # ----- COLONNE DROITE : Cellule Préparation Têtes de Pinceaux -----
        with col2:
            #st.subheader('Cellule Préparation Têtes de Pinceaux:')
            st.subheader('Atelier Pinceaux:')

            # Initialisation des clés pour pinceaux
            for key in ["select_all_PNX", "option_e", "option_f", "option_g", "option_h", "option_i"]:
                if key not in st.session_state:
                    st.session_state[key] = False

            def select_all_pnx():
                st.session_state.option_e = True
                st.session_state.option_f = True
                st.session_state.option_g = True
                st.session_state.option_h = True
                st.session_state.option_i = True
            st.write('**Cellule Préparation Têtes de Pinceaux**')

            st.checkbox("Tout sélectionner", key="select_all_PNX", on_change=select_all_pnx)
            st.checkbox("Table Manuelle 1", key="option_e")
            st.checkbox("Table Manuelle 2", key="option_f")
            st.checkbox("Table Manuelle 3", key="option_g")
            st.checkbox("Machine Automatique Plat", key="option_h")
            st.checkbox("Machine Automatique Rond", key="option_i")

            selected_PNX = []
            if st.session_state.option_e:
                selected_PNX.append("TM1")
            if st.session_state.option_f:
                selected_PNX.append("TM2")
            if st.session_state.option_g:
                selected_PNX.append("TM3")
            if st.session_state.option_h:
                selected_PNX.append("MAP")
            if st.session_state.option_i:
                selected_PNX.append("MAR")
    



        
        #________________EXTRACTION FROM EXCEL________________________
        articles = wo['ARTICLE'].tolist()
        quantites = wo['QUANTITE'].tolist()
        #-----------------EXTRACTION THERMO----------------------------
        liste_thermo= pd.read_excel(DATAS, sheet_name='THERMOListe', index_col=0)
        P_thermo = pd.read_excel(DATAS, sheet_name='THERMOP', index_col=0, header=None)
        S_thermo= pd.read_excel(DATAS, sheet_name='THERMOS', index_col=0)
        corr=pd.read_excel(DATAS, sheet_name='THERMOcorr')
        coef=pd.read_excel(DATAS, sheet_name='THERMOcoef')

        #-----------------EXTRACTION DECOUPE----------------------------------
        liste_decoupe= pd.read_excel(DATAS, sheet_name='DECOUPEListe', index_col=0)
        P_decoupe = pd.read_excel(DATAS, sheet_name='DECOUPEP', index_col=0)
        M_decoupe = pd.read_excel(DATAS, sheet_name='DECOUPEM', index_col=0)
        EDJo = pd.read_excel(DATAS, sheet_name='DECOUPEEDJ', index_col=0)
        L3o = pd.read_excel(DATAS, sheet_name='DECOUPE3L', index_col=0)
        L4o = pd.read_excel(DATAS, sheet_name='DECOUPE4L', index_col=0)

        #-------------------------EXTRACTION PNX----------------------------
        liste_pnx = pd.read_excel(DATAS, sheet_name='PNXListe', index_col=0)
        TMo = pd.read_excel(DATAS, sheet_name='PNXTM', index_col=0)
        MAPo = pd.read_excel(DATAS, sheet_name='PNXPLAT', index_col=0)
        MARo = pd.read_excel(DATAS, sheet_name='PNXROND', index_col=0)
        P_pnx = pd.read_excel(DATAS, sheet_name='PNXP', index_col=0)
        M_pnx = pd.read_excel(DATAS, sheet_name='PNXM', index_col=0)

        #----------------------ARTICLES?????????????????
        def chercher_articles_et_quantites_par_feuille(fichier_excel, articles, quantites, feuilles=["DECOUPEListe", "THERMOcorr", "PNXListe"]):
            # Charger le fichier Excel
            xls = pd.ExcelFile(fichier_excel)
            
            # Construire un dictionnaire article -> quantité
            dict_articles_quantites = dict(zip(articles, quantites))
            
            # Fonction interne pour trouver les articles dans une feuille
            def trouver_articles(df, articles):
                trouvés = set()
                for col in df.columns:
                    for val in df[col].dropna().astype(str):
                        if val in articles:
                            trouvés.add(val)
                return list(trouvés)

            résultats_articles = {}
            résultats_quantites = {}

            for feuille in feuilles:
                df = xls.parse(feuille)
                art_trouves = trouver_articles(df, articles)
                résultats_articles[feuille] = art_trouves
                # Récupérer la quantité correspondant à chaque article trouvé
                quant_trouvees = [dict_articles_quantites[art] for art in art_trouves]
                résultats_quantites[feuille] = quant_trouvees
            
            return (résultats_articles["DECOUPEListe"], résultats_quantites["DECOUPEListe"],
                    résultats_articles["THERMOcorr"], résultats_quantites["THERMOcorr"],
                    résultats_articles["PNXListe"], résultats_quantites["PNXListe"])

        decoupe, decoupe_q, thermo, thermo_q, pnx, pnx_q = chercher_articles_et_quantites_par_feuille(DATAS, articles, quantites)
        # Création des listes d'index simples
        liste_decoupe = list(range(len(decoupe)))
        liste_thermo = list(range(len(thermo)))
        y=960
        thermo_q_modifiee = [y / 3 if q > y else q for q in thermo_q]
        liste_pnx = list(range(len(pnx)))

        #_________________CONSTRUCTION DES MATRICES________________________
        #-----------------        THERMO           ------------------------
        thermo_q_array = np.array(thermo_q)
        thermo_q_modifiee_array = np.array(thermo_q_modifiee)
        sf_list = []
        coefs = []

        for produit in thermo:
            ligne_corresp = corr[corr['article'] == produit]
            ligne_coef = coef[coef['article'] == produit]

            if not ligne_corresp.empty and not ligne_coef.empty:
                sf = ligne_corresp['sf'].iloc[0]
                coef_val = ligne_coef['coef'].iloc[0]  # Remplacez 'coef' par le vrai nom de colonne dans coef
                sf_list.append(sf)
                coefs.append(coef_val)
            else:
                st.write(f"Produit inconnu ou coefficient manquant : {produit}")
        COEFS = np.array(coefs)
        P_thermo1= P_thermo.loc[sf_list]
        S_thermo1 = S_thermo.loc[sf_list,sf_list]
        S_THERMO = np.array(S_thermo1.values)
        P_thermo_iter = np.array(P_thermo1.values)
        P_THERMO = P_thermo_iter * thermo_q_array[:, np.newaxis] / COEFS[:, np.newaxis]
        P_THERMO_SEUIL = P_thermo_iter * thermo_q_modifiee_array[:, np.newaxis] / COEFS[:, np.newaxis]

        #-----------------        DECOUPE          ------------------------
        # Extraire la sous-matrice
        decoupe_q_array = np.array(decoupe_q)
        M_decoupe1 = M_decoupe.loc[decoupe,selected_DECOUPE]
        P_decoupe1 = P_decoupe.loc[decoupe,selected_DECOUPE]
        # Dictionnaire de correspondance entre noms et matrices sources
        matrices_source_DECOUPE = {
            'EDJ': EDJo,
            '3L': L3o,
            '4L1': L4o,
            '4L2': L4o
        }

        # Création de la liste des sous-matrices
        matrices_list_DECOUPE = [matrices_source_DECOUPE[item].loc[decoupe, decoupe].values for item in selected_DECOUPE]

        # Conversion en matrice NumPy
        S_DECOUPE = np.array(matrices_list_DECOUPE)
        M_DECOUPE= np.array( M_decoupe1.values)
        P_decoupe_iter = np.array(P_decoupe1.values)
        # Reshape du vecteur en colonne pour un broadcasting ligne par ligne
        P_DECOUPE = P_decoupe_iter * decoupe_q_array[:, np.newaxis]

        #-----------------        PNX              ------------------------
        pnx_q_array = np.array(pnx_q)
        M_pnx1 = M_pnx.loc[pnx,selected_PNX]
        P_pnx1 = P_pnx.loc[pnx,selected_PNX]
        # Dictionnaire de correspondance entre noms et matrices sources
        matrices_source_PNX = {
            'TM1': TMo,
            'TM2': TMo,
            'TM3': TMo,
            'MAP': MAPo,
            'MAR': MARo      }
        matrices_list_PNX = [matrices_source_PNX[item].loc[pnx,pnx].values for item in selected_PNX]
        S_PNX = np.array(matrices_list_PNX)
        M_PNX = np.array(M_pnx1 .values)
        P_pnx_iter = np.array(P_pnx1.values)
        P_PNX = P_pnx_iter * pnx_q_array[:, np.newaxis]
        valid_selection = (
            len(selected_DECOUPE) > 0 and
            len(selected_T) > 0 and
            len(selected_PNX) > 0
        )

        # Affichage du bouton Start
        start_button = st.button("Start", disabled=not valid_selection)
        if start_button:
            st.success("Lancement de la moulinette...")
            #________________________APPEL DES FONCTIONS______________________
            #-----------------------------THERMO---------------------
            #HEURISTIQUE1
            nmt=len(selected_T)
            
            schedule1_T_O, cmax1_T_O = H1(P_THERMO, S_THERMO, nmt)
            schedule1_T, cmax1_T,*_=amelioration_locale_T(schedule1_T_O, P_THERMO, S_THERMO)

            #HEURISTIQUE2
            schedule2_T_O, cmax2_T_O = H2(P_THERMO, S_THERMO, nmt)
            schedule2_T, cmax2_T,*_=amelioration_locale_T(schedule2_T_O, P_THERMO, S_THERMO)

            solutions_heuristiques_T = {
                "Heuristique1": (schedule1_T, cmax1_T),
                "Heuristique2": (schedule2_T, cmax2_T)}
            best_label_T = min(solutions_heuristiques_T, key=lambda k: solutions_heuristiques_T[k][1])
            worst_label_T = max(solutions_heuristiques_T, key=lambda k: solutions_heuristiques_T[k][1])
            best_schedule_T, best_cmax_T = solutions_heuristiques_T[best_label_T]
            worst_schedule_T, worst_cmax_T = solutions_heuristiques_T[worst_label_T]
            
            # Trie les solutions par Cmax croissant
            sorted_solutions_T= sorted(solutions_heuristiques_T.items(), key=lambda x: x[1][1])

            # Récupère les 3 meilleures solutions
            top_3_solutions_T= sorted_solutions_T[:2]

            # Crée deux listes : une pour les schedules, une pour les Cmax
            top_3_schedules_PNX = [sol[1][0] for sol in top_3_solutions_T]
            top_3_cmax_PNX = [sol[1][1] for sol in top_3_solutions_T]
            completion_times = get_completion_times(best_schedule_T, P_THERMO_SEUIL, S_THERMO, nmt)

            #------------------------DECOUPE-------------------------
            #HEURISTIQUE1
            schedule_DF,cmax_T,*_=GH_dynamique_disponibilite(P_DECOUPE, S_DECOUPE, M_DECOUPE, completion_times)
            
            

            #-----------------------------------PNX-----------------------------------------
            M=M_PNX
            #HEURISTIQUE1
            schedule1_PNX, cmax1_PNX,*_ = GH1(P=P_PNX, S=S_PNX, M=M_PNX)   
            #HEURISTIQUE1 + AMELIORATION
            better_schedule1_PNX, better_cmax1_PNX, _ = amelioration_locale(schedule1_PNX, P=P_PNX, S=S_PNX)       
            #HEURISTIQUE2
            #schedule2_PNX, cmax2_PNX = GH2(P=P_PNX, S=S_PNX, M=M_PNX)
            #HEURISTIQUE2 + AMELIORATION
            #better_schedule2_PNX, better_cmax2_PNX, _ = amelioration_locale(schedule2_PNX, P=P_PNX, S=S_PNX)
            #Best Cmax pour solution innitiale
            best_cmax_PNX = min(cmax1_PNX,better_cmax1_PNX)
            schedule_map_PNX = {
                cmax1_PNX: schedule1_PNX,
                better_cmax1_PNX: better_schedule1_PNX
            }
            best_schedule_PNX = schedule_map_PNX[best_cmax_PNX]
            #AG
            AG_schedule_PNX, history_AG_PNX = genetic_algorithm(
            P=P_PNX, S=S_PNX, M=M_PNX,
            schedule=best_schedule_PNX,
            pop_size=POP_SIZE,
            generations=GENERATIONS,
            crossover_rate=CROSSOVER_RATE,
            mutation_rate=MUTATION_RATE,
            elite_size=ELITE_SIZE)
            #RS
            RS_schedule_PNX, best_cmax_RS_PNX, history_RS_PNX, best_history, temp_history = simulated_annealing(
            best_schedule_PNX, P=P_PNX, S=S_PNX, M=M_PNX,
            initial_temp=INITIAL_TEMP, 
            cooling_rate=COOLING_RATE,
            min_temp=MIN_TEMP,
            iterations_per_temp=ITERATIONS_PER_TEMP)
            #BEST ONE
            solutions_heuristiques_PNX = {
                "Heuristique1": (schedule1_PNX, cmax1_PNX),
                "Heuristique1 Améliorée": (better_schedule1_PNX, better_cmax1_PNX),
                "Algorithme Génétique":(RS_schedule_PNX,best_cmax_RS_PNX),
                "Recuit Simulé": (RS_schedule_PNX,best_cmax_RS_PNX)}
            best_label_PNX = min(solutions_heuristiques_PNX, key=lambda k: solutions_heuristiques_PNX[k][1])
            worst_label_PNX = max(solutions_heuristiques_PNX, key=lambda k: solutions_heuristiques_PNX[k][1])
            best_schedule_PNX, best_cmax_PNX = solutions_heuristiques_PNX[best_label_PNX]
            worst_schedule_PNX, worst_cmax_PNX = solutions_heuristiques_PNX[worst_label_PNX]
            # Trie les solutions par Cmax croissant
            sorted_solutions_PNX = sorted(solutions_heuristiques_PNX.items(), key=lambda x: x[1][1])

            # Récupère les 3 meilleures solutions
            top_3_solutions_PNX= sorted_solutions_PNX[:2]

            # Crée deux listes : une pour les schedules, une pour les Cmax
            top_3_schedules_PNX = [sol[1][0] for sol in top_3_solutions_PNX]
            top_3_cmax_PNX = [sol[1][1] for sol in top_3_solutions_PNX]
            # Création des lignes pour chaque ordonnancement
            def build_schedule_dict(best_schedule, articles, machines):
                data = []
                for machine_id, article_indices in best_schedule.items():
                    for ordre, idx in enumerate(article_indices, start=1):
                        data.append({
                            "ARTICLE": articles[idx],
                            "Machine": machines[machine_id],
                            "Ordre": ordre
                        })
                return pd.DataFrame(data)
            #MACHINES
            
            machine_NOM_DECOUPE= {
                    'EDJ': "Machine Découpe EDJ",
                    '3L': "Machine Découpe 3L",
                    '4L1': "Machine Découpe 4L1",
                    '4L2': "Machine Découpe 4L2"
                }
            machine_NOM_PNX = {
                    'TM1': "Table Manuelle 1",
                    'TM2': "Table Manuelle 2",
                    'TM3': "Table Manuelle 3",
                    'MAP': "Machine Plat",
                    'MAR': "Machine Rond"
                }
            
            selected_names_DECOUPE = [machine_NOM_DECOUPE[machine] for machine in selected_DECOUPE]
            selected_names_PNX = [machine_NOM_PNX[machine] for machine in selected_PNX]
        
            df_thermo = build_schedule_dict(best_schedule_T, thermo, selected_T)
            df_pnx = build_schedule_dict(best_schedule_PNX, pnx, selected_names_PNX)
            
            df_decoupe = build_schedule_dict(schedule_DF, decoupe, selected_names_DECOUPE)

            # Fusion des deux ordonnancements
            df_merge = pd.concat([df_pnx, df_decoupe,df_thermo])

            # Fusion avec le fichier Excel de base
            df_final = wo.merge(df_merge, on="ARTICLE", how="left")
            # 1. Afficher le DataFrame
            st.subheader("📊 ORDONNANCEMENT")
            st.dataframe(df_final)

            # 2. Convertir en fichier Excel en mémoire
            def convert_df_to_excel(df):
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='Ordonnance')
                output.seek(0)
                return output

            # 3. Bouton de téléchargement
            excel_file = convert_df_to_excel(df_final)

            st.download_button(
                label="📥 Télécharger en Excel",
                data=excel_file,
                file_name="WO_RECOLTE_ORDONNANCE.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            #st.header('Cellule Thermo ')
            #MT=['M1','M2']
            #plot_gantt(S_THERMO, P_THERMO, best_schedule_T, thermo, MT)
            st.header('Cellule Thermofusion ')
            n_T=len(selected_T)
            with st.expander("Afficher les differents ordonnancements possibles"):
                               # Pré-calcul des temps machines pour toutes les solutions
                all_temps_machines_T = []
                for _, (schedule, _) in top_3_solutions_T:
                    all_temps_machines_T.append(calculer_temps_total_T(schedule, P_THERMO, S_THERMO))

                # Affichage
                for idx, ((label, (schedule, cmax)), temps_machines) in enumerate(zip(top_3_solutions_T, all_temps_machines_T), start=1):
                    st.write(f"## 💡 Solution {idx}:")
                    for machine in range(n_T):
                        tasks = schedule[machine]
                        noms_articles = [thermo[i] for i in tasks]
                        machine_time = temps_machines[machine]
                        st.write(f"**{selected_T[machine]}**: {noms_articles} (**{machine_time:.2f}** heures)")
                    st.success(f"**Cmax**: {cmax / 60:.2f} heures\n")
            plot_gantt_T(S_THERMO, P_THERMO, schedule1_T,thermo, selected_T)
            st.header('Cellule Découpe Manchons ')
            n_DECOUPE=len(selected_DECOUPE)
            with st.expander("Afficher les differents ordonnancements possibles"):
                    # Calcul des temps machines pour la solution unique
                    temps_machines = calculer_temps_total(schedule_DF, P_DECOUPE, S_DECOUPE)
                    temps_max = max(temps_machines.values())
        
        

                    # Affichage de l’ordonnancement
                    st.write(f"# 💡 Solution :")
                    for machine in range(n_DECOUPE):
                        tasks = schedule_DF[machine]
                        noms_articles = [decoupe[i] for i in tasks]
                        machine_time = temps_machines[machine]
                        st.write(f"**{selected_DECOUPE[machine]}** : {noms_articles} (**{machine_time:.2f}** heures)")
                    
                    # Cmax si tu l’as calculé quelque part
                    st.success(f"**Cmax** : {temps_max:.2f} heures")
            

                


            plot_gantt(S_DECOUPE, P_DECOUPE, schedule_DF, decoupe, selected_DECOUPE)
            st.header('Cellule Préparation Têtes de Pinceaux ')
            n_PNX=len(selected_PNX)
            with st.expander("Afficher les differents ordonnancement possible pour la cellule PNX"):
                # Pré-calcul des temps machines pour toutes les solutions
                all_temps_machines_PNX = []
                for _, (schedule, _) in top_3_solutions_PNX:
                    all_temps_machines_PNX.append(calculer_temps_total(schedule, P_PNX, S_PNX))

                # Affichage
                for idx, ((label, (schedule, cmax)), temps_machines) in enumerate(zip(top_3_solutions_PNX, all_temps_machines_PNX), start=1):
                    st.write(f"#💡 Solution{idx}:")
                    for machine in range(n_PNX):
                        tasks = schedule[machine]
                        noms_articles = [pnx[i] for i in tasks]
                        machine_time = temps_machines[machine]
                        st.write(f"**{selected_PNX[machine]}**: {noms_articles} (**{machine_time:.2f}** heures)")
                    st.success(f"**Cmax**: {cmax / 60:.2f} heures\n")
            plot_gantt(S_PNX, P_PNX, best_schedule_PNX, pnx, selected_PNX)
            

    else:
        st.write("Aucun fichier Excel n'a été téléchargé.")