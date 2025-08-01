# Construire pas à pas un mini‑moteur physique 3D en Python

> Alors que l'IA révolutionne le monde numérique, les robots incarnent cette révolution dans le monde physique. Cependant, la robotique rompt avec la promesse démocratique de l'IA : une explosion de nos capacités accessible depuis chez soi, au simple coût de votre imagination et d'un peu d'électricité.
> 
> En effet, les simulations 3D sont extrêmement coûteuses en termes de calcul, un coût nécessaire pour reproduire fidèlement la réalité physique. Le recours à la location de serveurs et de GPU devient alors indispensable, et la promesse d'accès libre s'efface.
> 
> Nous verrons ici comment simuler un environnement 3D avec une physique réaliste, dans lequel un robot articulé peut se mouvoir de manière autonome.
> 
> *Licence : MIT, libre de réutilisation.*

---

## 1 · Fondamentaux : le sol

Pour simuler un sol, il suffit de deux éléments :
- Une frontière séparant la matière de l'extérieur
- Une normale définissant l'orientation de la surface

Pour un sol plat horizontal, cela se traduit par :

$$
\boxed{\,y \geqslant 0\,} \qquad \boxed{\,\mathbf{n} = (0,\,1,\,0)\,}
$$

*Ici, **$y$** est la coordonnée verticale d'un point dans l'espace, et **$\mathbf{n}$** est le vecteur normal au plan.*

Tout point dont la composante **$y$** devient négative est **en pénétration**. On note $\delta$ cette valeur :

$$
\delta = \max(0,\,-p_y)
$$

### Implémentation en code :

---

## 3 · Cube rotatif avec physique avancée

### 3.1 · Intégration de la rotation

Pour un cube pouvant tourner autour de ses axes $x$, $y$ et $z$, nous étendons le système :

$$
\begin{aligned}
\mathbf{v}_{t+\Delta t} &= \mathbf{v}_t + \mathbf{g}\,\Delta t,\\[2pt]
\mathbf{x}_{t+\Delta t} &= \mathbf{x}_t + \mathbf{v}_{t+\Delta t}\,\Delta t,\\[2pt]
\boldsymbol{\theta}_{t+\Delta t} &= \boldsymbol{\theta}_t + \boldsymbol{\omega}_t\,\Delta t.
\end{aligned}
$$

### 3.2 · Collision par sommet

La correction physique s'applique maintenant sur chaque sommet individuellement :

Pour chaque sommet $\mathbf{v}_i$ du cube :

$$
\delta_i = \max(0, -v_{i,y}) \quad \Rightarrow \quad
\delta = \max_i(\delta_i)
$$

### 3.3 · Impulsion normale (collision verticale)

Quand un sommet heurte le sol **en descendant**, nous calculons l'impulsion nécessaire pour annuler la vitesse relative :

$$
v_{\text{rel}} = \mathbf{v}_{\text{sommet}} \cdot \mathbf{n} \\[10pt]
J = -\frac{v_{\text{rel}}}{\frac{1}{m} + \frac{(\mathbf{r} \times \mathbf{n}) \cdot (\mathbf{r} \times \mathbf{n})}{I_{yy}}}
$$

Où :
- $\mathbf{r} = \mathbf{v}_i - \mathbf{x}_{\text{cm}}$ (position relative du sommet)
- $I_{yy}$ : moment d'inertie autour de l'axe $Y$

### 3.4 · Couple et rotation

Lorsqu'un **coin** du cube touche en premier, l'impulsion génère un **couple** qui modifie la rotation :

$$
\Delta\boldsymbol{\omega} = \mathbf{I}^{-1}(\mathbf{r} \times J\,\mathbf{n}).
$$

```python
def handle_vertex_collision(self, vertex, vertex_velocity):
    if vertex[1] < 0 and vertex_velocity[1] < 0:
        # Position relative du sommet par rapport au centre de masse
        relative_position = vertex - self.position
        
        # Calcul de l'impulsion scalaire
        normal = np.array([0, 1, 0])
        relative_velocity = np.dot(vertex_velocity, normal)
        
        # Dénominateur de l'impulsion
        r_cross_n = np.cross(relative_position, normal)
        denom = (1/self.mass) + np.dot(normal, 
                np.cross(r_cross_n / self.inertia, relative_position))
        
        if denom != 0:
            scalar_impulse = -relative_velocity / denom
            
            # Appliquer l'impulsion au centre de masse
            self.velocity += (scalar_impulse * normal) / self.mass
            
            # Appliquer l'impulsion angulaire
            self.angular_velocity += np.cross(relative_position, 
                                            scalar_impulse * normal) / self.inertia
```

---

## 4 · Objets articulés : approche unifiée

### 4.1 · Choix de l'architecture

Pour simuler un objet articulé (comme un avant-bras et un biceps), deux approches s'offrent à nous :

1. **Approche multi-corps** : Déclarer 2 cubes + 1 jointure, traiter la physique pour chaque objet séparément
2. **Approche unifiée** : Traiter l'ensemble articulé comme un seul objet

La première approche, bien qu'intuitive, pose des problèmes de stabilité numérique. En effet, la simulation physique se faisant de manière séquentielle, l'ordre des mises à jour :

$$
\text{joint} \rightarrow \text{cube}_1 \rightarrow \text{cube}_2 \\
\text{cube}_1 \rightarrow \text{joint} \rightarrow \text{cube}_2
$$

Crée une asymétrie dans l'effet de la physique entre les deux corps et perturbe gravement la fiabilité de la simulation.

### 4.2 · Approche unifiée recommandée

Nous adoptons l'approche unifiée : l'objet articulé est traité comme un ensemble cohérent, où les articulations sont des contraintes géométriques plutôt que des objets physiques séparés.

### 4.3 · Quadrupède : 9 éléments articulés

Pour le quadrupède, nous déclarons 9 éléments :
- 1 corps principal (body)
- 4 pattes supérieures (upper legs)
- 4 pattes inférieures (lower legs)
- 8 articulations : 4 épaules + 4 coudes

### 4.4 · Protocole de mise à jour

L'update suit le protocole suivant :

1. **Position du corps** → positions des épaules
2. **Angles des épaules** → positions des pattes supérieures
3. **Positions des coudes** (extrémités des pattes supérieures)
4. **Angles des coudes** → positions des pattes inférieures
5. **Calcul de tous les sommets** du quadrupède
6. **Traitement unifié** : itération sur tous les sommets pour les corrections de collision

```python
def update_quadruped_vertices(self):
    # 1. Calculer les positions des épaules depuis le corps
    shoulder_positions = self.calculate_shoulder_positions()
    
    # 2. Appliquer les rotations d'épaules
    upper_leg_positions = self.apply_shoulder_rotations(shoulder_positions)
    
    # 3. Calculer les positions des coudes
    elbow_positions = self.calculate_elbow_positions(upper_leg_positions)
    
    # 4. Appliquer les rotations de coudes
    lower_leg_positions = self.apply_elbow_rotations(elbow_positions)
    
    # 5. Assembler tous les sommets
    all_vertices = (self.body_vertices + 
                   self.upper_leg_vertices + 
                   self.lower_leg_vertices)
    
    return all_vertices
```

Une fois tous les sommets calculés, nous traitons le quadrupède comme un seul objet, en itérant sur tous les sommets pour appliquer les corrections de collision avec le sol.

---

## 5 · Locomotion : traction horizontale

### 5.1 · Problème de la correction purement verticale

Étant donné que 100% des corrections de position se font par correction de pénétration face au sol le long de la normale, le quadrupède n'est pas capable de s'appuyer sur le sol pour se projeter vers l'avant. Il ne fait que monter et descendre.

### 5.2 · Solution : traction horizontale

Nous ajoutons un système de traction horizontale après la correction verticale. Ce système détecte les points en contact avec le sol et applique des forces de friction pour permettre le mouvement horizontal.

### 5.3 · Équations de traction

Pour chaque sommet en contact avec le sol :

$$
\begin{aligned}
\mathbf{r}_{\text{prev}} &= \mathbf{v}_i(t-1) - \mathbf{x}_{\text{cm}}(t-1) \\[2pt]
\mathbf{r}_{\text{curr}} &= \mathbf{v}_i(t) - \mathbf{x}_{\text{cm}}(t) \\[2pt]
\Delta\mathbf{r} &= \mathbf{r}_{\text{curr}} - \mathbf{r}_{\text{prev}} \\[2pt]
\Delta\mathbf{r}_{\text{horiz}} &= \Delta\mathbf{r} - (\Delta\mathbf{r} \cdot \mathbf{n})\mathbf{n}
\end{aligned}
$$

L'impulsion de traction nécessaire :

$$
\mathbf{J}_{\text{traction}} = -\frac{m \Delta\mathbf{r}_{\text{horiz}}}{\Delta t}
$$

Avec limitation par friction statique :

$$
|\mathbf{J}_{\text{traction}}| \leq \mu_s m g \Delta t
$$

### 5.4 · Implémentation

```python
def apply_horizontal_traction(self, prev_vertices, current_vertices):
    # Identifier les points au sol
    contact_threshold = 0.01
    prev_on_ground = prev_vertices[:, 1] <= contact_threshold
    curr_on_ground = current_vertices[:, 1] <= contact_threshold
    both_on_ground = prev_on_ground & curr_on_ground
    
    if np.any(both_on_ground):
        # Extraire les vertices concernés
        ground_prev = prev_vertices[both_on_ground].copy()
        ground_curr = current_vertices[both_on_ground].copy()
        
        # Normaliser la hauteur (points restent au sol)
        ground_prev[:, 1] = 0
        ground_curr[:, 1] = 0
        
        # Calculer les déplacements horizontaux
        deltas = ground_curr - ground_prev
        deltas[:, 1] = 0.0  # Composante horizontale uniquement
        
        # Filtrer par seuil de mouvement significatif
        delta_norms = np.linalg.norm(deltas, axis=1)
        significant_movement = delta_norms >= SLIP_THRESHOLD * DT
        
        if np.any(significant_movement):
            significant_deltas = deltas[significant_movement]
            significant_curr = ground_curr[significant_movement]
            
            # Calculer les impulsions nécessaires
            J_needed = -self.mass * significant_deltas / DT
            J_cap = STATIC_FRICTION_CAP * DT
            J_clipped = np.clip(J_needed, -J_cap, J_cap)
            
            # Appliquer les impulsions linéaires
            self.velocity += np.mean(J_clipped, axis=0) / self.mass
            
            # Appliquer les impulsions angulaires
            r_vectors = significant_curr - self.position
            angular_impulses = np.cross(r_vectors, J_clipped) / self.inertia
            self.angular_velocity += np.mean(angular_impulses, axis=0)
```

---

## Résultat final

Avec ces améliorations, vous obtenez un quadrupède capable de :
- Maintenir son équilibre grâce aux corrections de collision
- Se déplacer horizontalement grâce au système de traction
- Réagir de manière réaliste aux forces physiques
- S'adapter à différents terrains

Le système est maintenant prêt pour l'apprentissage par renforcement, où un agent pourra apprendre à contrôler les articulations pour accomplir des tâches de locomotion complexes.