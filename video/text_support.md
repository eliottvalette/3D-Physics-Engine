# Développement d'un moteur physique 3D en Python : de la théorie à l'implémentation

> Alors que l'intelligence artificielle révolutionne le monde numérique, la robotique incarne cette révolution dans le monde physique. Cependant, la robotique s'écarte de la promesse démocratique de l'IA : une explosion de nos capacités accessible depuis chez soi, au simple coût de votre imagination et d'un peu d'électricité.
> 
> En effet, les simulations 3D sont extrêmement coûteuses en termes de calcul, un coût nécessaire pour reproduire fidèlement la réalité physique. Le recours à la location de serveurs et de GPU devient alors indispensable, et la promesse d'accès libre s'efface.
> 
> Nous verrons ici comment concevoir et implémenter un environnement 3D avec une physique réaliste, dans lequel un robot articulé peut se mouvoir de manière autonome.
> 
> *Licence : MIT, libre de réutilisation.*

---

## 1 · Fondamentaux : modélisation du sol

Pour simuler un sol, seuls deux éléments sont nécessaires :
- Une frontière délimitant la matière et l'extérieur
- Une normale définissant l'orientation de la surface

Pour un sol plat horizontal, nous utilisons les équations suivantes :

$$
\boxed{\;y\;\geqslant\;0\;} \quad \boxed{\;\mathbf{n}\;=\;(0,\;1,\;0)^\top\;} 
$$

*Ici **$y$** représente la coordonnée verticale d'un point dans l'espace, et **$\mathbf{n}$** le vecteur normal au plan*

Tout point dont la composante **$y$** devient négative est **en pénétration**. On note $\delta$ cette valeur de pénétration :

Pour un point $\mathbf{p} = (p_x, p_y, p_z)$ :

$$
\delta = \max\bigl(0,\,-p_y\bigr)
$$

### Implémentation en code :

```python
# ground.py
class Ground:
    def __init__(self, height_y=0):
        self.height_y = height_y
        self.normal = np.array([0, 1, 0])
```

---

## 2 · Premier contact : simulation d'un cube en chute libre

### 2.1 · Intégration d'Euler semi-explicite

Considérons un cube unitaire (chaque côté mesure une unité). On lui attribue une position $(x,y,z)$ située au centre de ce cube. En appliquant toutes les combinaisons $\pm0.5$ dans les directions $x$, $y$ et $z$, on obtient la position de chacun des huit sommets du cube.

Pour la physique, à chaque pas de temps $\Delta t$ dans la simulation :

$$
\begin{aligned}
\mathbf{g} &= (0, -9.81, 0) \quad \text{(accélération gravitationnelle)}\\[2pt]
\mathbf{v}_{new} &= \mathbf{v}_{old} + \mathbf{g}\,\Delta t,\\[2pt]
\mathbf{x}_{new} &= \mathbf{x}_{old} + \mathbf{v}_{new}\,\Delta t.
\end{aligned}
$$

**Implémentation en code :**

```python
class Cube:
    def __init__(self, initial_position=[0, 4, 0], initial_velocity=[0, 0, 0]):
        self.position = np.array(initial_position)
        self.velocity = np.array(initial_velocity)
        self.side_length = 1.0
```

```python
# À chaque pas de simulation
def update(self, dt):
    self.velocity += GRAVITY * dt
    self.position += self.velocity * dt
```

### 2.2 · Détection et correction des collisions

Sans collision, notre cube tomberait indéfiniment. Ajoutons une détection et correction de collision avec le sol :

$$
\text{cube}_{\text{min}_y} = \text{cube}_y - 0.5 \\[10pt]
\boxed{
\delta = \max(0, -\text{cube}_{\text{min}_y}) \quad \Rightarrow \quad
\mathbf{p} \gets \mathbf{p} + \delta\,\mathbf{n}
}
$$

Afin de créer un rebond léger, nous appliquons une perte d'énergie lors du rebond, nous ajoutons donc dans le pas physique :
$$
\begin{aligned}
\mathbf{\delta} = \max(0, -\text{cube}_{\text{min}_y})\\[2pt]
\mathbf{x}_{new_y} &= \mathbf{x}_{old_y} + \mathbf{\delta} \\[2pt]
\mathbf{v}_{new_y} &= -0.7 \cdot \mathbf{v}_{old_y}
\end{aligned}
$$

```python
def handle_ground_collision(self):
    # Déterminer le point le plus bas du cube
    cube_y = self.position[1] - 0.5 # Le cube sans rotation, la face inférieure est à y - 0.5 (en partant du centre du cube)
    penetration = max(0, -cube_y)
    
    if penetration > 0:
        # Corriger la position
        self.position[1] += penetration
        self.velocity[1] *= -0.7  # rebond avec perte d'énergie

```

---

## 3 · Cube rotatif avec physique avancée

### 3.1 · Intégration de la rotation

Pour un cube pouvant tourner autour de ses axes $x$, $y$ et $z$, nous étendons le système :

$$
\begin{aligned}
\mathbf{v}_{new} &= \mathbf{v}_{old} + \mathbf{g}\,\Delta t,\\[2pt]
\mathbf{x}_{new} &= \mathbf{x}_{old} + \mathbf{v}_{new}\,\Delta t,\\[2pt]
\boldsymbol{\theta}_{new} &= \boldsymbol{\theta}_{old} + \boldsymbol{\omega}_{new}\,\Delta t.
\end{aligned}
$$

### 3.2 · Collision par sommets
Désormais, quand le cube heurte le sol, il est possible qu'il le frappe autrement qu'avec sa face inférieure parallèle au sol : en effet il peut le frapper avec une arête ou avec un seul sommet.

Ainsi, la correction physique doit maintenant s'appliquer sur chaque sommet individuellement :

Pour chaque sommet $\mathbf{v}_i$ du cube :

$$
\delta_i = \max(0, -v_{i,y}) \quad \Rightarrow \quad
\delta = \max_i(\delta_i)
$$

### 3.3 · Impulsion normale asymétrique

Quand un sommet heurte le sol **en descendant**, nous calculons l'impulsion nécessaire pour annuler la vitesse relative :

// expliquer les maths et la physique qui sera illustrée par le code qui suit


*Cela s'interprète dans le code de la façon suivante*

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

### 4.1 · Modélisation des articulations

Pour simuler un objet articulé (comme un avant-bras et un biceps) nous allons avoir besoin d'un nouvel objet : une articulation. 

Celle-ci reliera 2 objets, ici up-arm, low-arm (tous deux des objets de la classe Cube) et imposera un angle donné entre ces deux derniers. On assignera sur chacun de ces objets un point d'ancrage (situé sur la surface de l'objet). À chaque instant de la simulation, les deux objets seront en contact sur ce point, on définira à cet endroit la position de la jointure. 

D'autre part parmi ces deux objets, un aura le rôle de guide, et l'autre de guidé. Lorsque l'angle de l'articulation sera modifié durant la simulation, le guide restera fixe, et le guidé subira la rotation adaptée pour respecter l'angle imposé par la jointure.


### Implémentation en code :
```python
class Joint:
    def __init__(self, object_1, object_2, face_1, face_2, angle):
        self.object_1 = object_1
        self.object_2 = object_2
        self.face_1 = face_1
        self.face_2 = face_2
        self.angle = angle  # Angle du joint en radians
```

### 4.2 · Choix de l'architecture

Pour simuler les collisions sur notre bras articulé, deux approches s'offrent à nous :

1. **Approche multi-corps** : Déclarer les 2 cubes + 1 jointure et traiter la physique pour chaque objet séparément
2. **Approche unifiée** : Traiter l'ensemble articulé comme un seul objet

La première approche, bien que plus intuitive, pose des problèmes de stabilité numérique. En effet, la simulation physique se faisant de manière séquentielle, l'ordre des mises à jour :

$$
\text{joint} \rightarrow \text{cube}_1 \rightarrow \text{cube}_2 \\
\text{cube}_1 \rightarrow \text{joint} \rightarrow \text{cube}_2
$$

Crée une asymétrie dans l'effet de la physique entre les deux corps et perturbe gravement la fiabilité de la simulation.

Ainsi nous suivrons le protocole : à chaque pas, avant toute modification physique, si un angle différent est demandé par l'articulation : 
- On part de la position actuelle du guide
- On regarde l'angle demandé par l'articulation
- On détermine la position de l'objet guidé, relativement au guide, dans l'angle imposé par l'articulation
- On récupère enfin l'ensemble des sommets de nos objets, et on crée un objet unifié, et c'est sur cet objet que nous effectuerons l'actualisation physique.

### 4.3 · Exemple de code simple pour le bras : 

Voici un exemple simplifié d'un bras articulé avec deux segments (bras supérieur et avant-bras) :

```python
```

```python
class Arm:
    def __init__(self, upper_arm, lower_arm, joint):
        self.upper_arm = upper_arm
        self.lower_arm = lower_arm
        self.joint = joint
        self.vertices = []
    
    def update_vertices(self):
        # Calculer la position de l'avant-bras basée sur l'angle du joint
        joint_angle = self.joint.angle
        shoulder_pos = self.upper_arm.position
        elbow_pos = self.calculate_elbow_position(shoulder_pos, joint_angle)
        
        # Assembler tous les vertices 
        updated_upper_arm_vertices = self.upper_arm.get_vertices()
        updated_lower_arm_vertices = self.lower_arm.get_vertices()
        self.vertices = (updated_upper_arm_vertices + 
                        updated_lower_arm_vertices)                        

# Utilisation
upper_arm = Cube(position=[0, 4, 0])
lower_arm = Cube(position=[0, 2, 0])
joint = Joint(upper_arm, lower_arm, angle=0)
arm = Arm(upper_arm, lower_arm, joint)
```

Cet exemple illustre l'approche unifiée : plutôt que de traiter chaque segment séparément, nous calculons tous les vertices transformés en une seule fois, puis nous traitons l'ensemble comme un objet unique pour la physique.

### 4.4 · Quadrupède : 9 éléments articulés
Notre bras est fonctionnel mais ses cas d'usage sont très restreints, on peut passer à un objet plus complexe : un quadrupède

Pour le quadrupède, nous déclarons 9 éléments :
- 1 corps principal (body)
- 4 pattes supérieures (upper legs)
- 4 pattes inférieures (lower legs)

Et pour relier tout cela
- 8 articulations : 4 épaules + 4 coudes

Pour les relations guide / guidé, on choisira logiquement

Body guide 1 à 1 des 4 upper legs, et chaque upper leg sera la guide d'une lower leg

### 4.5 · Protocole de mise à jour

L'update suit le protocole suivant :
le symbole → signifie : *Permet de déterminer*

1. **Position du corps** → positions des épaules
2. **Angles des épaules** → positions des pattes supérieures
3. **Positions des coudes** (extrémités des pattes supérieures)
4. **Angles des coudes** → positions des pattes inférieures
5. **Calcul de tous les sommets** du quadrupède
6. **Création / actualisation de l'objet Quadruped**

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