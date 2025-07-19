import numpy as np

def create_quadruped_vertices():
    """
    Creates the vertex arrays for a quadruped with:
    - 1 body (8 vertices)
    - 4 upper legs (8 vertices each)
    - 4 lower legs (8 vertices each)
    Total: 40 vertices
    """
    
    # Body vertices (centered at origin, then will be positioned)
    body_vertices = [
        np.array([-2.,  5., -3.]), np.array([-2.,  5.,  3.]), 
        np.array([-2.,  6., -3.]), np.array([-2.,  6.,  3.]), 
        np.array([ 2.,  5., -3.]), np.array([ 2.,  5.,  3.]), 
        np.array([ 2.,  6., -3.]), np.array([ 2.,  6.,  3.])
    ]
    
    # Upper leg vertices (4 legs)
    upper_leg_0 = [  # Front right leg
        np.array([1. , 4., 2.]), np.array([1. , 4., 3.]), 
        np.array([1. , 5., 2.]), np.array([1. , 5., 3.]), 
        np.array([2. , 4., 2.]), np.array([2. , 4., 3.]), 
        np.array([2. , 5., 2.]), np.array([2. , 5., 3.])
    ]
    
    upper_leg_1 = [  # Front left leg
        np.array([ 1. ,  4., -3.]), np.array([ 1. ,  4., -2.]), 
        np.array([ 1. ,  5., -3.]), np.array([ 1. ,  5., -2.]), 
        np.array([ 2. ,  4., -3.]), np.array([ 2. ,  4., -2.]), 
        np.array([ 2. ,  5., -3.]), np.array([ 2. ,  5., -2.])
    ]
    
    upper_leg_2 = [  # Back right leg
        np.array([-2. ,  4.,  2.]), np.array([-2. ,  4.,  3.]), 
        np.array([-2. ,  5.,  2.]), np.array([-2. ,  5.,  3.]), 
        np.array([-1. ,  4.,  2.]), np.array([-1. ,  4.,  3.]), 
        np.array([-1. ,  5.,  2.]), np.array([-1. ,  5.,  3.])
    ]
    
    upper_leg_3 = [  # Back left leg
        np.array([-2. ,  4., -3.]), np.array([-2. ,  4., -2.]), 
        np.array([-2. ,  5., -3.]), np.array([-2. ,  5., -2.]), 
        np.array([-1. ,  4., -3.]), np.array([-1. ,  4., -2.]), 
        np.array([-1. ,  5., -3.]), np.array([-1. ,  5., -2.])
    ]
    
    # Lower leg vertices (4 legs)
    lower_leg_0 = [  # Front right leg
        np.array([1., 2. , 2. ]), np.array([1., 2. , 3. ]), 
        np.array([1., 4. , 2. ]), np.array([1., 4. , 3. ]), 
        np.array([2., 2. , 2. ]), np.array([2., 2. , 3. ]), 
        np.array([2., 4. , 2. ]), np.array([2., 4. , 3. ])
    ]
    
    lower_leg_1 = [  # Front left leg
        np.array([ 1.,  2. , -3. ]), np.array([ 1.,  2. , -2. ]), 
        np.array([ 1.,  4. , -3. ]), np.array([ 1.,  4. , -2. ]), 
        np.array([ 2.,  2. , -3. ]), np.array([ 2.,  2. , -2. ]), 
        np.array([ 2.,  4. , -3. ]), np.array([ 2.,  4. , -2. ])
    ]
    
    lower_leg_2 = [  # Back right leg
        np.array([-2.,  2. ,  2. ]), np.array([-2.,  2. ,  3. ]), 
        np.array([-2.,  4. ,  2. ]), np.array([-2.,  4. ,  3. ]), 
        np.array([-1.,  2. ,  2. ]), np.array([-1.,  2. ,  3. ]), 
        np.array([-1.,  4. ,  2. ]), np.array([-1.,  4. ,  3. ])
    ]
    
    lower_leg_3 = [  # Back left leg
        np.array([-2.,  2. , -3. ]), np.array([-2.,  2. , -2. ]), 
        np.array([-2.,  4. , -3. ]), np.array([-2.,  4. , -2. ]), 
        np.array([-1.,  2. , -3. ]), np.array([-1.,  2. , -2. ]), 
        np.array([-1.,  4. , -3. ]), np.array([-1.,  4. , -2. ])
    ]
    
    # Group all parts
    upper_legs = [upper_leg_0, upper_leg_1, upper_leg_2, upper_leg_3]
    lower_legs = [lower_leg_0, lower_leg_1, lower_leg_2, lower_leg_3]
    
    return {
        'body': body_vertices,
        'upper_legs': upper_legs,
        'lower_legs': lower_legs,
        'all_parts': [body_vertices] + upper_legs + lower_legs
    }

def get_quadruped_vertices():
    """Returns all quadruped vertices as a flat list"""
    vertices_dict = create_quadruped_vertices()
    all_vertices = []
    
    for part in vertices_dict['all_parts']:
        all_vertices.extend(part)
    
    return all_vertices
