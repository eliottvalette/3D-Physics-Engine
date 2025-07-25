# Run all the physics environments
```bash
clear
python -m physics_env.first_scene
python -m physics_env.floor_and_wall
python -m physics_env.floor_and_ramp
python -m physics_env.staircase
python -m physics_env.tilted_cube
```

# Run the quadruped environment
```bash
clear
python -m physics_env.quadruped_env
```

# Run the training
```bash
clear
python main.py
```

# Generate the code to send to the chat
```bash
clear
python files_to_send.py
```

# Visualize the training profile
```bash
snakeviz profiling/training_profile.prof
```