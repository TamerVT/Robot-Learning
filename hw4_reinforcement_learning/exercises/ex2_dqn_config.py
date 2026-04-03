"""
Hyperparameters for Exercise 2 (DQN).

You are encouraged to tune:
- lr
- epsilon
- target_update
- hidden_dim

Please keep the remaining parameters unchanged unless explicitly stated.
"""

DQN_PARAMETERS = {
    # Tuned hyperparameters
    "lr": 5e-4,            
    "epsilon": 0.1,        
    "target_update": 50,   
    "hidden_dim": 256,    

    # Fixed parameters
    "gamma": 0.99,
    "num_episodes": 500,
    "buffer_size": 10000,
    "minimal_size": 500,
    "batch_size": 64,
    "seed": 0,
}
