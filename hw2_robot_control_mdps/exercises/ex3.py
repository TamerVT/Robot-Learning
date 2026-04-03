import numpy as np

import __init__
from scripts.utils import quat_mul, quat_conjugate, quat_normalize, rot_mat_to_quat

"""
# Important note:
# In physical simulations in Python, it is necessary to correctly modify the values in arrays which are attributes
# of the data object. Be careful to modify the arrays in-place (e.g., using slicing array[:] = new_array) rather than overwriting the
# entire array reference, otherwise the physics engine will not see your changes!
"""

def reset_robot(default_qpos: np.ndarray) -> np.ndarray:
    """
    TODO: Implement robot reset to its default joint positions with some small uniform noise (-0.5, 0.5).
    You can add random noise to the default joint positions using np.random.uniform.
    
    Inputs:
    - default_qpos: np.ndarray. The default joint positions. Dimensionality: 1D array, Shape: (num_joints,).

    Returns:
    - reset_qpos: np.ndarray. The joint positions to reset the robot to. Dimensionality: 1D array, Shape: (num_joints,).
    """
    # Create uniform noise between -0.5 and 0.5 of the same shape as default_qpos
    noise = np.random.uniform(low=-0.5, high=0.5, size=default_qpos.shape)
    
    # Add noise to the default positions
    reset_qpos = default_qpos + noise
    
    return reset_qpos
    


def reset_target_position(base_pos: np.ndarray) -> np.ndarray:
    """
    TODO: Sample and compute a new random target position relative to the base from uniform distribution.
    The ranges for the uniform distribution are given by the following arrays:
    - x: [0.2, 0.4]
    - y: [-0.2, 0.2]
    - z: [0.1, 0.4]

    Inputs:
    - base_pos: np.ndarray. The 3D position of the robot's base. Dimensionality: 1D array, Shape: (3,).
    
    Returns:
    - target_pos: np.ndarray. The 3D position of the target relative to the base. Dimensionality: 1D array, Shape: (3,).
    """
    # Sample offsets based on the provided ranges
    x_offset = np.random.uniform(0.2, 0.4)
    y_offset = np.random.uniform(-0.2, 0.2)
    z_offset = np.random.uniform(0.1, 0.4)
    
    offset = np.array([x_offset, y_offset, z_offset])
    
    # Compute the target position in the world frame by adding the offset to the base position
    target_pos = base_pos + offset
    
    return target_pos


def process_action(action: np.ndarray, jnt_range: np.ndarray) -> np.ndarray:
    """
    TODO: Convert normalized actions [-1, 1] to target joint positions.
    
    You should map the normalized action [-1, 1] to the actual joint range defined by jnt_range. The mapping should be linear,
    where -1 corresponds to the lower limit of the joint and 1 corresponds to the upper limit of the joint, 
    and 0 corresponds to the midpoint of the joint range.

    Inputs:
    - action: np.ndarray. Normalized actions from the policy. Dimensionality: 1D array, Shape: (num_joints,).
    - jnt_range: np.ndarray. Lower and upper limits for joints. Dimensionality: 2D array, Shape: (num_joints, 2).

    Returns:
    - target_qpos: np.ndarray. Target joint positions to apply as control. Dimensionality: 1D array, Shape: (num_joints,).
    """
    # Extract lower and upper limits
    lower_limits = jnt_range[:, 0]
    upper_limits = jnt_range[:, 1]
    
    # Map [-1, 1] to [lower_limits, upper_limits]
    target_qpos = lower_limits + (action + 1.0) * 0.5 * (upper_limits - lower_limits)
    
    return target_qpos


def compute_reward(ee_tracking_error: float) -> float:
    """
    TODO: 
    Calculate the reward based on the distance (error) to the target. 
    Remember from the lecture slides that there are different types of rewards, e.g. dense and sparse. 
    In reward design, it is often useful to combine these approaches. 
    We do not expect you to take into account any advanced reward engineering in this exercise, such as penalizing large velocity and acceleration.
    You can design your own reward function for the bonus question.

    Descrtion of the reward function:
    - dense_reward = exp(-2 * ee_tracking_error)
    - sparse_reward = 1.0 if ee_tracking_error < 0.005 else 0.0
    - reward = dense_reward + sparse_reward

    Inputs:
    - ee_tracking_error: float. Distance between end-effector and target point. Dimensionality: scalar

    Returns:
    - reward: float. The computed reward based on the tracking error. Dimensionality: scalar
    """
    # Dense reward component
    dense_reward = np.exp(-2 * ee_tracking_error)
    
    # Sparse reward component
    sparse_reward = 1.0 if ee_tracking_error < 0.005 else 0.0
    
    # Total reward
    reward = dense_reward + sparse_reward
    
    return reward


def get_obs(qpos: np.ndarray, ee_pos_w: np.ndarray, ee_rot_w: np.ndarray, base_pos_w: np.ndarray, base_rot_w: np.ndarray, target_pos_w: np.ndarray) -> np.ndarray:
    """
    TODO: Extract the observation vector from the environment robot state variables. 

     Note that in Mujoco, states can be directly accessed in the world frame. But for policy genealization, it is important to represent 
     the states in the robot's base frame instead of the world frame, so that the policy can be invariant to the robot's absolute position in the world.
    
    Inputs:
    - qpos: np.ndarray. Current joint positions. Dimensionality: 1D array, Shape: (num_joints,).
    - ee_pos_w: np.ndarray. Current end-effector 3D position in world frame. Dimensionality: 1D array, Shape: (3,).
    - ee_rot_w: np.ndarray. Current end-effector 3D rotation matrix in world frame. Dimensionality: 2D array, Shape: (3, 3).
    - base_pos_w: np.ndarray. Current base 3D position in world frame. Dimensionality: 1D array, Shape: (3,).
    - base_rot_w: np.ndarray. Current base 3D rotation matrix in world frame. Dimensionality: 2D array, Shape: (3, 3).
    - target_pos_w: np.ndarray. Current target 3D position in world frame. Dimensionality: 1D array, Shape: (3,).

    Returns:
    - obs: np.ndarray. The observation vector containing the following robot state variables in order:
        [
            - joint positions (qpos)
            - end-effector position in robot's base frame (ee_pos_base)
            - end-effector quaternion in robot's base frame, must be normalized to represent a valid rotation (ee_quat_base)
            - target position in robot's base frame (target_pos_base)
        ]

    Hints: You can use the provided functions quat_mul, quat_conjugate, quat_normalize, rot_mat_to_quat for quaternion operations.
    """
    # Transform positions to base frame
    ee_rel_w = ee_pos_w - base_pos_w
    target_rel_w = target_pos_w - base_pos_w
    
    # Rotate vectors into base frame using the transpose of the rotation matrix
    ee_pos_base = base_rot_w.T @ ee_rel_w
    target_pos_base = base_rot_w.T @ target_rel_w
    
    # Transform orientation to base frame
    q_base_w = rot_mat_to_quat(base_rot_w)
    q_ee_w = rot_mat_to_quat(ee_rot_w)
    
    # Compute relative quaternion: q_rel = q_base_inverse * q_ee
    q_base_inv = quat_conjugate(q_base_w)
    ee_quat_base = quat_mul(q_base_inv, q_ee_w)
    
    # Normalize the resulting quaternion to ensure validity
    ee_quat_base = quat_normalize(ee_quat_base)
    
    # Concatenate all observations
    obs = np.concatenate([qpos, ee_pos_base, ee_quat_base, target_pos_base])
    
    return obs