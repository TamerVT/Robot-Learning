import numpy as np


def generate_quintic_spline_waypoints(start, end, num_points):

    """
    TODO:

    Steps:
    1. Generate `num_points` linearly spaced time steps `s` between 0 and 1.
    2. Apply the quintic time scaling polynomial function which can be found in the slides to get `f_s`.
    3. Interpolate between `start` and `end` using `start + (end - start) * f_s`.
    
    Args:
        start (np.ndarray): Starting waypoint.
        end (np.ndarray): Ending waypoint.
        num_points (int): Number of points in the trajectory.
        
    Returns:
        np.ndarray: Generated waypoints.
    """
    # Generate num_points linearly spaced time steps s between 0 and 1
    s = np.linspace(0, 1, num_points)
    
    # Apply the quintic time scaling polynomial
    f_s = 10 * s**3 - 15 * s**4 + 6 * s**5
    f_s = f_s[:, np.newaxis]
    
    # Interpolate between start and end
    waypoints = start + (end - start) * f_s
    
    return waypoints


def pid_control(tracking_error_history, timestep, Kp=150.0, Ki=0.0, Kd=0.01):
    """
    TODO:
    Compute the PID control signal based on the tracking error history.
    
    Steps:
    1. The Proportional (P) term is the most recent error.
    2. The Integral (I) term is the sum of all past errors, multiplied by the simulation timestep.
    3. The Derivative (D) term is the rate of change of the error (difference between the last two errors divided by the timestep).
       If there is only one error in history, the D term should be zero.
    4. Compute the final control signal: Kp * P + Ki * I + Kd * D.
    
    Args:
        tracking_error_history (np.ndarray): History of tracking errors.
        timestep (float): Simulation timestep.
        Kp (float): Proportional gain.
        Ki (float): Integral gain.
        Kd (float): Derivative gain.
        
    Returns:
        np.ndarray: Control signal.
    """
    # Proportional term
    error = tracking_error_history[-1]
    p_term = Kp * error
    
    # Integral term
    i_term = Ki * np.sum(tracking_error_history) * timestep
    
    # Derivative term
    if len(tracking_error_history) > 1:
        d_term = Kd * (tracking_error_history[-1] - tracking_error_history[-2]) / timestep
    else:
        d_term = 0.0
        
    # Compute final control signal
    return p_term + i_term + d_term