import numpy as np
import control as ct
import yaml

with open('params.yaml', 'r') as cfg:
    params = yaml.safe_load(cfg)

g = params["mech"]["g"]

# Link 1
m1 = params["mech"]["m1"]
com1 = params["mech"]["com1"]  # 3-element list [x, y, z]
I_com1 = np.array(params["mech"]["I1p"])  # 3x3 matrix
I1 = I_com1[2][2]

l1 = params["mech"]["l1"]
b1 = params["mech"]["b1"]
L1 = params["mech"]["L1"]

# Link 2
m2 = params["mech"]["m2"]
com2 = params["mech"]["com2"]  # 3-element list [x, y, z]
I_com2 = np.array(params["mech"]["I2p"])  # 3x3 matrix
I2 = I_com2[2][2]

l2 = params["mech"]["l2"]
b2 = params["mech"]["b2"]
L2 = params["mech"]["L2"]

def dh_transform(alpha: float, a: float, d: float, theta: float) -> np.ndarray:
    """
    Compute transformation matrix using modified DH parameters.

    T = Rot_x(alpha) * Trans_x(a) * Rot_z(theta) * Trans_z(d)
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    return np.array([
        [ct,    -st,     0,      a],
        [st*ca,  ct*ca, -sa, -sa*d],
        [st*sa,  ct*sa,  ca,  ca*d],
        [0,      0,      0,      1]
    ]) 

def forward_kinematics(q):
    dh_params = np.array(params["dh_params"])

    # Joint 1
    alpha0, a0, d0, theta_offset0 = dh_params[0]
    theta0 = q[0] + theta_offset0
    T_0_1 = dh_transform(alpha0, a0, d0, theta0)

    # Joint 2
    alpha1, a1, d1, theta_offset1 = dh_params[1]
    theta1 = q[1] + theta_offset1
    T_1_2 = dh_transform(alpha1, a1, d1, theta1)
    
    # End effector
    T_2_3 = np.array([
        [1, 0, 0, L2],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    T = T_0_1 @ T_1_2 @ T_2_3
    
    return T

def inverse_kinematics(target_pos, L1, L2, elbow_up=True):
    x_t, y_t, _, __ = target_pos
    r2 = x_t**2 + y_t**2

    # Check reachability with law of cosines
    cos_th2 = (r2 - L1**2 - L2**2) / (2 * L1 * L2)
    if abs(cos_th2) > 1.0:
      raise ValueError("Target out of reach")

    # elbow-up or elbow-down choice
    if elbow_up:
      th2 = np.arccos(cos_th2)
    else:
      th2 = -np.arccos(cos_th2)

    # theta1
    k1 = L1 + L2 * np.cos(th2)
    k2 = L2 * np.sin(th2)
    th1 = np.arctan2(y_t, x_t) - np.arctan2(k2, k1)

    return np.array([th1, th2]) 

def jacobian(q, L1, L2):
    th1 = q[0]
    th2 = q[1]

    J = np.array([
        [-L1*np.sin(th1) - L2*np.sin(th1+th2), -L2*np.sin(th1+th2)],
        [L1*np.cos(th1) + L2*np.cos(th1+th2), L2*np.cos(th1+th2)]
    ])
    return J

"""
M(q)*qdd + V(q, qd) + G(q) + F*qd = T

with q = [th1, th2]
"""

M = lambda q: np.array([
        [I1 + I2 + m1*l1**2 + m2*l1**2 + m2*l2**2 + 2*m2*l1*l2*np.cos(q[1]), I2 + m2*l2**2 + m2*l1*l2*np.cos(q[1])],
        [I2 + m2*l2**2 + m2*l1*l2*np.cos(q[1]), I2 + m2*l2**2]
    ])

V = lambda q, qd: np.array([
        [-2*m2*l1*l2*np.sin(q[1])*qd[0]*qd[1] - m2*l1*l2*np.sin(q[1])*qd[1]**2],
        [m2*l1*l2*np.sin(q[1])*qd[0]**2]
    ])

G = lambda q: np.array([
        [m1*g*l1*np.cos(q[0]) + m2*g*l1*np.cos(q[0]) + m2*g*l2*np.cos(q[0]+q[1])],
        [m2*g*l2*np.cos(q[0]+q[1])]
    ])

F = np.diag([b1, b2])