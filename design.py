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

q_curr = [0.523, -0.785]

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

# Find current end effector position from current joint angles
T = forward_kinematics(q_curr)
p_current = T @ [0, 0, 0, 1]

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

p_target = [0.7, 0.3, 0, 1]
q_target = inverse_kinematics(p_target, L1, L2, elbow_up=True)

def jacobian(q, L1, L2):
    th1 = q[0]
    th2 = q[1]

    J = np.array([
        [-L1*np.sin(th1) - L2*np.sin(th1+th2), -L2*np.sin(th1+th2)],
        [L1*np.cos(th1) + L2*np.cos(th1+th2), L2*np.cos(th1+th2)]
    ])
    return J

# Manipulability - measure how far away from singularity
J = jacobian(q_target, L1, L2)
detJ = np.linalg.det(J)
if np.isclose(detJ, 0.0):
    print("Manipulator at singular configuration")
else:
    detJJT = np.linalg.det(J @ J.T)
    w = np.sqrt(detJJT)
    print("Manipulability w: ", w)

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

qd_target = [0, 0]

M_arb = M(q_target)
V_arb = V(q_target, qd_target)
G_arb = G(q_target)

"""
Choose tau0 = G(q) for static config
V_arb = 0, F*qd = 0
Linearize dx = x - x0, du = tau - tau0

A = | 0               I   |
    | -M^-1*dG/dq     0   |
    
B = [
    0,
    M^-1
]
"""

# Partial of G
dG_dq = lambda q: np.array([
    [-m1*g*l1*np.sin(q[0]) - m2*g*l1*np.sin(q[0]) - m2*g*l2*np.sin(q[0]+q[1]),    -m2*g*l2*np.sin(q[0]+q[1])],
    [-m2*g*l2*np.sin(q[0]+q[1]),       -m2*g*l2*np.sin(q[0]+q[1])]
])

Minv = np.linalg.inv(M_arb)

A = np.block([
    [np.zeros((2, 2)), np.eye(2)],
    [-Minv @ dG_dq(q_target),  np.zeros((2, 2))]
])

print(A)

B = np.vstack([
    np.zeros((2, 2)),
    Minv
])

# x = [th1, th2, w1, w2]
# u = [tau1, tau2]
print("\nStatic")
eigval, eigvec = np.linalg.eig(A)
np.set_printoptions(precision=4, suppress=True)
for i in range(len(eigval)):
    print("x{} approx e ^ ({:.2f})t * {}".format(i, eigval[i], eigvec[:, i]))
    
"""
Linearizing about q0, qd0, qd0, tau0
Linearize dx = x - x0, du = tau - tau0

A = | 0      I   |
    | Aq     Aqd |

Aq = -M^-1*(dM/dq)*qdd - M^-1*(dV/dq + dG/dq)
Aqd = -M^-1*(dV/dqd + dF/dqd)

B = [
    0,
    M^-1
]
"""
# Partial of G
dG_dq = lambda q: np.array([
    [-m1*g*l1*np.sin(q[0]) - m2*g*l1*np.sin(q[0]) - m2*g*l2*np.sin(q[0]+q[1]),    -m2*g*l2*np.sin(q[0]+q[1])],
    [-m2*g*l2*np.sin(q[0]+q[1]),       -m2*g*l2*np.sin(q[0]+q[1])]
])

dV_dq = lambda q, qd: np.array([
    [0,      -2*m2*l1*l2*np.cos(q[1])*qd[0]*qd[1] - m2*l1*l2*np.cos(q[1])*qd[1]**2],
    [0,       m2*l1*l2*np.cos(q[1])*qd[0]**2]
])

# Partial derivative of V w.r.t qd
dV_dqd = lambda q, qd: np.array([
  [-2*m2*l1*l2*np.sin(q[1])*qd[1],      -2*m2*l1*l2*np.sin(q[1])*qd[0] - 2*m2*l1*l2*np.sin(q[1])*qd[1]],
  [ 2*m2*l1*l2*np.sin(q[1])*qd[0],       0]
])

# Partial derivative of F@qd w.r.t qd is simply F
dF_dqd = F

dM_dq = lambda q: np.array([
    [  # ∂M/∂q1
        [0.0, 0.0],
        [0.0, 0.0]
    ],
    [  # ∂M/∂q2
        [-2*m2*l1*l2*np.sin(q[1]), -m2*l1*l2*np.sin(q[1])],
        [-m2*l1*l2*np.sin(q[1]), 0.0]
    ]
])

tau = G(q_target)
qdd = np.linalg.inv(M(q_target)) @ (tau - V_arb - G_arb - (F @ qd_target).reshape(2, 1))

Minv = np.linalg.inv(M_arb)

dM_dq_12 = dM_dq(q_target)
dM_dq_1 = dM_dq_12[0]
dM_dq_2 = dM_dq_12[1]
Minv_dM_dq_qdd = -Minv @ dM_dq_1 @ qdd - Minv @ dM_dq_2 @ qdd
Aq = Minv_dM_dq_qdd - Minv @ (dV_dq(q_target, qd_target) + dG_dq(q_target))
Aqd = -Minv @ (dV_dqd(q_target, qd_target) + dF_dqd)

A = np.block([
    [np.zeros((2, 2)), np.eye(2)],
    [ Aq,              Aqd]
])

print(A)

B = np.vstack([
    np.zeros((2, 2)),
    Minv
])

# x = [th1, th2, w1, w2]
# u = [tau1, tau2]
print("\nAnalytical")
eigval, eigvec = np.linalg.eig(A)
np.set_printoptions(precision=4, suppress=True)
for i in range(len(eigval)):
    print("x{} approx e ^ ({:.2f})t * {}".format(i, eigval[i], eigvec[:, i]))

"""
Linearize about q0, qd0, qd0, tau0
Linearize dx = x - x0, du = tau - tau0
"""
def dynamics(x, u):
    """
    Nonlinear state dynamics

    x = [th1, th2, th1_dot, th2_dot]
    tau = [tau1, tau2]
    """

    q = x[:2]
    qd = x[2:]
    tau = u.flatten()

    qdd = np.linalg.solve(
        M(q),
        tau - V(q, qd).flatten() - G(q).flatten() - F @ qd
    )

    return np.concatenate((qd, qdd))

def linearize(x_eq, u_eq):
    eps = 1e-6
    n = len(x_eq)
    m = len(u_eq)
    A = np.zeros((n, n))
    B = np.zeros((n, m))

    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        A[:, i] = (
            dynamics(x_eq + dx, u_eq) - dynamics(x_eq - dx, u_eq)
        ) / (2*eps)

    for j in range(m):
        du = np.zeros(m)
        du[j] = eps
        B[:, j] = (
            dynamics(x_eq, u_eq + du) - dynamics(x_eq, u_eq - du)
        ) / (2*eps)

    return A, B

x0 = np.concatenate((q_target, qd_target))
u0 = G_arb.flatten()
A, B = linearize(x0, u0)
print(A)

# x = [th1, th2, w1, w2]
# u = [tau1, tau2]
print("\nNumerical")
eigval, eigvec = np.linalg.eig(A)
np.set_printoptions(precision=4, suppress=True)
for i in range(len(eigval)):
    print("x{} approx e ^ ({:.2f})t * {}".format(i, eigval[i], eigvec[:, i]))