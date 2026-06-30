import numpy as np
import control as ct
import yaml
from model_design import dh_transform, forward_kinematics, inverse_kinematics, jacobian
from model_design import g, m1, com1, I_com1, I1, l1, b1, L1
from model_design import m2, com2, I_com2, I2, l2, b2, L2
from model_design import M, V, G, F
from scipy.optimize import root

q_curr = [0.523, -0.785]

# Find current end effector position from current joint angles
T = forward_kinematics(q_curr)
p_current = T @ [0, 0, 0, 1]

p_arb = [0.7, 0.3, 0, 1]
q_arb = inverse_kinematics(p_arb, L1, L2, elbow_up=True)
qd_arb = [0, 0]

# Manipulability - measure how far away from singularity
J = jacobian(q_arb, L1, L2)
detJ = np.linalg.det(J)
if np.isclose(detJ, 0.0):
    print("Manipulator at singular configuration")
else:
    detJJT = np.linalg.det(J @ J.T)
    w = np.sqrt(detJJT)
    print("Manipulability w: ", w)

M_arb = M(q_arb)
V_arb = V(q_arb, qd_arb)
G_arb = G(q_arb)

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

"""
Equilibrium point of open loop system
M(q)(0) + 0 + G(q) + 0 = tau
tau = G(q), but tau = 0
G(q) = 0

[pi/2, 0], [pi/2, pi], [-pi/2, 0], [-pi/2, pi]
"""

# Static arbitrary x = (q_arb, qd=0)
tau = G(q_arb)

qdd = np.linalg.inv(M(q_arb)) @ (tau - V_arb - G_arb - (F @ qd_arb).reshape(2, 1))

Minv = np.linalg.inv(M_arb)

dM_dq_12 = dM_dq(q_arb)
dM_dq_1 = dM_dq_12[0]
dM_dq_2 = dM_dq_12[1]
Minv_dM_dq_qdd = -Minv @ dM_dq_1 @ qdd - Minv @ dM_dq_2 @ qdd
Aq = Minv_dM_dq_qdd - Minv @ (dV_dq(q_arb, qd_arb) + dG_dq(q_arb))
Aqd = -Minv @ (dV_dqd(q_arb, qd_arb) + dF_dqd)

A = np.block([
    [np.zeros((2, 2)), np.eye(2)],
    [ Aq,              Aqd]
])

print("\nA: \n", A)

B = np.vstack([
    np.zeros((2, 2)),
    Minv
])
print("\nB: \n", B)

# x = [th1, th2, w1, w2]
# u = [tau1, tau2]
print("\nAnalytical")
eigval, eigvec = np.linalg.eig(A)
np.set_printoptions(precision=4, suppress=True)
for i in range(len(eigval)):
    print("x{} approx e ^ ({:.2f})t * {}".format(i, eigval[i], eigvec[:, i]))