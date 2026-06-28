import numpy as np
import control as ct
import yaml
from model_design import dh_transform, forward_kinematics, inverse_kinematics, jacobian
from model_design import g, m1, com1, I_com1, I1, l1, b1, L1
from model_design import m2, com2, I_com2, I2, l2, b2, L2
from model_design import M, V, G, F

q_curr = [0.523, -0.785]

# Find current end effector position from current joint angles
T = forward_kinematics(q_curr)
p_current = T @ [0, 0, 0, 1]

p_target = [0.7, 0.3, 0, 1]
q_target = inverse_kinematics(p_target, L1, L2, elbow_up=True)
qd_target = [0, 0]

# Manipulability - measure how far away from singularity
J = jacobian(q_target, L1, L2)
detJ = np.linalg.det(J)
if np.isclose(detJ, 0.0):
    print("Manipulator at singular configuration")
else:
    detJJT = np.linalg.det(J @ J.T)
    w = np.sqrt(detJJT)
    print("Manipulability w: ", w)

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

print("\nA: \n", A)

B = np.vstack([
    np.zeros((2, 2)),
    Minv
])

print("\nB: \n", B)

# x = [th1, th2, w1, w2]
# u = [tau1, tau2]
print("\nStatic")
eigval, eigvec = np.linalg.eig(A)
np.set_printoptions(precision=4, suppress=True)
for i in range(len(eigval)):
    print("x{} approx e ^ ({:.2f})t * {}".format(i, eigval[i], eigvec[:, i]))  