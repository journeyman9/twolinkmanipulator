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
print("\nA: \n", A)
print("\nB: \n", B)

# x = [th1, th2, w1, w2]
# u = [tau1, tau2]
print("\nNumerical")
eigval, eigvec = np.linalg.eig(A)
np.set_printoptions(precision=4, suppress=True)
for i in range(len(eigval)):
    print("x{} approx e ^ ({:.2f})t * {}".format(i, eigval[i], eigvec[:, i]))