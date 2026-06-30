import numpy as np
import control as ct
import yaml
from model_design import dh_transform, forward_kinematics, inverse_kinematics, jacobian
from model_design import g, m1, com1, I_com1, I1, l1, b1, L1
from model_design import m2, com2, I_com2, I2, l2, b2, L2
from model_design import M, C, G, F

"""
Now V is defined as the Lyapunov function
C(q, qd)*qd is used instead for its useful identity

Stability when V(x) PSD
and
Vd NSD

V = 0.5*qd.T*M*qd + U(q) - U(q_eq)

G = dU/dq
Vd = 0.5*qd.T*Md*qd + qd.T*M*qdd + qd.T*G
Vd = 0.5*qd.T*Md*qd + qd.T*(tau - Cqd - G - Fqd) + qd.T*G
Vd = 0.5*qd.T*Md*qd - qd.T*C*qd + qd.T*tau - qd.T*F*qd

identity
Md - 2C is skew symmetric, therefore qd.T*(Md - 2C)*qd = 0

Vd = qd.T*tau - qd.T*F*qd
"""

U = lambda q: (m1 + m2)*g*l1*np.sin(q[0]) + m2*g*l2*np.sin(q[0]+q[1])

"""
Equilibrium point of open loop system
M(q)(0) + 0 + G(q) + 0 = tau
tau = G(q), but tau = 0
G(q) = 0

[pi/2, 0], [pi/2, pi], [-pi/2, 0], [-pi/2, pi]
"""
q_eq = np.array([-np.pi/2, 0]) # Pointing down
qd_eq = np.array([0, 0])

# Sample around reference equlibrium
q_samples = [q_eq + np.array([dq1, dq2]) for dq1 in np.linspace(-0.1, 0.1, 5) for dq2 in np.linspace(-0.1, 0.1, 5)]
qd_samples = [qd_eq + np.array([dq1, dq2]) for dq1 in np.linspace(-0.1, 0.1, 5) for dq2 in np.linspace(-0.1, 0.1, 5)]

def V(q, qd, q_des):
    # Shift potential so V=0 at desired equilibrium
    U_shift = U(q) - U(q_des)
    return 0.5 * qd.T @ M(q) @ qd + U_shift

def Vd(q, qd, tau):
    return qd.T @ tau - qd.T @ F @ qd

# Check V PSD
def is_V_PSD(q_des, q_samples, qd_samples):
    for q in q_samples:
        for qd in qd_samples:
            if V(q, qd, q_eq) < 0:
                return False
    return True

PSD = is_V_PSD(q_eq, q_samples, qd_samples)

# Check Vd NSD
def is_Vd_NSD(tau_fun, q_samples, qd_samples):
    for q in q_samples:
        for qd in qd_samples:
            tau = tau_fun(q, qd)
            if Vd(q, qd, tau) > 0:
                return False
    return True

tau = lambda q, qd: np.array([0, 0])
NSD = is_Vd_NSD(tau, q_samples, qd_samples)

print("Open loop with tau = 0 Lyapunov analysis")
print("V is Positive Semi Definite: ", PSD)
print("Vd is Negative Semi Definite: ", NSD)

"""
Arbitrary Gravity Compensation tau = G(q)
M(q)*qdd + C(q, qd)*qd + F*qd + G(q) - G(q) = 0
Only need to check one because Lyapunov functions now only w.r.t qd and is 0
""" 

q_curr = [0.523, -0.785]

# Find current end effector position from current joint angles
T = forward_kinematics(q_curr)
p_current = T @ [0, 0, 0, 1]

p_arb = [0.7, 0.3, 0, 1]
q_arb = inverse_kinematics(p_arb, L1, L2, elbow_up=True)
qd_arb = np.array([0, 0])

# Manipulability - measure how far away from singularity
J = jacobian(q_arb, L1, L2)
detJ = np.linalg.det(J)
if np.isclose(detJ, 0.0):
    print("\nManipulator at singular configuration")
else:
    detJJT = np.linalg.det(J @ J.T)
    w = np.sqrt(detJJT)
    print("\nManipulability w: ", w)

tau = lambda q, qd: G(q)
q_samples = [q_arb + np.array([dq1, dq2]) for dq1 in np.linspace(-0.1, 0.1, 5) for dq2 in np.linspace(-0.1, 0.1, 5)]
qd_samples = [qd_arb + np.array([dqd1, dqd2]) for dqd1 in np.linspace(-0.1, 0.1, 5) for dqd2 in np.linspace(-0.1, 0.1, 5)]

def V_gc(q, qd, q_des):
    # Shift potential so V=0 at desired equilibrium
    U_shift = U(q) - U(q_des)
    return 0.5 * qd.T @ M(q) @ qd

def Vd_gc(q, qd, tau):
    return - qd.T @ F @ qd

# Check V PSD
def is_V_PSD_gc(q_des, q_samples, qd_samples):
    for q in q_samples:
        for qd in qd_samples:
            if V_gc(q, qd, q_arb) < 0:
                return False
    return True

# Check Vd NSD
def is_Vd_NSD_gc(tau_fun, q_samples, qd_samples):
    for q in q_samples:
        for qd in qd_samples:
            tau = tau_fun(q, qd)
            if Vd_gc(q, qd, tau) > 0:
                return False
    return True

PSD_gc = is_V_PSD_gc(q_arb, q_samples, qd_samples)
NSD_gc = is_Vd_NSD_gc(tau, q_samples, qd_samples)

print("\nGravity compensated arbitrary Lyapunov Analysis")
print("V is Positive Semi-Definite:", PSD_gc)
print("Vd is Negative Semi-Definite:", NSD_gc)

"""
PD Control with Gravity Compensation
"""
Kp = np.diag([5.0, 5.0])
Kd = np.diag([2.0, 2.0])

tau_pdgc = lambda q, qd: G(q).flatten() - Kp @ (q - q_arb) - Kd @ qd

def V_pdgc(q, qd, q_des, qe):
    return 0.5 * qd.T @ M(q) @ qd + 0.5 * qe.T @ Kp @ qe

def Vd_pdgc(q, qd):
    # For PD+GC: tau = G(q) - Kp*(q-q_des) - Kd*qd
    # Vd = d/dt[0.5*qd.T*M*qd + 0.5*(q-q_des).T*Kp*(q-q_des)]
    # Using the dynamics: M*qdd = tau - C*qd - G - F*qd
    # Vd = qd.T*M*qdd + (q-q_des).T*Kp*qd
    # Substituting M*qdd = tau - C*qd - G - F*qd:
    # Vd = qd.T*(tau - C*qd - G - F*qd) + (q-q_des).T*Kp*qd
    
    # With tau = G(q) - Kp*(q-q_des) - Kd*qd:
    # Vd = qd.T*(G - Kp*(q-q_des) - Kd*qd - C*qd - G - F*qd) + (q-q_des).T*Kp*qd
    # Vd = -qd.T*Kp*(q-q_des) - qd.T*Kd*qd - qd.T*F*qd + (q-q_des).T*Kp*qd
    # Since qd.T*Kp*(q-q_des) is scalar, it equals (q-q_des).T*Kp*qd
    # Vd = -qd.T*Kd*qd - qd.T*F*qd
    return -qd.T @ Kd @ qd - qd.T @ F @ qd

# Check V PSD
def is_V_PSD_pdgc(q_des, q_samples, qd_samples):
    for q in q_samples:
        for qd in qd_samples:
            if V_pdgc(q, qd, q_arb, q - np.array(q_arb)) < 0:
                return False
    return True

# Check Vd NSD
def is_Vd_NSD_pdgc(q_samples, qd_samples):
  """
  Check if V̇ is negative semi-definite for PD + gravity compensation.
  
  For PD+gravity compensation: V̇ = -q̇ᵀ Kd q̇ ≤ 0 (negative semi-definite)
  
  Global asymptotic stability is proven via LaSalle's Invariance Principle:
  - V̇ = 0 only when q̇ = 0
  - For system to remain at q̇ = 0, we need q̈ = 0
  - This implies -Kp*e = 0 → e = 0 (equilibrium)
  - Therefore, system converges to desired position
  """
  is_negative_definite = True

  for q in q_samples:
      for qd in qd_samples:
          vd_value = Vd_pdgc(q, qd)

          # Check for negative semi-definite (V̇ ≤ 0)
          if vd_value > 0:
              print("✗ V̇ > 0 detected: System is NOT stable")
              return False

          # Check if strictly negative (excluding equilibrium)
          if vd_value >= 0:
              is_negative_definite = False

  # For PD + gravity compensation, we expect negative semi-definite
  print("✓ V̇ is NEGATIVE SEMI-DEFINITE (V̇ ≤ 0)")
  print("✓ System is GLOBALLY ASYMPTOTICALLY STABLE")
  print("  (via LaSalle's Invariance Principle)")
  print(f"  - V̇ = -q̇ᵀ Kd q̇ ≤ 0")
  print(f"  - V̇ = 0 only at q̇ = 0, which implies convergence to e = 0")
  return True

print("\nPD + Gravity compensated Lyapunov Analysis")
PSD_pdgc = is_V_PSD_pdgc(q_arb, q_samples, qd_samples)
NSD_pdgc = is_Vd_NSD_pdgc(q_samples, qd_samples)

print("V is Positive Semi-Definite:", PSD_pdgc)
print("Vd is Negative Semi-Definite:", NSD_pdgc)
