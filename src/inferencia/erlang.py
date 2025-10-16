# src/inferencia/erlang.py
import numpy as np

SLA_TARGET = 0.90
ASA_TARGET_S = 22
INTERVAL_S = 3600
MAX_OCC = 0.85
SHRINKAGE = 0.30
ABSENTEEISM_RATE  = 0.23

def erlang_c_prob_wait(agents: int, load_erlangs: float) -> float:
    if agents <= 0: return 1.0
    if load_erlangs <= 0: return 0.0
    rho = load_erlangs / agents
    if rho >= 1.0: return 1.0
    summation = 0.0
    term = 1.0
    for n in range(0, agents):
        if n > 0: term *= load_erlangs / n
        summation += term
    pn = term * (load_erlangs / agents) / (1 - rho)
    p_wait = pn / (summation + pn)
    return float(np.clip(p_wait, 0.0, 1.0))

def required_agents(arrivals: float, aht_s: float,
                    asa_target_s: float = ASA_TARGET_S,
                    sla_target: float = SLA_TARGET,
                    interval_s: int = INTERVAL_S,
                    max_occ: float = MAX_OCC):
    aht = max(float(aht_s), 1.0)
    lam = max(float(arrivals), 0.0)
    load = lam * aht / interval_s
    base = max(int(np.ceil(load / max_occ)), 1)
    agents = base
    for _ in range(2000):
        p_wait = erlang_c_prob_wait(agents, load)
        asa = (p_wait * aht) / max(agents - load, 1e-9)
        sla = 1.0 - p_wait * np.exp(-(agents - load) * (asa_target_s / aht))
        if sla >= sla_target and asa <= asa_target_s: break
        agents += 1
    return agents, load

def schedule_agents(agents_prod: int) -> int:
    return int(np.ceil(agents_prod / max(1e-9, (1 - SHRINKAGE)) / max(1e-9, (1 - ABSENTEEISM_RATE))))

