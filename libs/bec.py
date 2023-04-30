import itertools
import numpy as np
import math
from dataclasses import dataclass

import qutip
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class BEC_Qubits:
    """
    Parameters
    ----------

    coupling_strength :
        Coupling strength of single quantized mode coupled to the transition.

    transition_ampl: float
        Transition amplitude of off-resonant laser.

    transition_freq : float
        Transition frequency.

    resonance_freq : float
        Cavity photon resonance.

    phase: float
        Phase due to propagation of the field through the fiber.

    References
    ----------
    1. See 'Alexey N Pyrkov and Tim Byrnes. New J. Phys. 15 093019 (2013)'
    """

    n_bosons: float
    coupling_strength: float
    transition_ampl: float
    transition_freq: float
    resonance_freq: float
    phase: float

    @property
    def G(self):
        return self.coupling_strength

    @property
    def g(self):
        return self.transition_ampl

    @property
    def omega(self):
        return self.resonance_freq

    @property
    def omega0(self):
        return self.transition_freq

    @property
    def phi(self):
        return self.phase

    @property
    def Omega(self):
        return self.coupling_strength**2 * self.g**2 / 2 / self.delta**3

    @property
    def delta(self):
        """Detuning between the transition to the excited state and the cavity photon resonance."""
        return self.transition_freq - self.resonance_freq

    @classmethod
    def init_alexey2003(
        cls,
        n_bosons,
        phase,
        single_coupling_strength=1.35e6,
        transition_freq=1e7,
        detuning_param: float = 1,
    ):
        """
        See section '4. Estimated gate times' in [1] and [2] for details.

        Parameters
        ----------
        n_bosons :
            Number of bosons

        single_coupling_strength :
            Single atom cavity coupling strength
        d :
            Dimensionless detuning parameter.

        References
        ----------
        1. Alexey N Pyrkov and Tim Byrnes. New J. Phys. 15 093019 (2013)
        2. Colombe Y et al. Nature 450, 272 (2007)
        """

        coupling_strength = math.sqrt(n_bosons) * single_coupling_strength
        transition_ampl = coupling_strength
        delta = detuning_param * single_coupling_strength * n_bosons
        resonance_freq = transition_freq - delta
        return cls(
            n_bosons=n_bosons,
            coupling_strength=coupling_strength,
            transition_ampl=transition_ampl,
            transition_freq=transition_freq,
            resonance_freq=resonance_freq,
            phase=phase,
        )

    @classmethod
    def init_default(cls, n_bosons, phase):
        coupling_strength = 1
        delta = 10
        transition_freq = 11
        resonance_freq = transition_freq - delta
        return cls(
            n_bosons=n_bosons,
            coupling_strength=coupling_strength,
            transition_ampl=coupling_strength,
            transition_freq=transition_freq,
            resonance_freq=resonance_freq,
            phase=phase,
        )

    @property
    def sublevels(self):
        return 2  # Means only 'a' and 'b'


def _build_entire_space(qobj, n, k, m, i):
    """
    Parameters
    ----------
    n : int
        Number of qubits.
    k : int
        Qubit number.
    m : int
        Number of qubit sublevels.
    i : int
        Sublevel number.
    """
    if k is None and n > 1:
        raise NotImplementedError("n={n}; k={k}")

    if k is None and n == 1:
        k = 0

    if k > (n - 1):
        raise ValueError(
            f"qubit number out of range {k} > {n - 1}. Counting starts with zero."
        )

    if i > (m - 1):
        raise ValueError(
            f"sublevel number out of range {i} > {m - 1}. Counting starts with zero."
        )

    i_qobj_subspace = k * m + i
    identity = qutip.identity(qobj.shape[0])
    space = (
        [identity] * i_qobj_subspace
        + [qobj]
        + [identity] * (n * m - i_qobj_subspace - 1)
    )
    return qutip.tensor(*space)


def _destroy(model, n, k, i):
    return _build_entire_space(
        qutip.destroy(model.n_bosons + 1), n, k, m=model.sublevels, i=i
    )


def a(model, n=1, k=None):
    return _destroy(model, n, k, i=0)


def b(model, n=1, k=None):
    return _destroy(model, n, k, i=1)


def sz(model, n=1, k=None):
    return a(model, n, k).dag() * a(model, n, k) - b(model, n, k).dag() * b(model, n, k)


def h_eff_total(model, n=2):
    if n != 2:
        raise NotImplementedError("only qubit pair")

    omega = (model.n_bosons + 2) * model.Omega * math.cos(
        model.phase
    ) - model.g**2 * model.omega0 / 4 / model.delta
    d = model.Omega * math.cos(model.phase)
    print(f"omega = {omega:.3e}; d = {d:.3e}")
    return -d * sz(model, n, 0) * sz(model, n, 1) + omega * (
        sz(model, n, 0) + sz(model, n, 1)
    )


def hzz(model, n=2):
    if n != 2:
        raise NotImplementedError("only qubit pair")

    return model.Omega * sz(model, n, 0) * sz(model, n, 1)


def vacuum_state(model, n=2):
    return qutip.tensor(
        *itertools.chain(n * model.sublevels * [qutip.fock(model.n_bosons + 1, 0)])
    )


def coherent_state_constructor(
    model, n, k, alpha=1 / math.sqrt(2), beta=1 / math.sqrt(2)
):
    return (
        1
        / math.sqrt(math.factorial(model.n_bosons))
        * (alpha * a(model, n, k).dag() + beta * b(model, n, k).dag()) ** model.n_bosons
    )


def fock_state_constructor(model, n, k, i=0):
    """Return operator to create `i`-th eigenstate of Sz,i (Fock states) for `k` qubit from vacuum state."""
    norm = math.sqrt(math.factorial(i) * math.factorial(model.n_bosons - i))
    return (
        1
        / norm
        * a(model, n, k).dag() ** i
        * b(model, n, k).dag() ** (model.n_bosons - i)
    )
