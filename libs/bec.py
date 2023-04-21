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


@dataclass(frozen=True)
class H_eff_total:
    model: BEC_Qubits

    def a(self, i):
        dim = self.model.n_bosons + 1
        identity = [qutip.identity(dim)] * 2
        destroy = [qutip.destroy(dim), qutip.identity(dim)]
        if i == 0:
            return qutip.tensor(*(destroy + identity))
        if i == 1:
            return qutip.tensor(*(identity + destroy))
        raise ValueError(f"only two qubits supported (0 or 1), not {i}")

    def b(self, i):
        dim = self.model.n_bosons + 1
        identity = [qutip.identity(dim)] * 2
        destroy = [qutip.identity(dim), qutip.destroy(dim)]
        if i == 0:
            return qutip.tensor(*(destroy + identity))
        if i == 1:
            return qutip.tensor(*(identity + destroy))
        raise ValueError(f"only two qubits supported (0 or 1), not {i}")

    def sz(self, i):
        return self.a(i).dag() * self.a(i) - self.b(i).dag() * self.b(i)

    def __call__(self):
        omega = (self.model.n_bosons + 2) * self.model.Omega * math.cos(
            self.model.phase
        ) - self.model.g**2 * self.model.omega0 / 4 / self.model.delta
        d = self.model.Omega * math.cos(self.model.phase)
        print(f"omega = {omega:.3e}; d = {d:.3e}")
        return -d * self.sz(0) * self.sz(1) + omega * (self.sz(0) + self.sz(1))

    def vacuum_state(self):
        return qutip.tensor(
            *itertools.chain(4 * [qutip.fock(self.model.n_bosons + 1, 0)])
        )

    def coherent_state(self, alpha=1 / math.sqrt(2), beta=1 / math.sqrt(2)):
        return (
            1
            / math.sqrt(math.factorial(self.model.n_bosons))
            * (alpha * self.a(0).dag() + beta * self.b(0).dag()) ** self.model.n_bosons
            / math.sqrt(math.factorial(self.model.n_bosons))
            * (alpha * self.a(1).dag() + beta * self.b(1).dag()) ** self.model.n_bosons
            * self.vacuum_state()
        )
