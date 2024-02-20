"""A library of vectorial operations that can be applied to quantum registers.

Operations can be applied directly between registers, for example a bitwise
XOR or an addition. This is similar to how registers work on classical computers. Rotations, and
permutations in general, can be made without using any quantum gate because they just return a new
view on the qubits, i.e. a new :class:`QuantumRegister` that has the same logical qubits but in a
different order.

The aim of this library is to be able to implement classical algorithms on quantum computers, to
benefit from the speedup given by methods such as Grover's algorithm.
"""
from __future__ import annotations

import warnings
from itertools import chain
from math import floor, log2
from typing import TYPE_CHECKING

from qiskit import AncillaRegister, ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import Gate
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library import Barrier, CCXGate

if TYPE_CHECKING:
    from typing import Any, Final, Sequence

    from qiskit.result.counts import Counts


FEYNMAN_QC: Final[int] = 1
"""The quantum cost of the Feynman gate (CNOT)."""
TOFFOLI_QC: Final[int] = 5
"""The quantum cost of the Toffoli gate (CCNOT)."""
SINGLE_QC: Final[int] = 1
"""The quantum cost of a single qubit gate."""


def _int_to_bits(k: int, num_bits: int) -> list[int]:
    """Convert integer k into list of bits in LSB-first.

    :param  k: The integer to convert to list of bits.
    :param num_bits: The numbers of bits to use to encode k.
    :returns: A list of bits encoding k.
    """
    # Make sure k is in range
    if k >= 2**num_bits:
        msg = f"{k} cannot be encoded on {num_bits} bits."
        raise ValueError(msg)
    return [int(i) for i in f"{k:0{num_bits}b}"[::-1]]


def _hamming_weight(k: int) -> int:
    """Compute the Hamming weight of an integer.

    :param k: The integer to compute the Hamming weight of.
    :returns: The Hamming weight of k.
    """
    hamming_weight = 0
    while k:
        hamming_weight += 1
        k &= k - 1

    return hamming_weight


class RegisterCircuit(QuantumCircuit):
    """A wrapper around :class:`QuantumCircuit`.

    It implements the logic needed to chain operations on quantum registers.

    It supports new operations that operate on whole quantum registers and handles
    rotations without using any gate by rewiring the circuit when needed. This being also a fully
    valid :class:`QuantumCircuit`, it is also possible to apply operations on single qubits as it is
    normally done in Qiskit.

    :param inputs: The quantum registers to use as the circuit inputs.
    :param kwargs: Other parameters to pass to the underlying :class:`QuantumCircuit` object.
    """

    def __init__(self: RegisterCircuit, *inputs: QuantumRegister, **kwargs: Any) -> None:
        """Initialize a register circuit with inputs and arguments for the quantum circuit."""
        super().__init__(*inputs, **kwargs)

    @property
    def stats(self: RegisterCircuit) -> dict:
        """Some statistics about the circuit.

        - :py:data:`quantum_cost`: The quantum cost of the circuit as defined by [FoM2009]_.
        - :py:data:`depth`: The circuit depth when register operations are decomposed into NOT,
          CNOT and CCNOT gates.
        - :py:data:`gate_counts`: The number of basic gates in the circuit.

        .. warning::
            Quantum cost computation only works if the circuit contains only register
            operations or NOT, CNOT, and CCNOT gates.

        .. [FoM2009]
            Mohammadi, M., & Eshghi, M. (2009). On figures of merit in reversible and
            quantum logic designs. Quantum Information Processing, 8, 297-318.
        """
        stats = {}
        stats["ancilla"] = self.num_ancillas
        stats["qubits"] = self.num_qubits - self.num_ancillas

        # Decompose all register operations in the circuit
        decomposed_circuit = self.decompose(gates_to_decompose=[RegisterOperation])
        stop = False
        while not stop:
            stop = True
            for instruction in decomposed_circuit.data:
                if isinstance(instruction.operation, RegisterOperation):
                    decomposed_circuit = decomposed_circuit.decompose(
                        gates_to_decompose=[RegisterOperation],
                    )
                    stop = False
                    break
        operations = dict(decomposed_circuit.count_ops())
        stats["gate_counts"] = operations
        stats["quantum_cost"] = 0
        for operation, count in operations.items():
            cost = 0
            if operation == "cx":
                cost = FEYNMAN_QC
            elif operation == "ccx":
                cost = TOFFOLI_QC
            elif operation == "x":
                cost = SINGLE_QC
            else:
                warnings.warn(
                    f"Cannot compute quantum cost for operation {operation}.",
                    stacklevel=1,
                )
            stats["quantum_cost"] += cost * count
        stats["depth"] = decomposed_circuit.depth()
        # Some papers only count toffoli gates for the circuit depth
        ccx_only = decomposed_circuit.copy_empty_like()
        for instruction, qargs, cargs in decomposed_circuit:
            if isinstance(instruction, CCXGate):
                ccx_only.append(instruction, qargs, cargs)
        stats["toffoli_depth"] = ccx_only.depth()

        return stats

    def ror(self: RegisterCircuit, X: QuantumRegister, r: int) -> QuantumRegister:
        r"""Rotate right a register by a specified amount of qubits.

        :param X: The register to rotate.
        :param r: The number of qubits by which X should be rotated.
        :returns: The rotated register :math:`X'`.

        :operation: :math:`X' \leftarrow \mathrm{ror}(X, r)`
        """
        operation = RegisterROR(X, r)
        return operation.outputs[0]

    def rol(self: RegisterCircuit, X: QuantumRegister, r: int) -> QuantumRegister:
        r"""Rotate left a register by a specified amount of qubits.

        :param X: The register to rotate.
        :param r: The number of qubits by which X should be rotated.
        :returns: The rotated register :math:`X'`.

        :operation: :math:`X' \leftarrow \mathrm{rol}(X, r)`
        """
        operation = RegisterROL(X, r)
        return operation.outputs[0]

    def xor(self: RegisterCircuit, X: QuantumRegister, Y: QuantumRegister | int) -> QuantumRegister:
        r"""Apply a bitwise XOR between two registers or between one register and a constant.

        :param X: The first register to XOR (the result will be stored in this register, overwriting
          its previous value).
        :param Y: The second register to XOR or a constant.
        :returns: The output register :math:`X`.

        :operation: :math:`X \leftarrow X \oplus Y`

        .. note::
            If Y is a :class:`QuantumRegister`, this operation is a XOR between two registers,
            but if Y is an integer it becomes a XOR between a register and a constant.
        """
        if isinstance(Y, QuantumRegister):
            operation = RegisterXOR(X, Y)
            self.append(operation, list(chain(X, Y)))
        else:
            operation = RegisterConstantXOR(X, Y)
            self.append(operation, list(X))
        return operation.outputs[0]

    def neg(self: RegisterCircuit, X: QuantumRegister) -> QuantumRegister:
        r"""Apply a bitwise NOT on a register.

        :param X: The register to apply the NOT to.
        :returns: The output register :math:`X`.

        :operation: :math:`X \leftarrow \neg X`
        """
        operation = RegisterNOT(X)
        self.append(operation, list(X))
        return operation.outputs[0]

    def add(
        self: RegisterCircuit,
        X: QuantumRegister,
        Y: QuantumRegister | int,
        ancillas: AncillaRegister | None = None,
        mode: str = "ripple",
    ) -> QuantumRegister:
        r"""Add modulo :math:`2^n` two registers of size n or a register of size n and a constant.

        The adder can be implemented using different techniques:

        - :py:data:`ripple`: requires 0 ancilla qubits, uses the ripple-carry method
          from [TTK2009]_.
        - :py:data:`lookahead`: requires :math:`2n-w(n-1)-\lfloor \log(n-1) \rfloor-2` ancilla
          qubits, uses the carry-lookahead method from [DKRS2004]_.

        :param X: First register to add (the result will be stored in this register, overwriting
          its previous value).
        :param Y: Second register to add or a constant.
        :param ancillas: The anquilla register needed in :py:data:`lookahead` mode.
        :mode: The type of adder to use.
        :returns: The output register :math:`X`.

        :operation: :math:`X \leftarrow X+Y \bmod 2^n`

        .. note::
            If Y is a :class:`QuantumRegister`, this operation is an addition between two registers,
            but if Y is an integer it becomes an addition between a register and a constant. In the
            latter case, the mode is always :py:data:`lookahead`.
        """
        if isinstance(Y, QuantumRegister):
            if mode == "ripple":
                operation = RegisterTTKRippleCarryAdder(X, Y)
                self.append(operation, list(chain(X, Y)))
            elif mode == "lookahead":
                if ancillas is None:
                    msg = "Cannot make a carry-lookahead adder without ancilla qubits."
                    raise CircuitError(msg)
                operation = RegisterDKRSCarryLookaheadAdder(X, Y, ancillas)
                self.append(operation, list(chain(X, Y, ancillas)))
            else:
                msg = f"Unknown adder mode {mode}"
                raise CircuitError(msg)
        else:
            if ancillas is None:
                msg = "Cannot make a constant adder without ancilla qubits."
                raise CircuitError(msg)
            operation = RegisterConstantDKRSCarryLookaheadAdder(X, Y, ancillas)
            self.append(operation, list(chain(X, ancillas)))
        return operation.outputs[0]


class RegisterOperation:
    """An operation on registers of qubits.

    :ivar inputs: The inputs registers of the operation.
    :type inputs: Sequence[QuantumRegister]
    :ivar outputs: The outputs registers of the operation.
    :type outputs: Sequence[QuantumRegister]
    """

    def __init__(self: RegisterOperation) -> None:
        """Initialize a register operation."""
        self._inputs: Sequence[QuantumRegister] = []
        self._outputs: Sequence[QuantumRegister] = []

    @property
    def outputs(self: RegisterOperation) -> Sequence[QuantumRegister]:
        """Get the outputs of the operation.

        :returns: A list containing the output quantum registers of the operation.
        """
        return self._outputs

    @property
    def inputs(self: RegisterOperation) -> Sequence[QuantumRegister]:
        """Get the inputs of the operation.

        :returns: A list containing the input quantum registers of the operation.
        """
        return self._inputs


class RegisterPrepare(QuantumCircuit, RegisterOperation):
    r"""A circuit preparing a QuantumRegister to an initial classical integer value.

    :param X: The register to prepare.
    :param value: The value to prepare the register to.

    :operation: :math:`X \leftarrow \mathrm{value}`
    """

    def __init__(self: RegisterPrepare, X: QuantumRegister, value: int) -> None:
        """Initialize a RegisterPrepare operation."""
        num_qubits = len(X)
        circuit = QuantumCircuit(X, name=f"rPrepare {value}")
        bits = _int_to_bits(value, num_qubits)
        for i in range(num_qubits):
            if bits[i]:
                circuit.x(X[i])
            else:
                circuit.id(X[i])
        QuantumCircuit.__init__(self, *circuit.qregs, name=f"rPrepare {value}")
        RegisterOperation.__init__(self)
        self.compose(circuit.to_gate(), qubits=self.qubits, inplace=True)
        self._inputs = [X]
        self._outputs = [X]


class RegisterROR(RegisterOperation):
    r"""Defines the right rotation operation on a quantum register.

    :param X: The register to rotate.
    :param r: The number of qubits by which X should be rotated.
    :raises CircuitError: If r is negative.

    :operation: :math:`X' \leftarrow \mathrm{ror}(X, r)`

    .. warning::
        The result will be stored in register :attr:`self.outputs[0]`.
    """

    def __init__(self: RegisterROR, X: QuantumRegister, r: int) -> None:
        """Initialize a RegisterROR operation."""
        super().__init__()
        if r < 0:
            msg = "Rotation must be by a positive amount."
            raise CircuitError(msg)
        r = r % len(X)
        self._inputs = [X]
        self._outputs = [QuantumRegister(bits=X[r:] + X[:r], name=X.name)]


class RegisterROL(RegisterOperation):
    r"""Defines the left rotation operation on a quantum register.

    :param X: The register to rotate.
    :param r: The number of qubits by which X should be rotated.
    :raises CircuitError: If r is negative.

    :operation: :math:`X' \leftarrow \mathrm{rol}(X, r)`

    .. warning::
        The result will be stored in register :attr:`self.outputs[0]`.
    """

    def __init__(self: RegisterROL, X: QuantumRegister, r: int) -> None:
        """Initialize a RegisterROL operation."""
        super().__init__()
        if r < 0:
            msg = "Rotation must be by a positive amount."
            raise CircuitError(msg)
        r = r % len(X)
        self._inputs = [X]
        self._outputs = [QuantumRegister(bits=X[-r:] + X[:-r], name=X.name)]


class RegisterConstantXOR(Gate, RegisterOperation):
    r"""Bitwise XOR between a quantum register and a constant.

    :param X: The register to XOR with c.
    :param c: The constant to XOR with X.
    :param label: An optional label for the gate.

    :operation: :math:`X \leftarrow X \oplus c`
    """

    def __init__(
        self: RegisterConstantXOR,
        X: QuantumRegister,
        c: int,
        label: str | None = None,
    ) -> None:
        """Initialize a RegisterConstantXOR operation."""
        self.n = len(X)
        self.c = c
        self._inputs = [X]
        self._outputs = [X]
        Gate.__init__(self, "rXORc", self.n, [], label=label)
        RegisterOperation.__init__(self)

        bits = _int_to_bits(self.c, self.n)

        circuit = QuantumCircuit(X, name=f"rXOR {self.c}")

        for i, bit in enumerate(bits):
            if bit:
                circuit.x(X[i])

        self.definition = circuit


class RegisterNOT(Gate, RegisterOperation):
    r"""Bitwise NOT on a quantum register.

    :param X: The register to apply NOT on.
    :param label: An optional label for the gate.

    :oopoeration: :math:`X \leftarrow \neg X`
    """

    def __init__(
        self: RegisterNOT,
        X: QuantumRegister,
        label: str | None = None,
    ) -> None:
        """Initialize a RegisterNOT operation."""
        self.n = len(X)
        self._inputs = [X]
        self._outputs = [X]
        Gate.__init__(self, "rNOT", self.n, [], label=label)
        RegisterOperation.__init__(self)

        circuit = QuantumCircuit(X, name="rNOT")

        for qbit in X:
            circuit.x(qbit)

        self.definition = circuit


class RegisterXOR(Gate, RegisterOperation):
    r"""Bitwise XOR operation between two quantum registers.

    :param X: The first register to XOR.
    :param Y: The second register to XOR.
    :param label: An optional label for the gate.
    :raises CircuitError: If X and Y have a different size.

    :operation: :math:`X \leftarrow X \oplus Y`
    """

    def __init__(
        self: RegisterXOR,
        X: QuantumRegister,
        Y: QuantumRegister,
        label: str | None = None,
    ) -> None:
        """Initialize a RegisterXOR operation."""
        if len(X) != len(Y):
            msg = "rXOR operation must be between two QuantumRegisters of the same size."
            raise CircuitError(msg)
        self.n = len(X)
        self._inputs = [X, Y]
        self._outputs = [X, Y]
        Gate.__init__(self, "rXOR", self.n * 2, [], label=label)
        RegisterOperation.__init__(self)

        circuit = QuantumCircuit(X, Y, name="rXOR")

        for i in range(self.n):
            circuit.cx(Y[i], X[i])

        self.definition = circuit


class RegisterConstantDKRSCarryLookaheadAdder(Gate, RegisterOperation):
    r"""Addition modulo :math:`2^n` between a quantum register and a constant.

    :param X: The register of size n to add c to.
    :param c: The constant to add to X.
    :param label: An optional label for the gate.

    :operation: :math:`X \leftarrow X + c`
    """

    @staticmethod
    def get_num_ancilla_qubits(n: int) -> int:
        """Get the number of required ancilla qubits without instantiating the class.

        :param n: The number of qubits of the vectors to add.
        :returns: The number of ancilla qubits needed for the computation.
        """
        return RegisterDKRSCarryLookaheadAdder.get_num_ancilla_qubits(n)

    def __init__(
        self: RegisterConstantDKRSCarryLookaheadAdder,
        A: QuantumRegister,
        c: int,
        ancillas: AncillaRegister,
        label: str | None = None,
    ) -> None:
        """Initialize a RegisterConstantDKRSCarryLookaheadAdder operation."""
        self.n = len(A)
        self.c = c
        self._inputs = [A, ancillas]
        self._outputs = [A, ancillas]
        Gate.__init__(self, "rADDc", self.n + len(ancillas), [], label=label)
        RegisterOperation.__init__(self)

        if len(ancillas) != self.get_num_ancilla_qubits(self.n):
            msg = (
                f"Circuit needs {self.get_num_ancilla_qubits(self.n)} "
                f"ancilla qubits but {len(ancillas)} were given."
            )
            raise CircuitError(msg)

        circuit = RegisterCircuit(A, ancillas, name=f"rADD {self.c}")

        if self.c != 0:
            bits = _int_to_bits(self.c, self.n)
            Z = AncillaRegister(bits=ancillas[0 : self.n - 1])
            X = AncillaRegister(bits=ancillas[self.n - 1 : :])
            for i in range(self.n - 1):  # Z[i] = g[i, i+1]
                if bits[i]:
                    circuit.cx(A[i], Z[i])
            circuit.xor(A, c)  # A[i] = p[i, i+1] and A[0] = s0

            # Lookahead carry
            if self.n > 1:
                # Compute carry
                compute_carry = RegisterDKRSComputeCarry(QuantumRegister(bits=A[1:-1]), Z, X)
                circuit.append(compute_carry, list(chain(*compute_carry.inputs)))

                # Compute sum
                circuit.xor(QuantumRegister(bits=A[1:]), Z)  # A[i] = si
                # Now do everything in reverse
                circuit.neg(QuantumRegister(bits=A[:-1]))  # A = s'
                if len(A) > 2:
                    circuit.xor(
                        QuantumRegister(bits=A[1:-1]),
                        self.c >> 1 & (2 ** (self.n - 2) - 1),
                    )

                uncompute_carry = RegisterDKRSComputeCarry(
                    QuantumRegister(bits=A[1:-1]),
                    Z,
                    X,
                ).reverse_ops()
                circuit.append(uncompute_carry, list(chain(*uncompute_carry.inputs)))

                if len(A) > 2:
                    circuit.xor(
                        QuantumRegister(bits=A[1:-1]),
                        self.c >> 1 & (2 ** (self.n - 2) - 1),
                    )
                for i in range(self.n - 1):
                    if bits[i]:
                        circuit.cx(A[i], Z[i])
                circuit.neg(QuantumRegister(bits=A[:-1]))

        self.definition = circuit


class RegisterDKRSComputeCarry(Gate, RegisterOperation):
    r"""Carry computation as described in [DKRS2004]_.

    The last carry is not computed.

    :param P0: :math:`P_0[i] = p[i, i+1]`, 1 if and only if carry propagages from bit :math:`i` to
    bit :math:`i+1`.
    :param G: :math:`G[i] = g[i-1, i]`, 1 if and only if a carry is generated between
    bit :math:`i-1` and bit :math:`i`.
    :param ancillas: The ancilla qubits used for the computation. They must be set to 0 before the
    circuit and will be reset to 0.
    """

    def __init__(
        self: RegisterDKRSComputeCarry,
        P0: Sequence[QuantumRegister],
        G: Sequence[QuantumRegister],
        ancillas: AncillaRegister,
        label: str | None = None,
    ) -> None:
        """Initialize a RegisterDKRSComputeCarry operation."""
        self.n = len(P0) + 1
        Gate.__init__(self, "rCarry", len(P0) + len(G) + len(ancillas), [], label=label)
        RegisterOperation.__init__(self)
        self._inputs = [P0, G, ancillas]
        self._outputs = [P0, G, ancillas]

        circuit = RegisterCircuit(P0, G, ancillas, name="rCarry")

        P = [P0]
        ancilla_index = 0
        for t in range(1, floor(log2(self.n))):
            pt_size = floor(self.n / 2**t) - 1
            P.append(QuantumRegister(bits=ancillas[ancilla_index : ancilla_index + pt_size]))
            ancilla_index += pt_size

        # P-rounds
        for t in range(1, floor(log2(self.n))):
            for m in range(1, floor(self.n / 2**t)):
                circuit.ccx(P[t - 1][2 * m - 1], P[t - 1][2 * m], P[t][m - 1])
        # G-rounds
        for t in range(1, floor(log2(self.n)) + 1):
            for m in range(floor(self.n / 2**t)):
                circuit.ccx(G[2**t * m + 2 ** (t - 1) - 1], P[t - 1][2 * m], G[2**t * m + 2**t - 1])
        # C rounds
        for t in range(floor(log2(2 * self.n / 3)), 0, -1):
            for m in range(1, floor((self.n - 2 ** (t - 1)) / 2**t) + 1):
                circuit.ccx(G[2**t * m - 1], P[t - 1][2 * m - 1], G[2**t * m + 2 ** (t - 1) - 1])

        # P^-1 rounds
        for t in range(floor(log2(self.n)) - 1, 0, -1):
            for m in range(1, floor(self.n / 2**t)):
                circuit.ccx(P[t - 1][2 * m - 1], P[t - 1][2 * m], P[t][m - 1])

        self.definition = circuit


class RegisterDKRSCarryLookaheadAdder(Gate, RegisterOperation):
    r"""A n qubits carry-lookahead adder modulo :math:`2^n`.

    It implements the adder described in [DKRS2004]_ but skips the output carry compution.

    :param A: First register of size n to add.
    :param B: Second register of size n to add.
    :param ancillas: The ancilla qubits used for the computation. They must be set to 0 before the
    circuit and will be reset to 0.
    :param label: An optional label for the gate.
    :raises CircuitError: If A and B have a different size or if there is not the correct number of
    ancilla qubits.

    :operation: :math:`X \leftarrow X+Y \bmod 2^n`

    .. [DKRS2004]
        Draper, T. G., Kutin, S. A., Rains, E. M., & Svore, K. M. (2004).
        A logarithmic-depth quantum carry-lookahead adder. arXiv preprint quant-ph/0406142.
    """

    @staticmethod
    def get_num_ancilla_qubits(
        n: int,
    ) -> int:
        """Get the number of required ancilla qubits without instantiating the class.

        :param n: The number of qubits of the vectors to add.
        :returns: The number of ancilla qubits needed for the computation.
        """
        if n < 2:
            return 0
        return 2 * n - _hamming_weight(n - 1) - floor(log2(n - 1)) - 2

    def __init__(
        self: RegisterDKRSCarryLookaheadAdder,
        A: QuantumRegister,
        B: QuantumRegister,
        ancillas: AncillaRegister,
        label: str | None = None,
    ) -> None:
        """Initialize a RegisterDKRSCarryLookaheadAdder operation."""
        self.n = len(A)
        Gate.__init__(self, "rADD", self.n * 2 + len(ancillas), [], label=label)
        RegisterOperation.__init__(self)
        self._inputs = [A, B, ancillas]
        self._outputs = [A, B, ancillas]
        if len(A) != len(B):
            msg = "rADD operation must be between two QuantumRegisters of the same size."
            raise CircuitError(msg)
        circuit = RegisterCircuit(A, B, ancillas, name="rADD")

        if len(ancillas) != self.get_num_ancilla_qubits(self.n):
            msg = "Wrong number of ancilla qubits."
            raise CircuitError(msg)

        Z = AncillaRegister(bits=ancillas[0 : self.n - 1])
        X = AncillaRegister(bits=ancillas[self.n - 1 : :])
        for i in range(self.n - 1):  # Z[i] = g[i, i+1]
            circuit.ccx(B[i], A[i], Z[i])
        circuit.xor(A, B)  # A[i] = p[i, i+1] and A[0] = s0

        # Lookahead carry
        if self.n > 1:
            # Compute carry
            compute_carry = RegisterDKRSComputeCarry(QuantumRegister(bits=A[1:-1]), Z, X)
            circuit.append(compute_carry, list(chain(*compute_carry.inputs)))

            # Compute sum
            circuit.xor(QuantumRegister(bits=A[1:]), Z)  # A[i] = si
            # Now do everything in reverse
            circuit.neg(QuantumRegister(bits=A[:-1]))  # A = s'
            circuit.xor(QuantumRegister(bits=A[1:-1]), QuantumRegister(bits=B[1:-1]))

            uncompute_carry = RegisterDKRSComputeCarry(
                QuantumRegister(bits=A[1:-1]),
                Z,
                X,
            ).reverse_ops()
            circuit.append(uncompute_carry, list(chain(*uncompute_carry.inputs)))

            circuit.xor(QuantumRegister(bits=A[1:-1]), QuantumRegister(bits=B[1:-1]))
            for i in range(self.n - 1):
                circuit.ccx(B[i], A[i], Z[i])
            circuit.neg(QuantumRegister(bits=A[:-1]))

        self.definition = circuit


class RegisterTTKRippleCarryAdder(Gate, RegisterOperation):
    r"""A gate implementing the n qubits ripple-carry adder modulo :math:`2^n`.

    It implements the adder described in [TTK2009]_ but skips the output carry compution.

    :param X: First register of size n to add.
    :param Y: Second register of size n to add.
    :param label: An optional label for the gate.
    :raises CircuitError: If X and Y have a different size.

    :operation: :math:`X \leftarrow X+Y \bmod 2^n`

    .. [TTK2009]
        Takahashi, Y., Tani, S., & Kunihiro, N. (2009). Quantum addition circuits and
        unbounded fan-out. arXiv preprint arXiv:0910.2530.
    """

    def __init__(
        self: RegisterTTKRippleCarryAdder,
        X: QuantumRegister,
        Y: QuantumRegister,
        label: str | None = None,
    ) -> None:
        """Initialize a RegisterTTKRippleCarryAdder operation."""
        if len(X) != len(Y):
            msg = "rADD operation must be between two QuantumRegisters of the same size."
            raise CircuitError(msg)
        self.n = len(X)
        self._inputs = [X, Y]
        self._outputs = [X, Y]
        Gate.__init__(self, "rADD", self.n * 2, [], label=label)
        RegisterOperation.__init__(self)

        circuit = QuantumCircuit(X, Y, name="rADD")

        for i in range(1, self.n):
            circuit.cx(Y[i], X[i])
        for i in range(self.n - 2, 0, -1):
            circuit.cx(Y[i], Y[i + 1])
        for i in range(self.n - 1):
            circuit.ccx(X[i], Y[i], Y[i + 1])
        for i in range(self.n - 1, 0, -1):
            circuit.cx(Y[i], X[i])
            circuit.ccx(X[i - 1], Y[i - 1], Y[i])

        for i in range(1, self.n - 1):
            circuit.cx(Y[i], Y[i + 1])
        for i in range(self.n):
            circuit.cx(Y[i], X[i])

        self.definition = circuit


def simulate(
    circuit: QuantumCircuit,
    method: str = "automatic",
    device: str = "CPU",
    shots: int = 1024,
) -> Counts:
    """Simulate the given circuit and returns the results.

    :param circuit: The circuit to simulate.
    :param method: The method to use for the simulator.
    :param device: The device to run the simulation on (CPU or GPU).
    :param shots: The number of times to run the simulation.
    :returns: The result counts of the simulation.

    .. note::
        This function needs the `qiskit_aer` extra dependency.
    """
    try:
        # pylint: disable=import-outside-toplevel
        from qiskit_aer.backends import (
            AerSimulator,
        )

        # Manually set max_memory_mb to INT_MAX
        # workaround for https://github.com/Qiskit/qiskit-aer/issues/2056
        if method == "matrix_product_state":
            backend_sim = AerSimulator(
                method=method,
                device=device,
                n_qubits=circuit.num_qubits,
                max_memory_mb=2**64 - 1,
            )
        else:
            backend_sim = AerSimulator(method=method, device=device, n_qubits=circuit.num_qubits)

        # Transpile the circuit to convert rOperations into basis gates
        transpiled_circuit = transpile(circuit, backend_sim)

        job_sim = backend_sim.run(transpiled_circuit, shots=shots)
        result_sim = job_sim.result()
        return result_sim.get_counts(circuit)

    except ImportError as exc:
        msg = "qiskit_aer is not installed. Please install it to use the simulate function."
        raise ImportError(msg) from exc


def make_circuit(
    circuit: QuantumCircuit,
    inputs: Sequence[int],
    input_registers: Sequence[QuantumRegister],
    output_registers: Sequence[QuantumRegister],
) -> QuantumCircuit:
    """Make a circuit with registers as input and measurement operations for the output registers.

    Also prepare the initial values of the input registers.

    :param circuit: The base circuit that will be expanded with measurement operations and
        preparation operations.
    :param inputs: A list of the initial values to assign to the input registers.
    :param inputs_registers: A list of the input registers.
    :param output_registers: A list of the output registers.
    :returns: The final circuit containing the preparation step, the base circuit and the
        measurement step.
    """
    # Copy the circuit
    circuit = circuit.copy()
    circuit.compose(Barrier(circuit.num_qubits), inplace=True, front=True)
    # Add classical registers for the output registers
    classical_output_registers = []
    for output_register in output_registers:
        classical_output_register = ClassicalRegister(len(output_register))
        circuit.add_register(classical_output_register)
        classical_output_registers.append(classical_output_register)
    # Prepare the input by initializing it to given values
    for i, input_register in enumerate(input_registers):
        circuit.compose(
            RegisterPrepare(
                input_register,
                inputs[i],
            ),
            input_register,
            inplace=True,
            front=True,
        )
        circuit.barrier()
    # Add measurements for the output
    for output_register, classical_output_register in zip(
        output_registers,
        classical_output_registers,
    ):
        circuit.measure(output_register, classical_output_register)

    return circuit


def run_circuit(
    circuit: QuantumCircuit,
    method: str = "automatic",
    device: str = "CPU",
    shots: int = 1024,
) -> Sequence[int]:
    """Simulate a circuit and retrieve the integer values of the classical registers.

    :param circuit: The circuit to run.
    :param method: The method to use for the simulator.
    :param device: The device to run the simulation on (CPU or GPU).
    :param shots: The number of times to run the simulation.
    :returns: A list of the integers stored in the classical registers of the circuit after the
    circuit has been simulated. It takes into account only the most frequent result.

    .. note::
        This function needs the `qiskit_aer` extra dependency.
    """
    # Simulate the circuit
    raw_result = simulate(circuit, method, device, shots)
    result = raw_result.most_frequent()
    # Extract the output registers (result is MSB-first so we need to reverse the register order)
    outputs = result.split()[::-1]
    # Convert the outputs to integers
    return [int(output, 2) for output in outputs]
