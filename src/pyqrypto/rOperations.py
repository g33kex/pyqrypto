"""A library of vectorial operations that can be applied to quantum registers.

In this library, a :class:`QuantumRegister` from `Qiskit <https://qiskit.org/>`_ is seen as a view on a vector of qubits. Operations can be applied directly between registers, for example a bitwise XOR or an addition. This is similar to how registers work on classical computers. Rotations, and permutations in general, can be made without using any quantum gate because they just return a new view on the qubits, i.e. a new :class:`QuantumRegister` that has the same logical qubits but in a different order.

The aim of this library is to be able to implement classical algorithms on quantum computers, to benefit from the speedup given by methods such as Grover's algorithm.
"""
from __future__ import annotations
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister, transpile
from qiskit_aer.backends import AerSimulator
from qiskit.circuit import Gate
from qiskit.circuit.library import Barrier, CCXGate
from qiskit.result.counts import Counts
from qiskit.circuit.exceptions import CircuitError
from itertools import chain
from typing import Optional, Sequence, Final
import warnings
from abc import ABC
from math import floor, log2

FEYNMAN_QC: Final[int] = 1
"""The quantum cost of the Feynman gate (CNOT)."""
TOFFOLI_QC: Final[int] = 5
"""The quantum cost of the Toffoli gate (CCNOT)."""
SINGLE_QC: Final[int] = 1
"""The quantum cost of a single qubit gate."""

def _int_to_bits(k: int, num_bits):
    """Convert integer k into list of bits in LSB-first.

    :param k: The integer to convert to list of bits.
    :param num_bits: The numbers of bits to use to encode k.
    """
    # Make sure k is in range
    if k >= 2**num_bits:
        raise ValueError(f"{k} cannot be encoded on {num_bits} bits.")
    return [int(i) for i in '{:0{n}b}'.format(k, n=num_bits)][::-1]

def _hamming_weight(k: int):
    """Compute the Hamming weight of an integer.

    :param k: The integer to compute the Hamming weight of.
    :returns: The Hamming weight of k.
    """
    c = 0
    while k:
        c += 1
        k &= k - 1

    return c

class rCircuit(QuantumCircuit):
    """A wrapper around :class:`QuantumCircuit` that implements the logic needed to chain operations on qubit vectors. It supports new operations that operate on whole quantum registers and handles rotations without using any gate by rewiring the circuit when needed. This being also a fully valid :class:`QuantumCircuit`, it is also possible to apply operations on single qubits as it is normally done in Qiskit.

    :param inputs: The quantum registers to use as the circuit inputs.
    :param kwargs: Other parameters to pass to the underlying :class:`QuantumCircuit` object.
    """
    def __init__(self, *inputs: QuantumRegister, **kwargs):
        super().__init__(*inputs, **kwargs)

    @property
    def stats(self):
        """Some statistics about the circuit:

        - :py:data:`quantum_cost`: The quantum cost of the circuit as defined by [FoM2009]_.
        - :py:data:`depth`: The circuit depth when rOperations are decomposed into NOT, CNOT and CCNOT gates.
        - :py:data:`gate_counts`: The number of basic gates in the circuit.

        .. warning:: Quantum cost computation only works if the circuit contains only rOperations or NOT, CNOT, and CCNOT gates.

        .. [FoM2009] Mohammadi, M., & Eshghi, M. (2009). On figures of merit in reversible and quantum logic designs. Quantum Information Processing, 8, 297-318.
        """
        stats = {}
        stats['ancilla'] = self.num_ancillas
        stats['qubits'] = self.num_qubits-self.num_ancillas

        # Decompose all rOperations in the circuit
        decomposed_circuit = self.decompose(gates_to_decompose=[rOperation])
        stop = False
        while not stop:
            stop = True
            for instruction in decomposed_circuit.data:
                if isinstance(instruction.operation, rOperation):
                    decomposed_circuit = decomposed_circuit.decompose(gates_to_decompose=[rOperation])
                    stop = False
                    break
        operations = dict(decomposed_circuit.count_ops())
        stats['gate_counts'] = operations
        stats['quantum_cost'] = 0
        for operation, count in operations.items():
            k = 0
            if operation == 'cx':
                k = FEYNMAN_QC
            elif operation == 'ccx':
                k = TOFFOLI_QC
            elif operation == 'x':
                k = SINGLE_QC
            else:
                warnings.warn(f"Cannot compute quantum cost for operation {operation}.")   
            stats['quantum_cost'] += k*count
        stats['depth'] = decomposed_circuit.depth()
        # Some papers only count toffoli gates for the circuit depth...
        ccx_only = decomposed_circuit.copy_empty_like()
        for instruction, qargs, cargs in decomposed_circuit:
            if isinstance(instruction, CCXGate):
                ccx_only.append(instruction, qargs, cargs)
        stats['toffoli_depth'] = ccx_only.depth()
        
        return stats

    def ror(self, X: QuantumRegister, r: int) -> QuantumRegister:
        r"""Rotate right a register by a specified amount of qubits.

        :param X: The register to rotate.
        :param r: The number of qubits by which X should be rotated.
        :returns: The rotated register :math:`X'`.

        :operation: :math:`X' \leftarrow \mathrm{ror}(X, r)`
        """
        op = rROR(X, r)
        return op.outputs[0]

    def rol(self, X: QuantumRegister, r: int) -> QuantumRegister:
        r"""Rotate left a register by a specified amount of qubits.

        :param X: The register to rotate.
        :param r: The number of qubits by which X should be rotated.
        :returns: The rotated register :math:`X'`.

        :operation: :math:`X' \leftarrow \mathrm{rol}(X, r)`
        """
        op = rROL(X, r)
        return op.outputs[0]

    def xor(self, X: QuantumRegister, Y: QuantumRegister | int) -> QuantumRegister:
        r"""Apply a bitwise XOR between two registers or between one register and a constant and store the result in the first register.

        :param X: The first register to XOR (the result will be stored in this register, overwriting its previous value).
        :param Y: The second register to XOR or a constant.
        :returns: The output register :math:`X`.

        :operation: :math:`X \leftarrow X \oplus Y`

        .. note::
            If Y is a :class:`QuantumRegister`, this operation is a XOR between two registers, but if Y is an integer it becomes a XOR between a register and a constant.
        """
        if isinstance(Y, QuantumRegister):
            op = rXOR(X, Y)
            self.append(op, list(chain(X, Y)))
        else:
            op = rConstantXOR(X, Y)
            self.append(op, list(X))
        return op.outputs[0]

    def neg(self, X: QuantumRegister) -> QuantumRegister:
        r"""Apply a bitwise NOT on a register.

        :param X: The register to apply the NOT to.
        :returns: The output register :math:`X`.

        :operation: :math:`X \leftarrow \neg X`
        """
        op = rNOT(X)
        self.append(op, list(X))
        return op.outputs[0]

    def add(self, X: QuantumRegister, Y: QuantumRegister | int, ancillas: Optional[AncillaRegister]=None, mode='ripple') -> QuantumRegister:
        r"""Add modulo :math:`2^n` two registers of size n or a register of size n and a constant and store the result in the first register.

        The adder can be implemented using different techniques:

        - :py:data:`ripple`: requires 0 ancilla qubits, uses the ripple-carry method from [TTK2009]_.
        - :py:data:`lookahead`: requires :math:`2n-w(n-1)-\lfloor \log(n-1) \rfloor-2` ancilla qubits, uses the carry-lookahead method from [DKRS2004]_.

        :param X: First register to add (the result will be stored in this register, overwriting its previous value).
        :param Y: Second register to add or a constant.
        :param ancillas: The anquilla register needed in :py:data:`lookahead` mode.
        :mode: The type of adder to use.
        :returns: The output register :math:`X`.

        :operation: :math:`X \leftarrow X+Y \bmod 2^n`

        .. note::
            If Y is a :class:`QuantumRegister`, this operation is an addition between two registers, but if Y is an integer it becomes an addition between a register and a constant. In the latter case, the mode is always :py:data:`lookahead`.
        """
        if isinstance(Y, QuantumRegister):
            if mode == 'ripple':
                op = rTTKRippleCarryAdder(X, Y)
                self.append(op, list(chain(X, Y)))
            elif mode == 'lookahead':
                if ancillas is None:
                    raise CircuitError("Cannot make a carry-lookahead adder without ancilla qubits.")
                op = rDKRSCarryLookaheadAdder(X, Y, ancillas)
                self.append(op, list(chain(X, Y, ancillas)))
            else:
                raise CircuitError(f"Unknown adder mode {mode}.")
        else:
            if ancillas is None:
                raise CircuitError("Cannot make a constant adder without ancilla qubits.")
            op = rConstantDKRSCarryLookaheadAdder(X, Y, ancillas)
            self.append(op, list(chain(X, ancillas)))
        return op.outputs[0]

class rOperation(ABC):
    """Abstract class defining an operation on registers of qubits.
    
    :ivar inputs: The inputs registers of the operation.
    :type inputs: Sequence[QuantumRegister]
    :ivar outputs: The outputs registers of the operation.
    :type outputs: Sequence[QuantumRegister]
    """
    def __init__(self):
        self._inputs = []
        self._outputs = []

    @property
    def outputs(self) -> list[QuantumRegister]:
        return self._outputs

    @property
    def inputs(self) -> list[QuantumRegister]:
        return self._inputs

class rPrepare(QuantumCircuit, rOperation):
    r"""A circuit preparing a QuantumRegister to an initial classical integer value.

    :param X: The register to prepare.
    :param value: The value to prepare the register to.

    :operation: :math:`X \leftarrow \mathrm{value}`
    """
        
    def __init__(self, X: QuantumRegister, value: int) -> None:
        num_qubits = len(X)
        qc = QuantumCircuit(X, name=f'rPrepare {value}')
        bits = _int_to_bits(value, num_qubits)
        for i in range(num_qubits):
            if bits[i]:
                qc.x(X[i])
            else:
                qc.i(X[i])

        super().__init__(*qc.qregs, name=f'rPrepare {value}')
        self.compose(qc.to_gate(), qubits=self.qubits, inplace=True)
        self._inputs = [X]
        self._outputs = [X]

class rROR(rOperation):
    r"""Defines the right rotation operation on a QuantumRegiser.

    :param X: The register to rotate.
    :param r: The number of qubits by which X should be rotated.
    :raises CircuitError: If r is negative.

    :operation: :math:`X' \leftarrow \mathrm{ror}(X, r)`

    .. warning::
        The result will be stored in register :attr:`self.outputs[0]`.
    """

    def __init__(self, X: QuantumRegister, r: int) -> None:
        if r < 0:
            raise CircuitError("Rotation must be by a positive amount.")
        r = r % len(X)
        self._inputs = [X]
        self._outputs = [QuantumRegister(bits=X[r:]+X[:r], name=X.name)]

class rROL(rOperation):
    r"""Defines the left rotation operation on a QuantumRegister.

    :param X: The register to rotate.
    :param r: The number of qubits by which X should be rotated.
    :raises CircuitError: If r is negative.

    :operation: :math:`X' \leftarrow \mathrm{rol}(X, r)`


    .. warning::
        The result will be stored in register :attr:`self.outputs[0]`.
    """

    def __init__(self, X: QuantumRegister, r: int) -> None:
        if r < 0:
            raise CircuitError("Rotation must be by a positive amount.")
        r = r % len(X)
        self._inputs = [X]
        self._outputs = [QuantumRegister(bits=X[-r:]+X[:-r], name=X.name)]

class rConstantXOR(Gate, rOperation):
    r"""A gate implementing a logical bitwise XOR operation between a register of qubits and a constant.
    
    :param X: The register to XOR with c.
    :param c: The constant to XOR with X.
    :param label: An optional label for the gate.

    :operation: :math:`X \leftarrow X \oplus c`
    """

    def __init__(self, X: QuantumRegister, c: int, label: Optional[str] = None) -> None:
        self.n = len(X)
        self.c = c
        self._inputs = [X]
        self._outputs = [X]
        super().__init__("rXORc", self.n, [], label=label)

        bits = _int_to_bits(self.c, self.n)
    
        qc = QuantumCircuit(X, name=f'rXOR {self.c}')
   
        for i, bit in enumerate(bits):
            if bit:
                qc.x(X[i])
        
        self.definition = qc

class rNOT(Gate, rOperation):
    r"""A gate implementing a bitwise NOT operation on a register

    :param X: The register to apply NOT on.
    :param label: An optional label for the gate.

    :oopoeration: :math:`X \leftarrow \neg X`
    """

    def __init__(self, X: QuantumRegister, label: Optional[str] = None) -> None:
        self.n = len(X)
        self._inputs = [X]
        self._outputs = [X]
        super().__init__("rNOT", self.n, [], label=label)

        qc = QuantumCircuit(X, name="rNOT")

        for q in X:
            qc.x(q)

        self.definition = qc

class rXOR(Gate, rOperation):
    r"""A gate implementing a logical bitwise XOR operation between two registers.

    :param X: The first register to XOR.
    :param Y: The second register to XOR.
    :param label: An optional label for the gate.
    :raises CircuitError: If X and Y have a different size.

    :operation: :math:`X \leftarrow X \oplus Y`
    """

    def __init__(self, X: QuantumRegister, Y: QuantumRegister, label: Optional[str] = None) -> None:
        if len(X) != len(Y):
            raise CircuitError("rXOR operation must be between two QuantumRegisters of the same size.") 
        self.n = len(X)
        self._inputs = [X, Y]
        self._outputs = [X, Y]
        super().__init__("rXOR", self.n*2, [], label=label)

        qc = QuantumCircuit(X, Y, name='rXOR')

        for i in range(self.n):
            qc.cx(Y[i], X[i]) 

        self.definition = qc

class rConstantDKRSCarryLookaheadAdder(Gate, rOperation):
    r"""A gate implementing an addition modulo :math:`2^n` between a register of qubits and a constant.
    
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
        return rDKRSCarryLookaheadAdder.get_num_ancilla_qubits(n)

    def __init__(self, A: QuantumRegister, c: int, ancillas: AncillaRegister, label: Optional[str] = None) -> None:
        self.n = len(A)
        self.c = c
        self._inputs = [A, ancillas]
        self._outputs = [A, ancillas]
        super().__init__("rADDc", self.n+len(ancillas), [], label=label)

        if len(ancillas) != self.get_num_ancilla_qubits(self.n):
            raise CircuitError(f"Circuit needs {self.get_num_ancilla_qubits(self.n)} ancilla qubits but {len(ancillas)} were given.")

        qc = rCircuit(A, ancillas, name=f'rADD {self.c}')
   
        if self.c != 0:
            bits = _int_to_bits(self.c, self.n)
            Z = AncillaRegister(bits=ancillas[0:self.n-1])
            X = AncillaRegister(bits=ancillas[self.n-1::])
            for i in range(self.n-1): # Z[i] = g[i, i+1]
                if bits[i]:
                    qc.cx(A[i], Z[i])
            qc.xor(A, c) # A[i] = p[i, i+1] and A[0] = s0

            # Lookahead carry
            if self.n>1:
                # Compute carry
                compute_carry = rDKRSComputeCarry(QuantumRegister(bits=A[1:-1]), Z, X)
                qc.append(compute_carry, list(chain(*compute_carry.inputs)))
            
                # Compute sum
                qc.xor(QuantumRegister(bits=A[1:]), Z) # A[i] = si
                # Now do everything in reverse
                qc.neg(QuantumRegister(bits=A[:-1])) # A = s'
                if len(A)>2:
                    qc.xor(QuantumRegister(bits=A[1:-1]), self.c >> 1 & (2**(self.n-2)-1))

                uncompute_carry = rDKRSComputeCarry(QuantumRegister(bits=A[1:-1]), Z, X).reverse_ops()
                qc.append(uncompute_carry, list(chain(*uncompute_carry.inputs)))

                if len(A)>2:
                    qc.xor(QuantumRegister(bits=A[1:-1]), self.c >> 1 & (2**(self.n-2)-1))
                for i in range(self.n-1):
                    if bits[i]:
                        qc.cx(A[i], Z[i])
                qc.neg(QuantumRegister(bits=A[:-1]))

        self.definition = qc

class rDKRSComputeCarry(Gate, rOperation):
    r"""A gate implemented the carry computation described in [DKRS2004]_. The last carry is not computed.

    :param P0: :math:`P_0[i] = p[i, i+1]`, 1 if and only if carry propagages from bit :math:`i` to bit :math:`i+1`.
    :param G: :math:`G[i] = g[i-1, i]`, 1 if and only if a carry is generated between bit :math:`i-1` and bit :math:`i`.
    :param ancillas: The ancilla qubits used for the computation. They must be set to 0 before the circuit and will be reset to 0.
    """

    def __init__(self, P0, G, ancillas: AncillaRegister, label: Optional[str] = None) -> None:
        self.n = len(P0)+1
        super().__init__("rCarry", len(P0)+len(G)+len(ancillas), [], label=label)
        self._inputs = [P0, G, ancillas]
        self._outputs = [P0, G, ancillas]

        qc = rCircuit(P0, G, ancillas, name='rCarry')

        P = [P0]
        ancilla_index=0
        for t in range(1,floor(log2(self.n))):
            Pt_size = floor(self.n/2**t)-1
            P.append(QuantumRegister(bits=ancillas[ancilla_index:ancilla_index+Pt_size]))
            ancilla_index+=Pt_size

        # P-rounds
        for t in range(1, floor(log2(self.n))):
            for m in range(1, floor(self.n/2**t)):
                qc.ccx(P[t-1][2*m-1], P[t-1][2*m], P[t][m-1])
        # G-rounds
        for t in range(1, floor(log2(self.n))+1):
            for m in range(floor(self.n/2**t)):
                qc.ccx(G[2**t*m+2**(t-1)-1],P[t-1][2*m],G[2**t*m+2**t-1])
        # C rounds
        for t in range(floor(log2(2*self.n/3)), 0, -1):
            for m in range(1, floor((self.n-2**(t-1))/2**t)+1):
                qc.ccx(G[2**t*m-1], P[t-1][2*m-1], G[2**t*m+2**(t-1)-1])

        # P^-1 rounds
        for t in range(floor(log2(self.n))-1, 0, -1):
            for m in range(1, floor(self.n/2**t)):
                qc.ccx(P[t-1][2*m-1], P[t-1][2*m], P[t][m-1])

        self.definition = qc

class rDKRSCarryLookaheadAdder(Gate, rOperation):
    r"""A gate implementing the n qubits carry-lookahead adder modulo :math:`2^n` described in [DKRS2004]_. This implementation skips the output carry compution.


    :param A: First register of size n to add.
    :param B: Second register of size n to add.
    :param ancillas: The ancilla qubits used for the computation. They must be set to 0 before the circuit and will be reset to 0.
    :param label: An optional label for the gate.
    :raises CircuitError: If A and B have a different size or if there is not the correct number of ancilla qubits.

    :operation: :math:`X \leftarrow X+Y \bmod 2^n`

    .. [DKRS2004] Draper, T. G., Kutin, S. A., Rains, E. M., & Svore, K. M. (2004). A logarithmic-depth quantum carry-lookahead adder. arXiv preprint quant-ph/0406142.
    """
    @staticmethod
    def get_num_ancilla_qubits(n: int) -> int:
        """Get the number of required ancilla qubits without instantiating the class.

        :param n: The number of qubits of the vectors to add.
        :returns: The number of ancilla qubits needed for the computation.
        """
        if n<2:
            return 0
        return 2*n - _hamming_weight(n-1) - floor(log2(n-1))-2

    def __init__(self, A: QuantumRegister, B: QuantumRegister, ancillas: AncillaRegister, label: Optional[str] = None) -> None:
        self.n = len(A)
        super().__init__("rADD", self.n*2+len(ancillas), [], label=label)
        self._inputs = [A, B, ancillas]
        self._outputs = [A, B, ancillas]
        if len(A) != len(B):
            raise CircuitError("rADD operation must be between two QuantumRegisters of the same size.") 
        qc = rCircuit(A, B, ancillas, name='rADD')

        if len(ancillas) != self.get_num_ancilla_qubits(self.n):
            raise CircuitError("Wrong number of ancilla qubits.")

        Z = AncillaRegister(bits=ancillas[0:self.n-1])
        X = AncillaRegister(bits=ancillas[self.n-1::])
        for i in range(self.n-1): # Z[i] = g[i, i+1]
            qc.ccx(B[i], A[i], Z[i])
        qc.xor(A, B) # A[i] = p[i, i+1] and A[0] = s0

        # Lookahead carry
        if self.n>1:
            # Compute carry
            compute_carry = rDKRSComputeCarry(QuantumRegister(bits=A[1:-1]), Z, X)
            qc.append(compute_carry, list(chain(*compute_carry.inputs)))
        
            # Compute sum
            qc.xor(QuantumRegister(bits=A[1:]), Z) # A[i] = si
            # Now do everything in reverse
            qc.neg(QuantumRegister(bits=A[:-1])) # A = s'
            qc.xor(QuantumRegister(bits=A[1:-1]), QuantumRegister(bits=B[1:-1]))

            uncompute_carry = rDKRSComputeCarry(QuantumRegister(bits=A[1:-1]), Z, X).reverse_ops()
            qc.append(uncompute_carry, list(chain(*uncompute_carry.inputs)))

            qc.xor(QuantumRegister(bits=A[1:-1]), QuantumRegister(bits=B[1:-1]))
            for i in range(self.n-1):
                qc.ccx(B[i], A[i], Z[i])
            qc.neg(QuantumRegister(bits=A[:-1]))

        self.definition = qc

class rTTKRippleCarryAdder(Gate, rOperation):
    r"""A gate implementing the n qubits ripple-carry adder modulo :math:`2^n` described in [TTK2009]_. This implementation skips the output carry compution.


    :param X: First register of size n to add.
    :param Y: Second register of size n to add.
    :param label: An optional label for the gate.
    :raises CircuitError: If X and Y have a different size.

    :operation: :math:`X \leftarrow X+Y \bmod 2^n`

    .. [TTK2009] Takahashi, Y., Tani, S., & Kunihiro, N. (2009). Quantum addition circuits and unbounded fan-out. arXiv preprint arXiv:0910.2530.
    """

    def __init__(self, X: QuantumRegister, Y: QuantumRegister, label: Optional[str] = None) -> None:
        if len(X) != len(Y):
            raise CircuitError("rADD operation must be between two QuantumRegisters of the same size.") 
        self.n = len(X)
        self._inputs = [X, Y]
        self._outputs = [X, Y]
        super().__init__("rADD", self.n*2, [], label=label)

        qc = QuantumCircuit(X, Y, name='rADD')

        for i in range(1, self.n):
            qc.cx(Y[i], X[i]) 
        for i in range(self.n-2, 0, -1):
            qc.cx(Y[i], Y[i+1])     
        for i in range(self.n-1):
            qc.ccx(X[i], Y[i], Y[i+1])
        for i in range(self.n-1, 0, -1):
            qc.cx(Y[i], X[i])
            qc.ccx(X[i-1], Y[i-1], Y[i])
        for i in range(1, self.n-1):
            qc.cx(Y[i], Y[i+1])
        for i in range(self.n):
            qc.cx(Y[i], X[i])

        self.definition = qc

def simulate(circuit: QuantumCircuit) -> Counts:
    """This helper function simulates the given circuit and returns the results.

    :param circuit: The circuit to simulate.
    :returns: The results of the simulation.
    """
    backend_sim = AerSimulator()

    job_sim = backend_sim.run(transpile(circuit, backend_sim), shots=1024)
    result_sim = job_sim.result()
    counts = result_sim.get_counts(circuit)
    return counts

def make_circuit(circuit: QuantumCircuit, inputs: Sequence[int], input_registers: Sequence[QuantumRegister], output_registers: Sequence[QuantumRegister]):
    """This helper function makes a circuit with registers as input and measurement operations for the output registers. It also prepares the initial values of the input registers.

    :param circuit: The base circuit that will be expanded with measurement operations and preparation operations.
    :param inputs: A list of the initial values to assign to the input registers.
    :param inputs_registers: A list of the input registers.
    :param output_registers: A list of the output registers.
    :returns: The final circuit containing the preparation step, the base circuit and the measurement step.
    """
    # Copy the circuit
    qc = circuit.copy()
    qc.compose(Barrier(qc.num_qubits), inplace=True, front=True)
    # Add classical registers for the output registers
    classical_output_registers = []
    for output_register in output_registers:
        classical_output_register = ClassicalRegister(len(output_register))
        qc.add_register(classical_output_register)
        classical_output_registers.append(classical_output_register)
    # Prepare the input by initializing it to given values
    for i, input_register in enumerate(input_registers):
        qc.compose(rPrepare(input_register, inputs[i]), input_register, inplace=True, front=True)
    qc.barrier()
    # Add measurements for the output
    for output_register, classical_output_register in zip(output_registers, classical_output_registers):
        qc.measure(output_register, classical_output_register)

    return qc

def run_circuit(circuit: QuantumCircuit, verbose: bool = False) -> Sequence[int]:
    """This helper function simulates a given circuit and retrieves the integer values of the classical registers.

    :param circuit: The circuit to run.
    :param verbose: Print the raw result given by the simulator.
    :returns: A list of the integers stored in the classical registers of the circuit after the circuit has been simulated. It takes into account only the most frequent result.
    """
    # Simulate the circuit
    raw_result = simulate(circuit)
    if verbose:
        print("Circuit result:", raw_result)
    result = raw_result.most_frequent()
    # Extract the output registers (result is MSB-first so we need to reverse the register order)
    outputs = result.split()[::-1]
    # Convert the outputs to integers
    return [int(output, 2) for output in outputs]
