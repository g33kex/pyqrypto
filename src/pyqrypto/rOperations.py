"""A library of vectorial operations that can be applied to quantum registers.

In this library, a :class:`QuantumRegister` from `Qiskit <https://qiskit.org/>`_ is seen as a view on a vector of qubits. Operations can be applied directly between registers, for example a bitwise XOR or an addition. This is similar to how registers work on classical computers. Rotations, and permutations in general, can be made without using any quantum gate because they just return a new view on the qubits, i.e. a new :class:`QuantumRegister` that has the same logical qubits but in a different order.

The aim of this library is to be able to implement classical algorithms on quantum computers, to benefit from the speedup given by methods such as Grover's algorithm.
"""
from __future__ import annotations
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.quantumcircuit import QubitSpecifier, ClbitSpecifier, Operation, CircuitInstruction
from qiskit_aer.backends import AerSimulator
from qiskit.circuit import Gate
from qiskit.circuit.library import Barrier
from qiskit.result.counts import Counts
from qiskit.circuit.exceptions import CircuitError
from itertools import chain
from typing import Optional, Sequence, Final
from abc import ABC

FEYNMAN_QC: Final[int] = 1
"""The quantum cost of the Feynman gate (CNOT)."""
TOFFOLI_QC: Final[int] = 5
"""The quantum cost of the Toffoli gate."""
NOT_QC: Final[int] = 1
"""The quantum cost of a NOT gate."""

def _int_to_bits(k: int, num_bits):
    """Convert integer k into list of bits.

    :param k: The integer to convert to list of bits.
    :param num_bits: The numbers of bits to use to encode k.
    """
    return [int(i) for i in '{:0{n}b}'.format(k, n=num_bits)]

class rCircuit(QuantumCircuit):
    """A wrapper around :class:`QuantumCircuit` that implements the logic needed to chain operations on qubit vectors. It supports new operations that operate on whole quantum registers and handles rotations without using any gate by rewiring the circuit when needed. This being also a fully valid :class:`QuantumCircuit`, it is also possible to apply operations on single qubits as it is normally done in Qiskit.

    :param inputs: The quantum registers to use as the circuit inputs.
    :param kwargs: Other parameters to pass to the underlying :class:`QuantumCircuit` object.
    """
    def __init__(self, *inputs: QuantumRegister, **kwargs):
        super().__init__(*inputs, **kwargs)
        self._quantum_cost = 0

    @property
    def quantum_cost(self):
        return self._quantum_cost

    def append(self, instruction: Operation | CircuitInstruction, qargs: Optional[Sequence[QubitSpecifier]] = None, cargs: Optional[Sequence[ClbitSpecifier]] = None):
        """Append one or more instructions to the end of the circuit, modifying the circuit in
        place. Also updates the quantum cost of the circuit if the instruction is a :class:`rOperation`.

        :param instruction: The instruction to append.
        :param qargs: Qubits to attach instruction to.
        :param cargs: Classical bits to attach instruction to.
        :returns: The instruction that was added to the circuit.
        :raises CircuitError: If the operation passed is not an instance of :class:`Instruction`
        """
        super().append(instruction, qargs, cargs)
        if isinstance(instruction, rOperation):
            self._quantum_cost += instruction.quantum_cost

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
            op = rXORc(X, Y)
            self.append(op, list(X))
        return op.outputs[0]

    def add(self, X: QuantumRegister, Y: QuantumRegister) -> QuantumRegister:
        r"""Add two registers of size n and store the result in the first register.

        :param X: First register to add (the result will be stored in this register, overwriting its previous value).
        :param Y: Second register to add.
        :returns: The output register :math:`X`.

        :operation: :math:`X \leftarrow X+Y \bmod 2^n`
        """
        op = rADD(X, Y)
        self.append(op, list(chain(X, Y)))
        return op.outputs[0]

class rOperation(ABC):
    """Abstract class defining an operation on registers of qubits.
    
    :ivar inputs: The inputs registers of the operation.
    :type inputs: Sequence[QuantumRegister]
    :ivar outputs: The outputs registers of the operation.
    :type outputs: Sequence[QuantumRegister]
    :ivar quantum_cost: The quantum cost of the operation as defined in [FoM2009]_.
    :type quantum_cost: int

    .. [FoM2009] Mohammadi, M., & Eshghi, M. (2009). On figures of merit in reversible and quantum logic designs. Quantum Information Processing, 8, 297-318.
    """
    def __init__(self):
        self._inputs = []
        self._outputs = []
        self._quantum_cost = 0

    @property
    def outputs(self) -> list[QuantumRegister]:
        return self._outputs

    @property
    def inputs(self) -> list[QuantumRegister]:
        return self._inputs

    @property
    def quantum_cost(self) -> int:
        return self._quantum_cost

class rPrepare(QuantumCircuit, rOperation):
    r"""A circuit preparing a QuantumRegister to an initial classical integer value.

    :param X: The register to prepare.
    :param value: The value to prepare the register to.

    :operation: :math:`X \leftarrow \mathrm{value}`
    """
        
    def __init__(self, X: QuantumRegister, value: int) -> None:
        num_qubits = len(X)
        qc = QuantumCircuit(num_qubits, name=f'rPrepare {value}')
        bits = _int_to_bits(value, num_qubits)[::-1]
        for i in range(num_qubits):
            if bits[i]:
                qc.x(i)
            else:
                qc.i(i)

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
        self._quantum_cost = 0

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
        self._quantum_cost = 0

class rXORc(Gate, rOperation):
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

        bits = _int_to_bits(self.c, self.n)[::-1]
    
        qc = QuantumCircuit(self.n, name=f'rXOR {self.c}')
   
        self._quantum_cost = 0
        for i, bit in enumerate(bits):
            if bit:
                qc.x(i)
                self._quantum_cost += NOT_QC
        
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
        self._outputs = [X]
        super().__init__("rXOR", self.n*2, [], label=label)

        qc = QuantumCircuit(self.n*2, name='rXOR')

        self._quantum_cost = 0
        for i in range(self.n):
            qc.cx(self.n+i, i) 
            self._quantum_cost += FEYNMAN_QC

        self.definition = qc

class rADD(Gate, rOperation):
    r"""A gate implementing the n qubits ripple-carry adder modulo :math:`2^n` described in [TTK2009]_. This implementation skips the output carry compution.


    :param X: First register to add.
    :param Y: Second register to add.
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
        self._outputs = [X]
        super().__init__("rADD", self.n*2, [], label=label)

        qc = QuantumCircuit(self.n*2, name='rADD')

        self._quantum_cost = 0
        for i in range(1, self.n):
            qc.cx(self.n+i, i) 
            self._quantum_cost += FEYNMAN_QC
        for i in range(self.n-2, 0, -1):
            qc.cx(self.n+i, self.n+i+1)     
            self._quantum_cost += FEYNMAN_QC
        for i in range(self.n-1):
            qc.ccx(i, self.n+i, self.n+i+1)
            self._quantum_cost += TOFFOLI_QC
        for i in range(self.n-1, 0, -1):
            qc.cx(self.n+i, i)
            qc.ccx(i-1, self.n+i-1, self.n+i)
            self._quantum_cost += TOFFOLI_QC + FEYNMAN_QC
        for i in range(1, self.n-1):
            qc.cx(self.n+i, self.n+i+1)
            self._quantum_cost += FEYNMAN_QC
        for i in range(self.n):
            qc.cx(self.n+i, i)
            self._quantum_cost += FEYNMAN_QC

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
    # Extract the output registers (result is in big endian so we need to reverse the register order)
    outputs = result.split()[::-1]
    # Convert the outputs to integers
    return [int(output, 2) for output in outputs]
