"""A library of vectorial operations that can be applied to quantum registers."""
from __future__ import annotations
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, transpile
from qiskit.circuit import Operation, CircuitInstruction, Gate, Register
from qiskit.circuit.exceptions import CircuitError
import itertools
from typing import Optional, List, Dict


def _int_to_bits(k: int, num_bits):
    """Convert integer k into list of bits.

    :param k: The integer to convert to list of bits.
    :param num_bits: The numbers of bits to use to encode k.
    """
    return [int(i) for i in '{:0{n}b}'.format(k, n=num_bits)]

class rCircuit(QuantumCircuit):
    """A wrapper for ``QuantumCircuit`` that implements the logic needed to chain operations on qubit vectors. When adding a new rOperation to the circuit it takes care of connecting the right outputs to the right inputs"""
    def __init__(self, *regs: Register, name: Optional[str] = None):
        super().__init__(*regs, name=name)

class rOperation():
    """Abstract class defining an operation on vectors of qubits.
    
    :ivar inputs: The inputs registers of the operation.
    :type inputs: List[QuantumRegister]
    :ivar outputs: The outputs registers of the operation.
    :type outputs: List[QuantumRegister]
    """

    # def __xor__(self, other: rOperation):
    #     if isinstance(other, Register):
    #         if isinstance(self, Register):
    #             return rXOR(self, other)
    #         else:
    #             return rXOR(*self.outputs, other)
    #     else:
    #         return rXOR(*self.outputs, *other.inputs)

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
        self.inputs = [X]
        self.outputs = [X]

class rROR(rOperation):
    r"""Defines the right rotation operation on a QuantumRegiser.

    :param X: The register to rotate.
    :param r: The number of qubits by which X should be rotated.

    :operation: :math:`X' \leftarrow \mathrm{ror}(X, r)`

    .. warning::
        The result will be stored in vector ``self.outputs[0]``.
    """

    def __init__(self, X: QuantumRegister, r: int) -> None:
        if r < 0:
            raise CircuitError("Rotation must be by a positive amount.")
        r = r % len(X)
        self.inputs = [X]
        self.outputs = [QuantumRegister(bits=X[r:]+X[:r], name=X.name)]

class rROL(rOperation):
    r"""Defines the left rotation operation on a QuantumRegister.

    :param X: The register to rotate.
    :param r: The number of qubits by which X should be rotated.
    :raises CircuitError: If r is negative.

    :operation: :math:`X' \leftarrow \mathrm{rol}(X, r)`


    .. warning::
        The result will be stored in vector ``self.outputs[0]``.
    """

    def __init__(self, X: QuantumRegister, r: int) -> None:
        if r < 0:
            raise CircuitError("Rotation must be by a positive amount.")
        r = r % len(X)
        self.inputs = [X]
        self.outputs = [QuantumRegister(bits=X[-r:]+X[:-r], name=X.name)]

class rXORc(Gate, rOperation):
    r"""A gate implementing a logical bitwise XOR operation between a vector of qubits and a constant.
    
    :param X: The vector to XOR with c.
    :param c: The constant to XOR with X.
    :param label: An optional label for the gate.

    :operation: :math:`X \leftarrow X \oplus c`
    """

    def __init__(self, X: QuantumRegister, c: int, label: Optional[str] = None) -> None:
        self.n = len(X)
        self.c = c
        self.inputs = [X]
        self.outputs = [X]
        super().__init__("rXORc", self.n, [], label=label)

        bits = _int_to_bits(self.c, self.n)[::-1]
    
        qc = QuantumCircuit(self.n, name=f'rXOR {self.c}')
   
        for i, bit in enumerate(bits):
             if bit:
                 qc.x(i)
        
        self.definition = qc

class rXOR(Gate, rOperation):
    r"""A gate implementing a logical bitwise XOR operation between two vectors of qubits.

    :param X: The first vector to XOR.
    :param Y: The second vector to XOR.
    :param label: An optional label for the gate.

    :operation: :math:`X \leftarrow X \oplus Y`
    """

    def __init__(self, X: QuantumRegister, Y: QuantumRegister, label: Optional[str] = None) -> None:
        if len(X) != len(Y):
            raise CircuitError("rXOR operation must be between two QuantumRegisters of the same size.") 
        self.n = len(X)
        self.inputs = [X, Y]
        self.outputs = [X]
        super().__init__("rXOR", self.n*2, [], label=label)

        qc = QuantumCircuit(self.n*2, name='rXOR')

        for i in range(self.n):
            qc.cx(self.n+i, i) 

        self.definition = qc

class rADD(Gate, rOperation):
    r"""A gate implementing the ripple-carry adder described in [TTK2009]_.


    :param X: First vector to add.
    :param Y: Second vector to add.
    :param label: An optional label for the gate.

    :operation: :math:`X \leftarrow X+Y`

    .. [TTK2009] Takahashi, Y., Tani, S., & Kunihiro, N. (2009). Quantum addition circuits and unbounded fan-out. arXiv preprint arXiv:0910.2530.
    """

    def __init__(self, X: QuantumRegister, Y: QuantumRegister, label: Optional[str] = None) -> None:
        if len(X) != len(Y):
            raise CircuitError("rADD operation must be between two QuantumRegisters of the same size.") 
        self.n = len(X)
        self.inputs = [X, Y]
        self.outputs = [X]
        super().__init__("rADD", self.n*2, [], label=label)

        qc = QuantumCircuit(self.n*2, name='rADD')

        for i in range(1, self.n):
            qc.cx(self.n+i, i) 
        for i in range(self.n-2, 0, -1):
            qc.cx(self.n+i, self.n+i+1)     
        for i in range(self.n-1):
            qc.ccx(i, self.n+i, self.n+i+1)
        for i in range(self.n-1, 0, -1):
            qc.cx(self.n+i, i)
            qc.ccx(i-1, self.n+i-1, self.n+i)
        for i in range(1, self.n-1):
            qc.cx(self.n+i, self.n+i+1)
        for i in range(self.n):
            qc.cx(self.n+i, i)

        self.definition = qc

def rAppend(circuit: QuantumCircuit, operation: rOperation) -> QuantumRegister | List[QuantumRegister]:
    """Append a register operation to a quantum circuit and wire it according to the input registers of the operation.

    :param circuit: Quantum circuit to add the operation to.
    :param operation: The operation to add to the circuit.
    :returns: The new output registers of the circuit. This is a list if there are multiple output registers and a single QuantumRegister if there is only one."""
    if isinstance(operation, (QuantumCircuit, Operation, CircuitInstruction)):
        circuit.append(operation, itertools.chain(*operation.inputs))
    return operation.outputs if len(operation.outputs)>1 else operation.outputs[0] if len(operation.outputs)>0 else None

def simulate(circuit: QuantumCircuit) -> Dict[int, str]:
    """This helper function simulates the given circuit and returns the results.

    :param circuit: The circuit to simulate.
    :returns: The results of the simulation.
    """
    backend_sim = Aer.get_backend('qasm_simulator')
    job_sim = backend_sim.run(transpile(circuit, backend_sim), shots=1024)
    result_sim = job_sim.result()
    counts = result_sim.get_counts(circuit)
    return counts

def make_circuit(circuit: QuantumCircuit, inputs: List[int], input_registers: List[QuantumRegister], output_registers: List[QuantumRegister]):
    """This helper function makes a circuit with registers as input and measurement operations for the output registers. It also prepares the initial values of the input registers.

    :param circuit: The base circuit that will be expanded with measurement operations and preparation operations.
    :param inputs: A list of the initial values to assign to the input registers.
    :param inputs_registers: A list of the input registers.
    :param output_registers: A list of the output registers.
    :returns: The final circuit containing the preparation step, the base circuit and the measurement step.
    """
    # Copy the circuit
    qc = circuit.copy()
    # Add classical registers for the output vectors
    classical_output_registers = []
    for output_register in output_registers:
        classical_output_register = ClassicalRegister(len(output_register))
        qc.add_register(classical_output_register)
        classical_output_registers.append(classical_output_register)
    # Prepare the input by initializing it to given values
    for i, input_register in enumerate(input_registers):
        qc.compose(rPrepare(input_register, inputs[i]), input_register, inplace=True, front=True)
    # Add measurements for the output
    for output_register, classical_output_register in zip(output_registers, classical_output_registers):
        qc.measure(output_register, classical_output_register)

    return qc

def run_circuit(circuit: QuantumCircuit, verbose: bool = False) -> List[int]:
    """This helper function simulates a given circuit and retrives the integer values of the classical registers.

    :param circuit: The circuit to run.
    :param verbose: Print the raw result given by the simulator.
    :returns: A list of the integers stored in the classical registers of the circuit after the circuit has been simulated. It takes into account only the most frequent result.
    """
    # Simulate the circuit
    raw_result = simulate(circuit)
    if verbose:
        print("Circuit result:", raw_result)
    result = raw_result.most_frequent()
    # Extract the output vectors (result is in big endian so we need to the vector order)
    outputs = result.split()[::-1]
    # Convert the outputs to integers
    return [int(output, 2) for output in outputs]
