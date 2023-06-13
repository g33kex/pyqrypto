from __future__ import annotations
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister, Aer, transpile
from qiskit.circuit import Operation, CircuitInstruction, Register
from qiskit.circuit.exceptions import CircuitError
import itertools
import abc

class bOperation():
    """Abstract class defining an operation on vectors of qubits.
    
    :ivar inputs: The inputs registers of the operation
    :type inputs: QuantumRegister
    :ivar outputs: The outputs registers of the operation
    :type outputs: QuantumRegister
    """

    # def __xor__(self, other: bOperation):
    #     if isinstance(other, Register):
    #         if isinstance(self, Register):
    #             return bXOR(self, other)
    #         else:
    #             return bXOR(*self.outputs, other)
    #     else:
    #         return bXOR(*self.outputs, *other.inputs)

class bRegister(QuantumRegister, bOperation):

    def __init__(self, n: int, name=None):
        self.operations = []
        super().__init__(n, name)

class bPrepare(QuantumCircuit, bOperation):
    """A quantum circuit preparing a Qubit register to a classical integer value"""
        
    def __init__(self, X: QuantumRegister, value: int):
        num_qubits = len(X)
        circuit = QuantumCircuit(num_qubits, name='prepare {:0{n}b}'.format(value, n=num_qubits))
        bits = [int(i) for i in '{:0{n}b}'.format(value, n=num_qubits)][::-1]
        for i in range(num_qubits):
            if bits[i]:
                circuit.x(i)
            else:
                circuit.i(i)

        super().__init__(*circuit.qregs, name=f'prepare {value}')
        self.compose(circuit.to_gate(), qubits=self.qubits, inplace=True)
        self.inputs = [X]
        self.outputs = [X]

class bROR(bOperation):

    def __init__(self, X: QuantumRegister , r: int):
        """Performs the right rotation by r qubits of a quantum register."""
        if r < 0:
            raise CircuitError("Rotation must be by a positive amount.")
        r = r % len(X)
        self.inputs = [X]
        self.outputs = [QuantumRegister(bits=X[r:]+X[:r], name=X.name)]

class bROL(bOperation):

    def __init__(self, X: QuantumRegister, r: int):
        """Returns the left rotation by r qubits of a quantum register."""
        if r < 0:
            raise CircuitError("Rotation must be by a positive amount.")
        r = r % len(X)
        self.inputs = [X]
        self.outputs = [QuantumRegister(bits=X[-r:]+X[:-r], name=X.name)]

class bXOR(QuantumCircuit, bOperation):
    """A circuit implementing a logical bitwise XOR operation on a number of qubits."""

    def __init__(self, X, Y):
        if len(X) != len(Y):
            raise CircuitError("bXOR operation must be between two QuantumRegisters of the same size.") 
        num_qubits = len(X)
        circuit = QuantumCircuit(num_qubits*2, name='bXOR')

        for i in range(num_qubits):
            circuit.cx(num_qubits+i, i) 

        super().__init__(num_qubits*2, name='bXOR')
        self.compose(circuit.to_gate(), qubits=self.qubits, inplace=True)

        self.inputs = [X, Y]
        self.outputs = [X]

class bAND(QuantumCircuit, bOperation):
    """A circuit implementing a logical bitwise AND operation on a number of qubits. It needs one ancilla register with one Qubit."""

    def __init__(self, A: AncillaRegister, X: QuantumRegister, Y: QuantumRegister):
        if len(X) != len(Y):
            raise CircuitError("AND operation must be made on two QuantumRegisters of the same size.")
        if len(A) != 1:
            raise CircuitError("AND operation needs 1 ancilla Qubit.")

        num_qubits = len(X)
        circuit = QuantumCircuit(num_qubits*2+1, name='bAND')

        for i in range(num_qubits):
            circuit.reset(i)
            circuit.ccx(i+1, num_qubits+i+1, i)

        super().__init__(A, X, Y, name='bAND')
        self.compose(circuit.to_instruction(), qubits=self.qubits, inplace=True)
        self.inputs = [A, X, Y]
        self.outputs = [QuantumRegister(bits=[A[0]]+X[:-1], name=X.name), QuantumRegister(bits=[X[-1]], name=A.name)]

def bAppend(circuit: QuantumCircuit, operation: bOperation):
    """Append a bitwise operation to a Quantum Circuit operating on input registers."""
    if isinstance(operation, (QuantumCircuit, Operation, CircuitInstruction)):
        circuit.append(operation, itertools.chain(*operation.inputs))
    return operation.outputs if len(operation.outputs)>1 else operation.outputs[0] if len(operation.outputs)>0 else None

def simulate(circuit):
    """Simulate the given circuit and return the results"""
    backend_sim = Aer.get_backend('qasm_simulator')
    job_sim = backend_sim.run(transpile(circuit, backend_sim), shots=1024)
    result_sim = job_sim.result()
    counts = result_sim.get_counts(circuit)
    return counts

def make_circuit(circuit, inputs, input_registers, output_registers):
    """Make a circuit with registers as input and measurement for the output registers. It also sets the initial values of the input registers and sets the output registers as circuit metadata."""
    # Copy the circuit
    qc = circuit.copy()
    # Add classical registers for the output vectors
    classical_output_registers = []
    for output_register in output_registers:
        classical_output_register = ClassicalRegister(len(output_register))
        qc.add_register(classical_output_register)
        classical_output_registers.append(classical_output_register)
    # Prepare the input by initializing it manually because the initialization function of Qiskit seems to have a bug where specifying integers for register values yields to 0 output in the simulator (but the QASM is correct) TODO: investigate this bug further
    for i, input_register in enumerate(input_registers):
        qc.compose(bPrepare(input_register, inputs[i]), input_register, inplace=True, front=True)
    # Add measurements for the output
    for output_register, classical_output_register in zip(output_registers, classical_output_registers):
        qc.measure(output_register, classical_output_register)
    return qc

def run_circuit(circuit, verbose=False):
    """Run circuit returns the integer value of the output registers. Output registers must be stored in the circuit metadata under the 'output_registers' key."""
    # Simulate the circuit
    result = simulate(circuit).most_frequent()
    if verbose:
        print("Circuit result:", result)
    # Extract the output vectors (result is in big endian so we need to the vector order)
    outputs = result.split()[::-1]
    # output_index = 0
    # for output_register in circuit.metadata['output_registers']:
    #     outputs = [result[output_index:output_index+len(output_register)]] + outputs
    #     output_index += len(output_register)
    if verbose:
        print("Outputs:", outputs)
    # Convert the outputs to integers
    return [int(output, 2) for output in outputs]
