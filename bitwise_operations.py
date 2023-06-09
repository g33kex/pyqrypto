from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, transpile
from qiskit.circuit.exceptions import CircuitError
import itertools

class bPrepare(QuantumCircuit):
    """A quantum circuit preparing a Qubit register to a classical integer value"""
        
    def __init__(self, num_qubits: int, value: int):
        circuit = QuantumCircuit(num_qubits, name='prepare {:0{n}b}'.format(value, n=num_qubits))
        bits = [int(i) for i in '{:0{n}b}'.format(value, n=num_qubits)][::-1]
        for i in range(num_qubits):
            if bits[i]:
                circuit.x(i)
            else:
                circuit.i(i)

        super().__init__(*circuit.qregs, name=f'prepare {value}')
        self.compose(circuit.to_gate(), qubits=self.qubits, inplace=True)

def ROR(register: QuantumRegister , r: int) -> QuantumRegister:
    """Returns the right rotation by r qubits of a quantum register."""
    return QuantumRegister(bits=register[r:]+register[:r], name=register.name)

def ROL(register: QuantumRegister, r: int) -> QuantumRegister:
    """Returns the left rotation by r qubits of a quantum register."""
    return QuantumRegister(bits=register[-r:]+register[:-r], name=register.name)

class bXOR(QuantumCircuit):
    """A circuit implementing a logical bitwise XOR operation on a number of qubits."""

    def __init__(self, num_qubits):
        circuit = QuantumCircuit(num_qubits*2, name='bXOR')

        for i in range(num_qubits):
            circuit.cx(num_qubits+i, i) 

        super().__init__(num_qubits*2, name='bXOR')
        self.compose(circuit.to_gate(), qubits=self.qubits, inplace=True)
    
def bAppend(circuit: QuantumCircuit, operation, registers):
    """Append a bitwise operation to a Quantum Circuit operating on specified registers."""
    return circuit.append(operation, itertools.chain(*registers))

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
    # Add classical register for the output vectors
    qc.add_register(ClassicalRegister(sum(map(len, output_registers))))
    # Prepare the input by initializing it manually because the initialization function of Qiskit seems to have a bug where specifying integers for register values yields to 0 output in the simulator (but the QASM is correct) TODO: investigate this bug further
    for i, input_register in enumerate(input_registers):
        qc.compose(bPrepare(len(input_register), inputs[i]), input_register, inplace=True, front=True)
    # for input_register in input_registers:
    #     for qubit in input_register:
    #         qc.reset(qubit)
    # for i, input_register in enumerate(input_registers):
    #     bits = [int(i) for i in '{:0{n}b}'.format(inputs[i], n=len(input_register))][::-1]
    #     for i, qubit in enumerate(input_register):
    #         if bits[i]:
    #             qc.h(qubit)
    #         else:
    #             qc.i(qubit)
    # for input_register in input_registers:
    #     for qubit in input_register:
    #         qc.barrier(qubit)
    # Add measurements for the output
    output_index = 0
    for output_register in output_registers:
        qc.measure(output_register, range(output_index, output_index+len(output_register)))
        output_index += len(output_register)
    # Add metadata to mark output registers
    qc.metadata['output_registers'] = output_registers
    return qc

def run_circuit(circuit):
    """Run circuit returns the integer value of the output registers. Output registers must be stored in the circuit metadata under the 'output_registers' key."""
    # Simulate the circuit
    result = simulate(circuit).most_frequent()
    # Extract the output vectors (result is in big endian so we need to the vector order)
    outputs = []
    output_index = 0
    for output_register in circuit.metadata['output_registers']:
        outputs = [(result[output_index:output_index+len(output_register)])] + outputs
        output_index += len(output_register)
    # Convert the outputs to integers
    return [int(output, 2) for output in outputs]
