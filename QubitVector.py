from __future__ import annotations
from qiskit import QuantumCircuit, ClassicalRegister, Aer, transpile

class QubitVector:
    """This is a helper class to make it easier to work with vectors of Qubits, ensuring that rotations are gate free operations. It is compatible with the Qiskit library."""

    def __init__(self, qubits: list, initial_value: int=0):
        """Create a qubit vector from specified qubits

        :param qubits: a list of the qubits to use for the vector, the first qubit in the list is the LSB
        :param circuit: the circuit in which this QubitVector lives
        :param initial_value: an integer representing the initial value of the qubitvector
        """
        # The length of the QubitVector
        self.n = len(qubits)
        # The initial positions of the qubits
        self.initial_qubits = list(qubits)
        # The current positions of the qubits
        self.qubits = list(qubits)
        if initial_value is not None:
            self.initial_value = initial_value

    def __getitem__(self, i):
        return self.qubits[i]

    def __len__(self):
        return self.n

    def __str__(self):
        return str(self.qubits)

    def prepare(self):
        """Returns a circuit that prepares the QubitVector to its initial value. QubitVector are prepared in little endian."""
        qc = QuantumCircuit(self.n, name=f'prepare {self.initial_value}')
        bits = [int(i) for i in '{:0{n}b}'.format(self.initial_value, n=self.n)][::-1]
        for i in range(self.n):
            qc.reset(i)
        for i in range(self.n):
            if bits[i]:
                qc.x(i)
            else:
                qc.i(i)
        prepare = qc.to_instruction()
        return prepare
    
    def ROR(self, r):
        """Returns the right rotation by r qubits of the QubitVector"""
        if r < 0:
            raise ValueError("Rotation must be by a positive amount.")
        r = r % self.n
        
        self.qubits = self.qubits[r:]+self.qubits[:r]

    def ROL(self, r):
        """Returns the left rotation by r qubits of the QubitVector"""
        if r < 0:
            raise ValueError("Rotation must be by a positive amount.")
        r = r % self.n
        self.qubits = self.qubits[-r:]+self.qubits[:-r]

    # TODO: Is there a way this could returna  qubit vector, and this vector could be added to the circuit as a circuit part? Or directly add to the whole circuit. Cause there is a problem if we have more than two qubitvectors in the circuit right now...
    def XOR(self, other: QubitVector):
        """Returns a gate that performs a XOR between QubitVector and another QubitVector, storing the result in this QubitVector and leaving as is the other QubitVector"""
        if len(self) != len(other):
            raise ValueError(f"Trying to XOR two QubitVectors of different size ({len(self)} != {len(other)})")

        qc = QuantumCircuit(self.n*2, name=f'XOR')

        for i in range(self.n):
            qc.cx(other[i], self[i])

        xor = qc.to_instruction()
        return xor

def _simulate(qc):
    """Simulate the given circuit and return the results"""
    backend_sim = Aer.get_backend('qasm_simulator')
    job_sim = backend_sim.run(transpile(qc, backend_sim), shots=1024)
    result_sim = job_sim.result()
    counts = result_sim.get_counts(qc)
    return counts

def make_circuit(qc, input_vectors, output_vectors):
    """Make a circuit with vectors as input and measurement for the output vectors"""
    # Copy the circuit to avoid modifying it
    qc = qc.copy()
    # Add classical register for the output vectors
    qc.add_register(ClassicalRegister(sum(map(len, output_vectors))))
    # Prepare the input
    for input_vector in input_vectors:
        qc.compose(input_vector.prepare(), input_vector.initial_qubits, inplace=True, front=True)
    output_index = 0
    # Add measurements for the output
    for output_vector in output_vectors:
        qc.measure(output_vector, range(output_index, output_index+len(output_vector)))
        output_index += len(output_vector)
    return qc


def run_circuit(qc, output_vectors):
    """Run circuit returns the integer value of the output vectors"""
    # Make the circuit and simulate it
    result = _simulate(qc).most_frequent()
    # Extract the output vectors (result is in big endian so we need to the vector order)
    outputs = []
    output_index = 0
    for output_vector in output_vectors:
        outputs = [(result[output_index:output_index+len(output_vector)])] + outputs
        output_index += len(output_vector)
    # Convert the outputs to integers
    return [int(output, 2) for output in outputs]
