from __future__ import annotations
from qiskit import QuantumCircuit

class QubitVector:
    """This is a helper class to make it easier to work with vectors of Qubits, ensuring that rotations are gate free operations. It is compatible with the Qiskit library."""

    def __init__(self, qubits: list, initial_value=None):
        """Create a qubit vector from specified qubits

        :param qubits: a list of the qubits to use for the vector, the first qubit in the list is the LSB
        :param initial_value: a list of the initial values of the qubits in the QubitVector (used to prepare the vector)
        """
        self.n = len(qubits)
        self.qubits = list(qubits)
        if initial_value is not None:
            self.initial_value = initial_value

    def __getitem__(self, i):
        return self.qubits[i]

    def __iter__(self):
        return iter(self.qubits)

    def __len__(self):
        return self.n

    def __str__(self):
        return str(self.qubits)

    def prepare(self):
        """Returns a circuit that prepares the QubitVector to its initial value. QubitVector are prepared in little endian."""
        qc = QuantumCircuit(self.n, name=f'prepare {self.initial_value}')
        # TODO: ignores rotation, maybe take the current circuit as parameter of __init__ and add to that circuit? Or we'd have to add to the circuit according to list(self)
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
        return QubitVector(self.qubits[r:]+self.qubits[:r])

    def ROL(self, r):
        """Returns the left rotation by r qubits of the QubitVector"""
        if r < 0:
            raise ValueError("Rotation must be by a positive amount.")
        r = r % self.n
        return QubitVector(self.qubits[-r:]+self.qubits[:-r])

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

