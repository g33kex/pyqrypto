from bitwise_operations import bOperation, bROR, bADD, bXOR, bXORc, bAppend, make_circuit, run_circuit
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.exceptions import CircuitError
import json

def ror(x, r, n):
    """Classical ror"""
    r = r % n
    return ((x >> r) | (x << (n - r))) & ((2**n) - 1)

def alzette(x, y, c, n):
    """Classical implementation of Alzette for reference"""
    x = (x + ror(y, 31, n)) % 2**n
    y = y ^ ror(x, 24, n)
    x = x ^ c
    x = (x + ror(y, 17, n)) % 2**n
    y = y ^ ror(x, 17, n)
    x = x ^ c
    x = (x + y) % 2**n
    y = y ^ ror(x, 31, n)
    x = x ^ c
    x = (x + ror(y, 24, n)) % 2**n
    y = y ^ ror(x, 16, n)
    x = x ^ c
    return [x, y]

class Alzette(QuantumCircuit, bOperation):
    """A reversible quantum circuit implementing n bit Alzette"""
    def __init__(self, X: QuantumRegister, Y: QuantumRegister, c: int):
        if len(X) != len(Y):
            raise CircuitError("X and Y must have the same size.")
        num_qubits = len(X)

        self.inputs = [X, Y]
        circuit = QuantumCircuit(X, Y, name='Alzette')
        X = bAppend(circuit, bADD(X, bAppend(circuit, bROR(Y, 31))))
        Y = bAppend(circuit, bXOR(Y, bAppend(circuit, bROR(X, 24))))
        X = bAppend(circuit, bXORc(X, c)) 
        X = bAppend(circuit, bADD(X, bAppend(circuit, bROR(Y, 17))))
        Y = bAppend(circuit, bXOR(Y, bAppend(circuit, bROR(X, 17))))
        X = bAppend(circuit, bXORc(X, c)) 
        X = bAppend(circuit, bADD(X, Y))
        Y = bAppend(circuit, bXOR(Y, bAppend(circuit, bROR(X, 31))))
        X = bAppend(circuit, bXORc(X, c)) 
        X = bAppend(circuit, bADD(X, bAppend(circuit, bROR(Y, 24))))
        Y = bAppend(circuit, bXOR(Y, bAppend(circuit, bROR(X, 16))))
        X = bAppend(circuit, bXORc(X, c)) 

        super().__init__(num_qubits*2, name='Alzette')
        self.compose(circuit.to_instruction(), qubits=self.qubits, inplace=True)
        self.outputs = [X, Y]


if __name__ == '__main__':

    n = 32
    X = QuantumRegister(32, name='X')
    Y = QuantumRegister(32, name='Y')
    a = 384973
    b = 1238444859
    c = 0xb7e15162
    print(f"Running alzette_{c}({a}, {b})")


    qc = QuantumCircuit(X, Y)
    X, Y = bAppend(qc, Alzette(X, Y, c))

    decomposition_reps = 4
    circuit_depth = qc.decompose(reps=decomposition_reps).depth()
    gate_counts = qc.decompose(reps=decomposition_reps).count_ops()

    final_circuit = make_circuit(qc, [a, b], [X, Y], [X, Y])

    final_circuit.decompose(reps=4).qasm(filename='alzette.qasm')

    result = run_circuit(final_circuit)

    print(f"Classical result: {alzette(a, b, c, n)}")
    print(f"Quantum simulated result: {result}")
    print("Circuit depth:", circuit_depth)
    print("Circuit gate count:", json.dumps(gate_counts))
