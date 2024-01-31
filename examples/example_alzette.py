from qiskit import QuantumRegister
from pyqrypto.sparkle import Alzette, c_alzette
from pyqrypto.rOperations import rCircuit, make_circuit, run_circuit
from itertools import chain
import json

n = 32
X = QuantumRegister(n, name='X')
Y = QuantumRegister(n, name='Y')
a = 384973
b = 1238444859
c = 0xb7e15162
print(f"Running alzette_{c}({a}, {b})")


qc = rCircuit(X, Y)
gate = Alzette(X, Y, c)
qc.append(gate, list(chain(*gate.inputs)))

decomposition_reps = 2
qc.decompose(reps=decomposition_reps).qasm(filename='alzette_only.qasm')

final_circuit = make_circuit(qc, [a, b], [X, Y], [X, Y])

final_circuit.decompose(reps=2).qasm(filename='alzette.qasm')

result = run_circuit(final_circuit)

circuit_depth = qc.decompose(reps=decomposition_reps).depth()
gate_counts = qc.decompose(reps=decomposition_reps).count_ops()
print(f"Classical result: {c_alzette(a, b, c, n)}")
print(f"Quantum simulated result: {result}")
print("Circuit depth:", circuit_depth)
print("Circuit gate count:", json.dumps(gate_counts))
print("Circuit stats:", qc.stats)
