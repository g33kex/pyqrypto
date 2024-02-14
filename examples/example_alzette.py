from qiskit import QuantumRegister, qasm2

from pyqrypto.sparkle import Alzette, c_alzette
from pyqrypto.rOperations import rCircuit, make_circuit, run_circuit
from itertools import chain
from pathlib import Path

n = 8
X = QuantumRegister(n, name='X')
Y = QuantumRegister(n, name='Y')
a = 384973 % 2**n
b = 1238444859 % 2**n
c = 0xb7e15162 % 2**n
print(f"Running alzette_{c}({a}, {b})")


qc = rCircuit(X, Y)
gate = Alzette(X, Y, c)
qc.append(gate, list(chain(*gate.inputs)))

qasm2.dump(qc.decompose(reps=2), Path('alzette_only.qasm'))

final_circuit = make_circuit(qc, [a, b], [X, Y], [X, Y])

result = run_circuit(final_circuit, method='automatic')

print(f"Classical result: {c_alzette(a, b, c, n)}")
print(f"Quantum simulated result: {result}")
print("Circuit stats:", qc.stats)
