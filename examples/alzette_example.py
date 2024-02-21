"""Example on how to use the Alzette gate."""
from itertools import chain
from pathlib import Path

from pyqrypto.register_operations import RegisterCircuit, make_circuit, run_circuit
from pyqrypto.sparkle import Alzette, c_alzette
from qiskit import QuantumRegister, qasm3

n = 8
# Create two quantum registers
X = QuantumRegister(n, name="X")
Y = QuantumRegister(n, name="Y")
a = 384973 % 2**n
b = 1238444859 % 2**n
c = 0xB7E15162 % 2**n
print(f"Running alzette_{c}({a}, {b})")


qc = RegisterCircuit(X, Y)

# Add the Alzette get
gate = Alzette(X, Y, c)
qc.append(gate, list(chain(*gate.inputs)))

# Save the QASM to a file
with Path("alzette.qasm").open("w") as f:
    qasm3.dump(qc.decompose(reps=2), f)

final_circuit = make_circuit(qc, [a, b], [X, Y], [X, Y])

# Simulate the circuit
result = run_circuit(final_circuit, method="automatic")

# Print some statistics about the circuit (depth, gate counts and quantum cost)
print("Circuit stats:", qc.stats)
# Make sure the classical result matches the quantum result
print(f"Classical result: {c_alzette(a, b, c, n)}")
print(f"Quantum simulated result: {result}")
