"""Example on how to use the TRAX-L gate."""
import random
from itertools import chain
from pathlib import Path

from pyqrypto.register_operations import RegisterCircuit, make_circuit, run_circuit
from pyqrypto.sparkle import TraxlEnc, c_traxl_enc, c_traxl_genkeys
from qiskit import QuantumRegister, qasm3
from qiskit.circuit.quantumregister import AncillaRegister

random.seed(42)

# Create the inputs, tweak and encryption key
n = 256
x = [random.getrandbits(n // 8) for _ in range(4)]
y = [random.getrandbits(n // 8) for _ in range(4)]
tweak = [random.getrandbits(n // 8) for _ in range(4)]
key = [random.getrandbits(n // 8) for _ in range(8)]
print("x:", list(map(hex, x)))
print("y:", list(map(hex, y)))
print("key:", list(map(hex, key)))
print("tweak:", list(map(hex, tweak)))

# Create the quantum registers
X = [QuantumRegister(n // 8, name=f"X{i}") for i in range(4)]
Y = [QuantumRegister(n // 8, name=f"Y{i}") for i in range(4)]
K = [QuantumRegister(n // 8, name=f"K{i}") for i in range(8)]
ancillas = AncillaRegister(TraxlEnc.get_num_ancilla_qubits(n))

# Create the circuit and add the TRAX-L encryption gate
qc = RegisterCircuit(*X, *Y, *K, ancillas)
gate = TraxlEnc(X, Y, K, tweak, ancillas)
qc.append(gate, list(chain(*gate.inputs)))

# Print circuit statistics
print(qc.stats)

# Save the QASM to a file
with Path("traxl.qasm").open("w") as f:
    qasm3.dump(qc.decompose(reps=2), f)

# Add state preparation and measurement
final_circuit = make_circuit(qc, x + y + key, gate.inputs[:-1], gate.outputs[:-1])

# Run the simulation (it might take a while)
result = run_circuit(final_circuit, method="matrix_product_state")

# Print the results
print("Quantum simulated results:")
print(f"quantum_x: {list(map(hex, result[0:4]))}")
print(f"quantum_y: {list(map(hex, result[4:8]))}")
print(f"quantum_last_subkey: {list(map(hex, result[8:16]))}")

# Do the same computation classically to compare the results
subkeys = c_traxl_genkeys(key, n=n // 8)
true_x, true_y = c_traxl_enc(x, y, subkeys, tweak, n=n // 8)
print(f"last_subkeys: {list(map(hex, subkeys[-8::]))}")
print(f"true_x: {list(map(hex, true_x))}")
print(f"true_y: {list(map(hex, true_y))}")

assert true_x == result[0:4] and true_y == result[4:8]
