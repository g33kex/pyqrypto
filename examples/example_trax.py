from qiskit.circuit.quantumregister import AncillaRegister
from pyqrypto.rOperations import rCircuit, make_circuit, run_circuit
from pyqrypto.sparkle import c_traxl_genkeys, c_traxl_enc, Traxl_enc
from qiskit import QuantumRegister
from itertools import chain
import random
random.seed(42)

n = 256
x = [random.getrandbits(n//8) for _ in range(4)]
y = [random.getrandbits(n//8) for _ in range(4)]
tweak = [random.getrandbits(n//8) for _ in range(4)]
key = [random.getrandbits(n//8) for _ in range(8)]
print("x:", list(map(hex, x)))
print("y:", list(map(hex, y)))
print("key:", list(map(hex, key)))
print("tweak:", list(map(hex, tweak)))

X = [QuantumRegister(n//8, name=f'X{i}') for i in range(4)]
Y = [QuantumRegister(n//8, name=f'Y{i}') for i in range(4)]
K = [QuantumRegister(n//8, name=f'K{i}') for i in range(8)]
ancillas = AncillaRegister(Traxl_enc.get_num_ancilla_qubits(n))

qc = rCircuit(*X, *Y, *K, ancillas)
gate = Traxl_enc(X, Y, K, tweak, ancillas)
qc.append(gate, list(chain(*gate.inputs)))
print(qc.stats)

final_circuit = make_circuit(qc, x+y+key, gate.inputs[:-1], gate.outputs[:-1])

#final_circuit.decompose().qasm(filename='trax.qasm')
result = run_circuit(final_circuit)
print("end")

print("Quantum simulated results:")
print(f"quantum_x: {list(map(hex, result[0:4]))}")
print(f"quantum_y: {list(map(hex, result[4:8]))}")
print(f"quantum_last_subkey: {list(map(hex, result[8:16]))}")


subkeys = c_traxl_genkeys(key, n=n//8)
true_x, true_y = c_traxl_enc(x, y, subkeys, tweak, n=n//8)
print(f"last_subkeys: {list(map(hex, subkeys[-8::]))}")
print(f"true_x: {list(map(hex, true_x))}")
print(f"true_y: {list(map(hex, true_y))}")

if true_x == result[0:4] and true_y == result[4:8]:
    print("Test passed!")
else:
    print("Test failed!")

