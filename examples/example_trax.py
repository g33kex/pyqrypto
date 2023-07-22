from qiskit.circuit.quantumregister import AncillaRegister
from pyqrypto.rOperations import rCircuit, make_circuit, run_circuit, rOperation
from pyqrypto.alzette import c_traxl_genkeys, c_traxl_enc, Traxl_enc
from qiskit import QuantumRegister
from itertools import chain
import random
random.seed(42)
import sys
sys.settrace

# x = [0xaea9f4e8, 0xc926a22f, 0x9d1078f8, 0x8a779f98]
# y = [0x24e002dc, 0x8f34f225, 0x13ff3742, 0x510e85ea]
#
# key = [0x3b9c0bb1, 0xcc2106fb, 0x28bc5755, 0xb146dc0f, 0xe111aad7, 0xca29eea5, 0x612eef46, 0x5ace7bc]
# tweak = [0x7c856a02, 0x62e011b3, 0xc016150f, 0xc0045ae6];
#
# subkeys = c_traxl_genkeys(key)
#
# print(list(map(hex, subkeys)))
#
# new_x, new_y = c_traxl_enc(x, y, subkeys, tweak)
#
# print("X:", list(map(hex, new_x)))
# print("Y:", list(map(hex, new_y)))
#
n = 256
for _ in range(1):
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
        pass
    else:
        print("Test failed!")
        exit(1)

print("Test passed!")
