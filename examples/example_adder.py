from pyqrypto.rOperations import rCircuit, make_circuit, run_circuit, rDKRSCarryLookaheadAdder
from pyqrypto.alzette import c_traxl_genkeys, c_traxl_enc, Traxl_enc
from qiskit import QuantumRegister, AncillaRegister
import matplotlib.pyplot as plt
from itertools import chain
import random
random.seed(42)

#n = 10
#n = random.randint(1, 10)
n = 5
wire_order = [0,5,10,1,6,11,2,7,14,12,3,8,13,4,9]

a = random.getrandbits(n)
b = random.getrandbits(n)

print(f"Doing operation {a}+{b}={a+b}")

A = QuantumRegister(n, name='A')
B = QuantumRegister(n, name='B')
ancillas = AncillaRegister(rDKRSCarryLookaheadAdder.get_num_ancilla_qubits(n))
print("total_qubits",len(A)+len(B)+len(ancillas))
qc = rCircuit(A, B, ancillas)

qc.add(A, B, ancillas, mode='lookahead')

fig = qc.decompose(gates_to_decompose=["rADD", "rCarry", "rCarry_reverse", "rXOR", 'rNOT'], reps=2).draw(wire_order=wire_order,output='mpl')
plt.show()


final_circuit = make_circuit(qc, [a, b], [A, B], [A])

result = run_circuit(final_circuit)

print("Result:", result[0])
