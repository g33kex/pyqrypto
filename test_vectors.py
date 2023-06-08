from qiskit import QuantumCircuit, QuantumRegister
import matplotlib.pyplot as plt

from QubitVector import QubitVector

from qiskit.circuit.library import MCXGate
gate = MCXGate(4)

x = QubitVector(range(4))
y = QubitVector(range(4, 8))
print(x)
circuit = QuantumCircuit(8)
circuit.append(gate, x)
circuit.append(gate, y)
circuit.draw('mpl')
plt.show()
X = QubitVector(4)
print(X)
qc = QuantumCircuit(0, 4)
qc.add_register(X.qc)
print(qc)
