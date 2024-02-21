"""A simple test showcasing the general use of pyqrypto."""
from pyqrypto.register_operations import RegisterCircuit
from qiskit import QuantumRegister

# Create two 4 qubit registers
X1 = QuantumRegister(4, name="X")
Y1 = QuantumRegister(4, name="Y")

# Create a register circuit from registers X and Y.
# A register circuit can operate on quantum registers instead of on individual qubits.
qc = RegisterCircuit(X1, Y1, name="rCircuit")

# Rotate left register X by 3 qubits.
# A register can be seen as a "view" on logical qubits,
# so rotating a register just yields another view on these qubits with swapped indexes.
X2 = qc.ror(X1, 3)

# Let's XOR X2 Y1. The result will be stored in X3.
# Note here that X3 = X2 because xor doesn't modify the view.
X3 = qc.xor(X2, Y1)

# If we print the resulting circuit, we can see the XOR was done on a rotated version of X1.
print(qc.decompose())
#           ┌───┐
# X_0: ─────┤ X ├──────────
#           └─┬─┘┌───┐
# X_1: ───────┼──┤ X ├─────
#             │  └─┬─┘┌───┐
# X_2: ───────┼────┼──┤ X ├
#      ┌───┐  │    │  └─┬─┘
# X_3: ┤ X ├──┼────┼────┼──
#      └─┬─┘  │    │    │
# Y_0: ──■────┼────┼────┼──
#             │    │    │
# Y_1: ───────■────┼────┼──
#                  │    │
# Y_2: ────────────■────┼──
#                       │
# Y_3: ─────────────────■──

from pyqrypto.register_operations import make_circuit, run_circuit
from qiskit import qasm3

# Create a final circuit with measurement and preparation steps
# Let's initialize X1 to 4 and Y1 to 11
# Let's also measure X2 and Y1 at the end of the circuit
# It is possible to measure any QuantumRegistrer that has its qubits in the circuit
# This means we can measure the final state of any "view" of the qubits
# However note that the qubits will be in their final state during the measurement
# If we tried to measure X1, we wouldn't get the initial value of X1
# but the value of X2 left rotated by 3 bits
# This is because the value of X1 was overwritten by the XOR operation.
# This is why it is important to keep track of your registers during operations!
final_circuit = make_circuit(qc, [4, 11], [X1, Y1], [X2, Y1])

# We can print the final circuit
# As you can see the measurements are done on
print(final_circuit)
#       ┌─────────────┐ ┌───────┐   ┌─┐
#  X_0: ┤0            ├─┤1      ├───┤M├──────────────────
#       │             │ │       │   └╥┘┌─┐
#  X_1: ┤1            ├─┤2      ├────╫─┤M├───────────────
#       │  rPrepare 4 │ │       │    ║ └╥┘┌─┐
#  X_2: ┤2            ├─┤3      ├────╫──╫─┤M├────────────
#       │             │ │       │┌─┐ ║  ║ └╥┘
#  X_3: ┤3            ├─┤0      ├┤M├─╫──╫──╫─────────────
#       ├─────────────┴┐│  Rxor │└╥┘ ║  ║  ║ ┌─┐
#  Y_0: ┤0             ├┤4      ├─╫──╫──╫──╫─┤M├─────────
#       │              ││       │ ║  ║  ║  ║ └╥┘┌─┐
#  Y_1: ┤1             ├┤5      ├─╫──╫──╫──╫──╫─┤M├──────
#       │  rPrepare 11 ││       │ ║  ║  ║  ║  ║ └╥┘┌─┐
#  Y_2: ┤2             ├┤6      ├─╫──╫──╫──╫──╫──╫─┤M├───
#       │              ││       │ ║  ║  ║  ║  ║  ║ └╥┘┌─┐
#  Y_3: ┤3             ├┤7      ├─╫──╫──╫──╫──╫──╫──╫─┤M├
#       └──────────────┘└───────┘ ║  ║  ║  ║  ║  ║  ║ └╥┘
# c0: 4/══════════════════════════╩══╩══╩══╩══╬══╬══╬══╬═
#                                 0  1  2  3  ║  ║  ║  ║
# c1: 4/══════════════════════════════════════╩══╩══╩══╩═
#                                             0  1  2  3

# We can generate the QASM of that circuit and save it to a file
# This can be used to run the circuit on an actual quantum computer
with open("circuit.qasm", "w") as f:
    qasm3.dump(final_circuit, f)

# Let's run this circuit in a simulation and gather the result
results = run_circuit(final_circuit)

# We can verify that ror(4, 3)^11 = 3, and that Y1 was unchanged.
print(results)
# [3, 11]
