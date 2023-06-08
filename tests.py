from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
# Transpiler reduce the circuit to QASM instructions
from qiskit import transpile
# AerSimulator
from qiskit_aer import AerSimulator

# Create a 3 qubits quantum circuit
circ = QuantumCircuit(3)

# Add H gate on qubit 0 to put it in superposition
circ.h(0)

# Add CNOT on control qubit 0 and target qubit 1 to put them in Bell state
circ.cx(0, 1)

# Same but with target qubit 2
circ.cx(0, 2)

# Set initial state of the simulator
state = Statevector.from_int(0, 2**3)
# Evolve the state by the quantum circuit
state = state.evolve(circ)

# Draw the final state
state.draw('qsphere')

plt.show()

# Create measurement circuit
meas = QuantumCircuit(3, 3)
meas.barrier(range(3))
# Map the quantum measurements to classical bits
meas.measure(range(3), range(3))

# Compose two circuits, front=True means meas will be first
qc = meas.compose(circ, range(3), front=True)

print(qc.draw('text'))

backend = AerSimulator()

# Transpile the quantum circuit to low-level QASM instructions
qc_compiled = transpile(qc, backend)

print(qc_compiled.qasm())

# Execute the circuit on the simulator 1024 times
job_sim = backend.run(qc_compiled, shots=1024)

# Grab the results
result_sim = job_sim.result()

# Get the aggregated binary outcomes on the circuit
counts = result_sim.get_counts(qc_compiled)
print("Results:", counts)


# my_gate = Gate(name='my_gate', num_qubits=2, params=[])
# qr = QuantumRegister(3, 'q')
# circ2 = QuantumCircuit(qr)
# circ2.append(my_gate, [qr[0], qr[1]])
# circ2.append(my_gate, [qr[1], qr[2]])
#
# print(circ2.draw('text'))
#
# print(circ.decompose().draw('text']))

# customize a gate instruction (e.g., X gate)
qc = QuantumCircuit(2, name='X')
qc.x(0)
qc.h(1)
custom_gate = qc.to_instruction()
print(custom_gate)

# append custom gate to a new circuit
new_circ = QuantumCircuit(2)
new_circ.append(custom_gate, range(2))
print(new_circ)
