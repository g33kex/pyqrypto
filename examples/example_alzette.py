from qiskit import QuantumRegister, Aer, QuantumCircuit, transpile
from qiskit_aer.backends import AerSimulator

from pyqrypto.sparkle import Alzette, c_alzette
from pyqrypto.rOperations import rCircuit, make_circuit, run_circuit
from itertools import chain
import json

# tensor_network 100
# stabilizer 100
# extended_stabilizer 63
# matrix_product_state 100 > 63
backend = AerSimulator(method="stabilizer", device="GPU")
print(backend)
qc = QuantumCircuit(100)
for i in range(100):
    qc.x(i)
qc.measure_all()
transpiled = transpile(qc, backend=backend, basis_gates=backend.configuration().basis_gates)
print(backend.run(transpiled).result().get_counts(transpiled))

#
n = 8
X = QuantumRegister(n, name='X')
Y = QuantumRegister(n, name='Y')
a = 384973 % 2**n
b = 1238444859 % 2**n
c = 0xb7e15162 % 2**n
print(f"Running alzette_{c}({a}, {b})")


qc = rCircuit(X, Y)
gate = Alzette(X, Y, c)
qc.add(X, Y, mode='ripple')
# qc.append(gate, list(chain(*gate.inputs)))

decomposition_reps = 2
qc.decompose(reps=decomposition_reps).qasm(filename='alzette_only.qasm')

final_circuit = make_circuit(qc, [a, b], [X, Y], [X, Y])

final_circuit.decompose(reps=2).qasm(filename='alzette.qasm')

print(Aer.backends())
# backend_sim = AerSimulator(method='stabilizer', device='GPU')
# backend_sim = AerSimulator.from_backend('aer_simulator')


backend_sim = AerSimulator()

transpiled = transpile(qc, backend=None, basis_gates=backend_sim.configuration().basis_gates, optimization_level=2)
transpiled.measure_all()
print(transpiled)
job_sim = backend_sim.run(transpiled)
result = job_sim.result()
# print(result.get_statevector(transpiled))
# job_sim = backend_sim.run(qc, shots=1024)
# result_sim = job_sim.result()
counts = result.get_counts(qc)
print(counts)
exit(1)


# result = run_circuit(final_circuit, method='automatic', device='GPU')

circuit_depth = qc.decompose(reps=decomposition_reps).depth()
gate_counts = qc.decompose(reps=decomposition_reps).count_ops()
print(f"Classical result: {c_alzette(a, b, c, n)}")
print(f"Quantum simulated result: {result}")
print("Circuit depth:", circuit_depth)
print("Circuit gate count:", json.dumps(gate_counts))
print("Circuit stats:", qc.stats)
