from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, transpile
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from QubitVector import QubitVector

def simulate(qc):
    """Simulate the given circuit and returns the results"""
    backend_sim = Aer.get_backend('qasm_simulator')
    job_sim = backend_sim.run(transpile(qc, backend_sim), shots=1024)
    result_sim = job_sim.result()
    counts = result_sim.get_counts(qc)
    return counts

def test_gate(gate, X, Y):
    n = len(X)
    if len(X) != len(Y):
        raise Exception("X and Y should have the same length")
    qc = QuantumCircuit(n*2, n)
    qc.append(X.prepare(), X)
    qc.append(Y.prepare(), Y)
    qc.append(gate, range(n*2))
    qc.measure(range(n), range(n))
    qc.decompose().draw('mpl')
    plt.show()

    print(f"Running {a} {gate.name} {b}")
    print("Results:", simulate(qc))
    with open('circuit.qasm', 'w') as f:
        f.write(qc.decompose().qasm())

def rol(x, r, n):
    r = r%n
    return ((x << r) | (x >> (n - r))) & ((2**n)-1)

def fmt(x, n):
    return '{0:0{n}b}'.format(x, n=n)
n = 3

# #a = 0b100110
# a = 0b100
# b = 0b010
# result = a^rol(b, 2, n)
# print(f"{fmt(a, n)}^rol({fmt(b, n)}, 2) should be {fmt(result, n)}")
#
# X = QubitVector(range(n), initial_value=a)
# Y = QubitVector(range(n, n*2), initial_value=b)
# # We can create gates, add them, and do permutations on the bitvectors between creating them
#
# # Test this, unit tests?
# # Could we include the operations in the __str__ representation??
# xor = X.XOR(Y.ROL(2))
# test_gate(xor, X, Y)
#
# # How can we set the initial value of QubitVector? Should that be includedi n the QubitVector?

import random

def make_circuit(circuit, input_vectors, output_vectors):
    """Make a circuit with vectors as input and measurement for the output vectors"""
    # How to specify output? This should add the measure gate maybe? 
    # Maybe just mark which qubitvector is the output! So we have a circuit that operates on qubitvectors instead of qubits...
    # TODO: automatically add the classical bits for output vector, this is easy
    # TODO: multiple output/input vectors needed, test with other number than 3, like X XOR Y XOR S3(Z)
    # Copy the circuit to avoid modifying it
    qc = circuit.copy()
    # Add classical register for the output vectors
    qc.add_register(ClassicalRegister(sum(map(len, output_vectors))))
    for input_vector in input_vectors:
        qc.compose(input_vector.prepare(), input_vector, inplace=True, front=True)
    output_index = 0
    for output_vector in output_vectors:
        qc.measure(output_vector, range(output_index, output_index+len(output_vector)))
        output_index += len(output_vector)
    print(qc)
    return qc

def run_circuit(circuit, input_vectors, output_vectors, expected_output):
    """Run circuit with vectors as input and returns the integer value of the output vectors"""
    results = simulate(make_circuit(circuit, input_vectors, output_vectors)).most_frequent()
    print(results)
    #return result == expected_output

def test_xor():
    n = 3#random.randint(1, 100)
    r = 2#random.randint(0, 10*n)
    a = 0b101#random.getrandbits(n) 
    b = 0b100#random.getrandbits(n)
    true_result = a^rol(b, r, n)
    #print(f"{fmt(a, n)}^rol({fmt(b, n)}, {r}) should be {fmt(true_result, n)}")
    
    X = QubitVector(range(n), initial_value=a)
    Y = QubitVector(range(n, n*2), initial_value=b)
    gate = X.XOR(Y.ROL(r))

    qc = QuantumCircuit(n*2)
    qc.append(gate, range(n*2))
    
    run_circuit(qc, (X, Y), (X,), true_result)



test_xor()
