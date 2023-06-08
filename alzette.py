from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, transpile
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from QubitVector import QubitVector, make_circuit, run_circuit


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
def test_xor():
    n = 3#random.randint(1, 100)
    r = 1#random.randint(0, 10*n)
    a = 0b101#random.getrandbits(n) 
    b = 0b110#random.getrandbits(n)
    true_result = a^rol(b, r, n)
    #print(f"{fmt(a, n)}^rol({fmt(b, n)}, {r}) should be {fmt(true_result, n)}")
    
    X = QubitVector(range(n), initial_value=a)
    Y = QubitVector(range(n, n*2), initial_value=b)
    gate = X.XOR(Y.ROL(r))

    qc = QuantumCircuit(2*n)
    qc.append(gate, range(2*n))
    print(qc.decompose())

    circuit = make_circuit(qc, (X, Y), (X ,Y))
    print(circuit)
    result = run_circuit(circuit, (X, Y))

    print("Expected result:\t", true_result)
    print("Actual result:\t\t", result)

test_xor()
