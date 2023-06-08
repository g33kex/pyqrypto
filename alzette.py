from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, transpile
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from QubitVector import QubitVector, make_circuit, run_circuit


# def test_gate(gate, X, Y):
#     n = len(X)
#     if len(X) != len(Y):
#         raise Exception("X and Y should have the same length")
#     qc = QuantumCircuit(n*2, n)
#     qc.append(X.prepare(), X)
#     qc.append(Y.prepare(), Y)
#     qc.append(gate, range(n*2))
#     qc.measure(range(n), range(n))
#     qc.decompose().draw('mpl')
#     plt.show()
#
#     print(f"Running {a} {gate.name} {b}")
#     print("Results:", simulate(qc))
#     with open('circuit.qasm', 'w') as f:
#         f.write(qc.decompose().qasm())

def rol(x, r, n):
    r = r%n
    return ((x << r) | (x >> (n - r))) & ((2**n)-1)

def ror(x, r, n):
    r = r % n
    return ((x >> r) | (x << (n - r))) & ((2**n) - 1)

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

def test_gate(gate, true_func, inputs: list, outputs: list, verbose=False):
    true_result = true_func(*[input.initial_value for input in inputs])
    nb_qubits = sum(map(len, inputs))

    qc = QuantumCircuit(nb_qubits)
    qc.append(gate, range(nb_qubits))
    if verbose:
        print("Testing the following gate:")
        print(qc.decompose(reps=5))

    circuit = make_circuit(qc, inputs, outputs)
    if verbose:
        print("Assembling it into the following circuit:")
        print(circuit)
        with open('circuit.qasm', 'w') as f:
            f.write(circuit.decompose().qasm())
    result = run_circuit(circuit, outputs)

    if verbose:
        print("Expected result:\t", true_result)
        print("Actual result:\t\t", result)

    if true_result != result:
        return False

    return True


import random

def test_xor():
    for _ in range(20):
        n = random.randint(1, 10)
        a = random.getrandbits(n)
        b = random.getrandbits(n)

        X = QubitVector(range(n), a)
        Y = QubitVector(range(n, n*2), b)
        gate = X.XOR(Y)

        result = test_gate(gate, lambda a,b: [a^b, b], [X,Y], [X,Y])
        if not result:
            print("XOR test failed!")
            return False
    print("XOR test passed!")
    return True

def test_ror():
    for _ in range(20):
        n = random.randint(1, 10)
        a = random.getrandbits(n)
        r = random.randint(1, 3*n)

        X = QubitVector(range(n), a)
        X.ROR(r)

        qc = QuantumCircuit(n, name=f'ID')
        gate = qc.to_instruction()

        result = test_gate(gate, lambda a: [ror(a,r,n)], [X], [X])
        if not result:
            print("ROR test failed!")
            return False
    print("ROR test passed!")
    return True
    

def test_xorror():
    for _ in range(20):
        n = random.randint(1, 10)
        a = random.getrandbits(n)
        b = random.getrandbits(n)
        r1 = random.randint(1, n*3)
        r2 = random.randint(1, n*3)

        X = QubitVector(range(n), a)
        Y = QubitVector(range(n, n*2), b)
        X.ROR(r1)
        Y.ROR(r2)
        gate = X.XOR(Y)

        result = test_gate(gate, lambda a,b: [ror(a,r1,n)^ror(b, r2, n), ror(b, r2, n)], [X,Y], [X,Y])
        if not result:
            print("XORROR test failed!")
            return False
    print("XORROR test passed!")
    return True

def test_rorxorrolxor():
    for _ in range(20):
        n = random.randint(1, 10)
        a = random.getrandbits(n)
        b = random.getrandbits(n)
        r1 = random.randint(1, n*3)
        r2 = random.randint(1, n*3)

        X = QubitVector(range(n), a)
        Y = QubitVector(range(n, n*2), b)
        X.ROR(r1)
        gate1 = X.XOR(Y)
        Y.ROL(r2)
        gate2 = X.XOR(Y)

        qc = QuantumCircuit(n*2, name=f'XORXOR')
        qc.append(gate1, range(n*2))
        qc.append(gate2, range(n*2))
        gate = qc.to_instruction()

        result = test_gate(gate, lambda a,b: [ror(a,r1,n)^b^(rol(b, r2, n)), rol(b, r2, n)], [X,Y], [X,Y])

        if not result:
            print("RORXORROLXOR test failed!")
            return False
    print("RORXORROLXOR test passed!")
    return True

# The issue with this test is that the ROR of X is not taken into account, the XOR gate should return the new X and Y. But how to do composition? 
# TODO: we can make the register unmutable, but we have to reset the bits to the previous position if they are reused, each register should have a gate path from the beginning to their state... but what if they use other registers to get into their current state? Maybe those other registers are not avaliable...
# Maybe we could use a start state

if __name__ == '__main__':
    assert(test_xor())
    assert(test_ror())
    assert(test_xorror())
    assert(test_rorxorrolxor())
    n = 3
    a = 2
    b = 6
    r1 = 1
    r2 = 2

    X = QubitVector(range(n), a)
    Y = QubitVector(range(n, n*2), b)
    X.ROR(r1)
    gate1 = X.XOR(Y)
    Y.ROL(r2)
    gate2 = X.XOR(Y)

    qc = QuantumCircuit(n*2, name=f'XORXOR')
    qc.append(gate1, range(n*2))
    qc.append(gate2, range(n*2))
    gate = qc.to_instruction()

    test_gate(gate, lambda a,b: [ror(a,r1,n)^b^(rol(b, r2, n)), rol(b, r2, n)], [X,Y], [X,Y], verbose=True)


