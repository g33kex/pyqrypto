from qiskit import QuantumCircuit, QuantumRegister
import random

from rOperations import rROR, rROL, rXOR, rADD, rXORc, rAppend, make_circuit, run_circuit
from alzette import Alzette, alzette

nb_tests = 20

# Utils
def rol(x, r, n):
    r = r%n
    return ((x << r) | (x >> (n - r))) & ((2**n)-1)

def ror(x, r, n):
    r = r % n
    return ((x >> r) | (x << (n - r))) & ((2**n) - 1)

# Generic circuit test
def test_circuit(circuit, classical_function, inputs, input_registers, output_registers, verbose=False):
    """Tests if a circuit's output corresponds to the classical function's output"""
    true_result = classical_function(*inputs)

    if verbose:
        print("Testing the following gate:")
        print(circuit.decompose(reps=8))

    final_circuit = make_circuit(circuit, inputs, input_registers, output_registers)
    if verbose:
        print("Assembling it into the following circuit:")
        print(final_circuit)
        final_circuit.decompose().qasm(filename='circuit.qasm')
    result = run_circuit(final_circuit)

    if verbose:
        print("True result:\t", true_result)
        print("Actual result:\t", result)

    if true_result != result:
        return False
    return True

# Specific tests
def test_xor():
    for _ in range(nb_tests):
        n = random.randint(1, 10)
        a = random.getrandbits(n)
        b = random.getrandbits(n)

        X = QuantumRegister(n)
        Y = QuantumRegister(n)
        
        qc = QuantumCircuit(X, Y)
        rAppend(qc, rXOR(X, Y))

        result = test_circuit(qc, lambda x,y: [x^y], [a, b], [X, Y], [X])

        if not result:
            print("XOR test failed!")
            return False
    print("XOR test passed!")
    return True

def test_xorc():
    for _ in range(nb_tests):
        n = random.randint(1, 10)
        a = random.getrandbits(n)
        c = random.getrandbits(n)

        X1 = QuantumRegister(n)
        
        qc = QuantumCircuit(X1)
        X2 = rAppend(qc, rXORc(X1, c))

        result = test_circuit(qc, lambda x: [x^c], [a], [X1], [X2])

        if not result:
            print("XORc test failed!")
            return False
    print("XORc test passed!")
    return True

def test_prepare():
    for _ in range(nb_tests):
        n1 = random.randint(1, 10)
        n2 = random.randint(1, 10)
        a = random.getrandbits(n1)
        b = random.getrandbits(n2)

        X = QuantumRegister(n1)
        Y = QuantumRegister(n2)

        qc = QuantumCircuit(X, Y)
        
        result = test_circuit(qc, lambda x,y: [x,y], [a, b], [X, Y], [X, Y])

        if not result:
            print("Prepare test failed!")
            return False
    print("Prepare test passed!")
    return True

def test_ror():
    for _ in range(nb_tests):
        n = random.randint(1, 10)
        a = random.getrandbits(n)
        r = random.randint(1, 3*n)

        X1 = QuantumRegister(n)

        qc = QuantumCircuit(X1, name=f'ID')
        X2 = rAppend(qc, rROR(X1, r))

        result = test_circuit(qc, lambda x: [ror(x, r, n)], [a], [X1], [X2])
        if not result:
            print("ROR test failed!")
            return False
    print("ROR test passed!")
    return True
    
def test_rorrorxor():
    for _ in range(nb_tests):
        n = random.randint(1, 10)
        a = random.getrandbits(n)
        b = random.getrandbits(n)
        r1 = random.randint(1, n*3)
        r2 = random.randint(1, n*3)

        X1 = QuantumRegister(n)
        Y = QuantumRegister(n)

        qc = QuantumCircuit(X1, Y)
        X2 = rAppend(qc, rROR(X1, r1))
        Y2 = rAppend(qc, rROR(Y, r2))
        rAppend(qc, rXOR(X2, Y2))

        result = test_circuit(qc, lambda x,y: [ror(x, r1, n)^ror(y, r2, n), y], [a, b], [X1, Y], [X2, Y])
        if not result:
            print("RORORXOR test failed!")
            return False
    print("RORRORXOR test passed!")
    return True

def test_rorxorrolxor():
    for _ in range(nb_tests):
        n = random.randint(1, 10)
        a = random.getrandbits(n)
        b = random.getrandbits(n)
        r1 = random.randint(1, n*3)
        r2 = random.randint(1, n*3)

        X1 = QuantumRegister(n)
        Y1 = QuantumRegister(n)

        qc = QuantumCircuit(X1, Y1)
        X2 = rAppend(qc, rROR(X1, r1))
        rAppend(qc, rXOR(X2, Y1))
        Y2 = rAppend(qc, rROL(Y1, r2))
        rAppend(qc, rXOR(X2, Y2))

        result = test_circuit(qc, lambda x,y: [ror(x, r1, n)^y^rol(y, r2, n), rol(y, r2, n)], [a, b], [X1, Y1], [X2, Y2])

        if not result:
            print("RORXORROLXOR test failed!")
            return False
    print("RORXORROLXOR test passed!")
    return True

def test_complexxor():
    for _ in range(nb_tests):
        n = random.randint(1, 10)
        a = random.getrandbits(n)
        b = random.getrandbits(n)
        c = random.getrandbits(n)
        r1 = random.randint(0, n*3)
        r2 = random.randint(0, n*3)
        r3 = random.randint(0, n*3)

        X1 = QuantumRegister(n, name='X')
        Y1 = QuantumRegister(n, name='Y')
        Z = QuantumRegister(n, name='Z')

        qc = QuantumCircuit(X1, Y1, Z)
        X2 = rAppend(qc, rROR(X1, r1))

        rAppend(qc, rXOR(X2, Y1))
        Y2 = rAppend(qc, rROR(Y1, r2))
        rAppend(qc, rXOR(Y2, Z))
        rAppend(qc, rXOR(Z, rAppend(qc, rROL(X2, r3))))

        result = test_circuit(qc, lambda x,y,z: [ror(x, r1, n)^y, ror(y, r2, n)^z, z^rol(ror(x, r1, n)^y, r3, n)], [a, b, c], [X1, Y1, Z], [X2, Y2, Z])

        if not result:
            print("COMPLEXXOR test failed!")
            return False
    print("COMPLEXXOR test passed!")
    return True

def test_alzette():
    for _ in range(nb_tests):
        n = random.randint(1, 10)
        a = random.getrandbits(n)
        b = random.getrandbits(n)
        c = random.getrandbits(n)

        X = QuantumRegister(n)
        Y = QuantumRegister(n)

        qc = QuantumCircuit(X, Y)
        rAppend(qc, Alzette(X, Y, c))

        result = test_circuit(qc, lambda x,y: alzette(x, y, c, n), [a, b], [X, Y], [X, Y])

        if not result:
            print("Alzette test failed!")
            return False
    print("Alzette test passed!")
    return True

def test_add():
    for _ in range(nb_tests):
        n = random.randint(1, 10)
        a = random.getrandbits(n)
        b = random.getrandbits(n)

        X1 = QuantumRegister(n, name='X')
        Y = QuantumRegister(n, name='Y')

        qc = QuantumCircuit(X1, Y)

        X2 = rAppend(qc, rADD(X1, Y))

        result = test_circuit(qc, lambda x,y: [(x+y)%(2**n), y], [a, b], [X1, Y], [X2, Y])

        if not result:
            print("ADD test failed!")
            return False
    print("ADD test passed!")
    return True

def showcase_complex_circuit():
    n=4
    r1 = 1
    r2 = 2
    r3 = 5

    X1 = QuantumRegister(n, name='X')
    Y1 = QuantumRegister(n, name='Y')
    Z = QuantumRegister(n, name='Z')

    qc = QuantumCircuit(X1, Y1, Z)
    X2 = rAppend(qc, rROR(X1, r1))

    rAppend(qc, rXOR(X2, Y1))
    Y2 = rAppend(qc, rROR(Y1, r2))
    rAppend(qc, rXOR(Y2, Z))
    rAppend(qc, rXOR(Z, *rROL(X2, r3).outputs))

    result = test_circuit(qc, lambda x,y,z: [ror(x, r1, n)^y, ror(y, r2, n)^z, z^rol(ror(x, r1, n)^y, r3, n)], [9, 11, 2], [X1, Y1, Z], [X2, Y2, Z], verbose=True)

    if result:
        print("Test passed!")
    else:
        print("Test failed!")

def showcase_basic_xor():
    X = QuantumRegister(16, name='X')
    Y = QuantumRegister(16, name='Y')

    qc = QuantumCircuit(X, Y)
    rAppend(qc, rXOR(X, Y))

    print("Classical result:",30000^20000)
    final_circuit = make_circuit(qc, [30000, 20000], [X, Y], [X, Y])

    print("Assembling it into the following circuit:")
    print(final_circuit)
    result = run_circuit(final_circuit)
    print(result)
        
def showcase_add():
    X1 = QuantumRegister(8, name='X')
    Y = QuantumRegister(8, name='Y')

    qc = QuantumCircuit(X1, Y)

    X2 = rAppend(qc, rADD(X1, Y))

    print(qc.decompose(reps=2))

    final_circuit = make_circuit(qc, [74, 42], [X1, Y], [X2], filename='adder.qasm')
    print(final_circuit)

    result = run_circuit(final_circuit)

    print(result)

if __name__ == '__main__':
    random.seed(42)

    assert(test_prepare())
    assert(test_xor())
    assert(test_xorc())
    assert(test_ror())
    assert(test_rorrorxor())
    assert(test_rorxorrolxor())
    assert(test_complexxor())
    assert(test_add())
    assert(test_alzette())

    # Showcase complex circuit
    # showcase_complex_circuit()

    # Showcase basic XOR
    # showcase_basic_xor()

    #  Showcase ADD
    # showcase_add()

    # Test gates
    # n = 4
    # X = QuantumRegister(n, name='X')
    # Y = QuantumRegister(n, name='Y')
    # R = ClassicalRegister(n)
    # qc = QuantumCircuit(X, Y, R)
    # gate = rXOR(X, Y, label='XOR')
    # qc.append(gate, range(n*2))
    # print(qc.decompose())
    #
    # qc.measure(X, R)
    #
    # results = simulate(qc)
    # print(results)

    # Trying to simplify the interface
    # A1 = bRegister(3, name='A')
    # B1 = bRegister(3, name='B')
    # C1 = bRegister(3, name='C')
    # D1 = bRegister(3, name='D')
    #
    # A2 = A1^B1
    # C2 = C1^D1
    #
    # A3 = A2^C2
    # qc = QuantumCircuit(A1, B1, C1, D1)
    # for operation in A3.operation:
    #     rAppend(qc, operation)
    # print(qc)

    # X1 = QuantumRegister(3, name='X')
    # Y = QuantumRegister(3, name='Y')
    # A = AncillaRegister(1, name='ancilla')
    #
    # qc = QuantumCircuit(A, X1, Y)
    #
    # # How to get back our Ancilla Qubit??
    # X2, A2 = rAppend(qc, bAND(A, X1, Y))
    # print(qc.decompose(reps=2))
    #
    # final_circuit = make_circuit(qc, [5, 6], [X1, Y], [A2, X2, Y])
    # print(final_circuit.decompose())
    # with open('circuit.qasm', 'w') as f:
    #     f.write(final_circuit.decompose(reps=8).qasm())
    #
    # result = run_circuit(final_circuit, verbose=True)
    # print(result)
