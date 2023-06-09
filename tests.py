from qiskit import QuantumCircuit, QuantumRegister
import random

from bitwise_operations import ROR, ROL, bXOR, bAppend, make_circuit, run_circuit

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
    for _ in range(20):
        n = random.randint(1, 10)
        a = random.getrandbits(n)
        b = random.getrandbits(n)

        X = QuantumRegister(n)
        Y = QuantumRegister(n)
        
        qc = QuantumCircuit(X, Y)
        bAppend(qc, bXOR(n), [X, Y])

        result = test_circuit(qc, lambda x,y: [x^y], [a, b], [X, Y], [X])

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

        X1 = QuantumRegister(n)

        qc = QuantumCircuit(X1, name=f'ID')
        X2 = ROR(X1, r)

        result = test_circuit(qc, lambda x: [ror(x, r, n)], [a], [X1], [X2])
        if not result:
            print("ROR test failed!")
            return False
    print("ROR test passed!")
    return True
    
def test_rorrorxor():
    for _ in range(20):
        n = random.randint(1, 10)
        a = random.getrandbits(n)
        b = random.getrandbits(n)
        r1 = random.randint(1, n*3)
        r2 = random.randint(1, n*3)

        X1 = QuantumRegister(n)
        Y = QuantumRegister(n)

        qc = QuantumCircuit(X1, Y)
        X2 = ROR(X1, r1)
        bAppend(qc, bXOR(n), [X2, ROR(Y, r2)])

        result = test_circuit(qc, lambda x,y: [ror(x, r1, n)^ror(y, r2, n), y], [a, b], [X1, Y], [X2, Y])
        if not result:
            print("RORORXOR test failed!")
            return False
    print("RORRORXOR test passed!")
    return True

def test_rorxorrolxor():
    for _ in range(20):
        n = random.randint(1, 10)
        a = random.getrandbits(n)
        b = random.getrandbits(n)
        r1 = random.randint(1, n*3)
        r2 = random.randint(1, n*3)

        X1 = QuantumRegister(n)
        Y1 = QuantumRegister(n)

        qc = QuantumCircuit(X1, Y1)
        X2 = ROR(X1, r1)
        bAppend(qc, bXOR(n), [X2, Y1])
        Y2 = ROL(Y1, r2)
        bAppend(qc, bXOR(n), [X2, Y2])

        result = test_circuit(qc, lambda x,y: [ror(x, r1, n)^y^rol(y, r2, n), rol(y, r2, n)], [a, b], [X1, Y1], [X2, Y2])

        if not result:
            print("RORXORROLXOR test failed!")
            return False
    print("RORXORROLXOR test passed!")
    return True

def test_complexxor():
    for _ in range(40):
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
        X2 = ROR(X1, r1)

        bAppend(qc, bXOR(n), [X2, Y1])
        Y2 = ROR(Y1, r2)
        bAppend(qc, bXOR(n), [Y2, Z])
        bAppend(qc, bXOR(n), [Z, ROL(X2, r3)])

        result = test_circuit(qc, lambda x,y,z: [ror(x, r1, n)^y, ror(y, r2, n)^z, z^rol(ror(x, r1, n)^y, r3, n)], [a, b, c], [X1, Y1, Z], [X2, Y2, Z])

        if not result:
            print("COMPLEXXOR test failed!")
            return False
    print("COMPLEXXOR test passed!")
    return True

if __name__ == '__main__':
    random.seed(42)

    assert(test_xor())
    assert(test_ror())
    assert(test_rorrorxor())
    assert(test_rorxorrolxor())
    assert(test_complexxor())
    
    # Showcase complex circuit
    n=4
    r1 = 1
    r2 = 2
    r3 = 5

    X1 = QuantumRegister(n, name='X')
    Y1 = QuantumRegister(n, name='Y')
    Z = QuantumRegister(n, name='Z')

    qc = QuantumCircuit(X1, Y1, Z)
    X2 = ROR(X1, r1)

    bAppend(qc, bXOR(n), [X2, Y1])
    Y2 = ROR(Y1, r2)
    bAppend(qc, bXOR(n), [Y2, Z])
    bAppend(qc, bXOR(n), [Z, ROL(X2, r3)])

    result = test_circuit(qc, lambda x,y,z: [ror(x, r1, n)^y, ror(y, r2, n)^z, z^rol(ror(x, r1, n)^y, r3, n)], [9, 11, 2], [X1, Y1, Z], [X2, Y2, Z], verbose=True)

    if result:
        print("Test passed!")
    else:
        print("Test failed!")
    
