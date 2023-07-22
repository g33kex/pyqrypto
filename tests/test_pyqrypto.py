from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
import random

from pyqrypto.rOperations import make_circuit, run_circuit, rCircuit, rDKRSCarryLookaheadAdder, rConstantDKRSCarryLookaheadAdder
from pyqrypto.alzette import Alzette, Traxl_enc, c_alzette, c_traxl_genkeys, c_traxl_enc
from itertools import chain
import matplotlib.pyplot as plt

nb_tests = 100

# Utils
def rol(x, r, n):
    r = r%n
    return ((x << r) | (x >> (n - r))) & ((2**n)-1)

def ror(x, r, n):
    r = r % n
    return ((x >> r) | (x << (n - r))) & ((2**n) - 1)

# Generic circuit test
def circuit_test(circuit, classical_function, inputs, input_registers, output_registers, verbose=False):
    """Tests if a circuit's output corresponds to the classical function's output"""
    true_result = classical_function(*inputs)

    if verbose:
        print("Testing the following gate:")
        print(circuit.decompose(reps=1))

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
def test_prepare():
    for _ in range(nb_tests):
        n1 = random.randint(1, 10)
        n2 = random.randint(1, 10)
        a = random.getrandbits(n1)
        b = random.getrandbits(n2)

        X = QuantumRegister(n1)
        Y = QuantumRegister(n2)

        qc = QuantumCircuit(X, Y)
        
        result = circuit_test(qc, lambda x,y: [x,y], [a, b], [X, Y], [X, Y])

        assert(result)

def test_ror():
    for _ in range(nb_tests):
        n = random.randint(1, 10)
        a = random.getrandbits(n)
        r = random.randint(1, 3*n)

        X1 = QuantumRegister(n)

        qc = rCircuit(X1)
        X2 = qc.ror(X1, r)

        result = circuit_test(qc, lambda x: [ror(x, r, n)], [a], [X1], [X2])

        assert(result)

def test_xor():
    for _ in range(nb_tests):
        n = random.randint(1, 10)
        a = random.getrandbits(n)
        b = random.getrandbits(n)

        X = QuantumRegister(n)
        Y = QuantumRegister(n)
        
        qc = rCircuit(X, Y)
        qc.xor(X, Y)

        result = circuit_test(qc, lambda x,y: [x^y], [a, b], [X, Y], [X])

        assert(result)

def test_xorc():
    for _ in range(nb_tests):
        n = random.randint(1, 10)
        a = random.getrandbits(n)
        c = random.getrandbits(n)

        X1 = QuantumRegister(n)
        
        qc = rCircuit(X1)
        X2 = qc.xor(X1, c)

        result = circuit_test(qc, lambda x: [x^c], [a], [X1], [X2])

        assert(result)

def test_rorrorxor():
    for _ in range(nb_tests):
        n = random.randint(1, 10)
        a = random.getrandbits(n)
        b = random.getrandbits(n)
        r1 = random.randint(1, n*3)
        r2 = random.randint(1, n*3)

        X1 = QuantumRegister(n)
        Y1 = QuantumRegister(n)

        qc = rCircuit(X1, Y1)
        X2 = qc.ror(X1, r1)
        Y2 = qc.ror(Y1, r2)
        qc.xor(X2, Y2)

        result = circuit_test(qc, lambda x,y: [ror(x, r1, n)^ror(y, r2, n), y], [a, b], [X1, Y1], [X2, Y1])

        assert(result)

def test_rorxorrolxor():
    for _ in range(nb_tests):
        n = random.randint(1, 10)
        a = random.getrandbits(n)
        b = random.getrandbits(n)
        r1 = random.randint(1, n*3)
        r2 = random.randint(1, n*3)

        X1 = QuantumRegister(n)
        Y1 = QuantumRegister(n)

        qc = rCircuit(X1, Y1)
        X2 = qc.xor(qc.ror(X1, r1), Y1)
        Y2 = qc.rol(Y1, r2)
        qc.xor(X2, Y2)

        result = circuit_test(qc, lambda x,y: [ror(x, r1, n)^y^rol(y, r2, n), rol(y, r2, n)], [a, b], [X1, Y1], [X2, Y2])

        assert(result)

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

        qc = rCircuit(X1, Y1, Z)
        X2 = qc.ror(X1, r1)
        qc.xor(X2, Y1)
        Y2 = qc.ror(Y1, r2)
        qc.xor(Y2, Z)
        qc.xor(Z, qc.rol(X2, r3))

        result = circuit_test(qc, lambda x,y,z: [ror(x, r1, n)^y, ror(y, r2, n)^z, z^rol(ror(x, r1, n)^y, r3, n)], [a, b, c], [X1, Y1, Z], [X2, Y2, Z])

        assert(result)

def test_alzette():
    for _ in range(nb_tests):
        n = random.randint(1, 10)
        a = random.getrandbits(n)
        b = random.getrandbits(n)
        c = random.getrandbits(n)

        X = QuantumRegister(n)
        Y = QuantumRegister(n)

        qc = QuantumCircuit(X, Y)
        gate = Alzette(X, Y, c)
        qc.append(gate, chain(*gate.outputs))

        result = circuit_test(qc, lambda x,y: c_alzette(x, y, c, n), [a, b], [X, Y], [X, Y])

        assert(result)

def test_traxl():
    # This test is really slow so we must run it less times
    for i in range(nb_tests//10):
        # if n = 64 or 128 this fails with segfault, probably a bug in Qiskit
        #n = random.choice([16, 32, 64, 128, 256])
        n = 32
        x = [random.getrandbits(n//8) for _ in range(4)]
        y = [random.getrandbits(n//8) for _ in range(4)]
        tweak = [random.getrandbits(n//8) for _ in range(4)]
        key = [random.getrandbits(n//8) for _ in range(8)]

        X = [QuantumRegister(n//8, name=f'X{i}') for i in range(4)]
        Y = [QuantumRegister(n//8, name=f'Y{i}') for i in range(4)]
        K = [QuantumRegister(n//8, name=f'K{i}') for i in range(8)]
        ancillas = AncillaRegister(Traxl_enc.get_num_ancilla_qubits(n))

        qc = rCircuit(*X, *Y, *K, ancillas)
        gate = Traxl_enc(X, Y, K, tweak, ancillas)
        qc.append(gate, list(chain(*gate.inputs)))

        result = circuit_test(qc, lambda *params: [a for sublist in c_traxl_enc(list(params[0:4]), list(params[4:8]), c_traxl_genkeys(list(params[8::]), n=n//8), tweak, n=n//8) for a in sublist], x+y+key, gate.inputs[:-1], gate.outputs[:-9])

        assert(result)

def test_ripple_add():
    for _ in range(nb_tests):
        n = random.randint(1, 10)
        a = random.getrandbits(n)
        b = random.getrandbits(n)

        X1 = QuantumRegister(n, name='X')
        Y = QuantumRegister(n, name='Y')

        qc = rCircuit(X1, Y)

        X2 = qc.add(X1, Y, mode='ripple')

        result = circuit_test(qc, lambda x,y: [(x+y)%(2**n), y], [a, b], [X1, Y], [X2, Y])

        assert(result)

def test_lookahead_add():
    for _ in range(nb_tests):
        #n = 10
        #n = random.randint(1, 10)
        n = 32
        wire_order=None
        # if n==10:
        #     wire_order = [0,10,20,1,11,21,2,12,29,22,3,13,23,4,14,30,24,5,15,32,25,6,16,31,26,7,17,27,8,18,28,9,19]
        # elif n==5:
        #     wire_order = [0,5,10,1,6,11,2,7,14,12,3,8,13,4,9]

        a = random.getrandbits(n)
        b = random.getrandbits(n)

        print(f"Doing operation {a}+{b}={a+b}")

        A = QuantumRegister(n, name='A')
        B = QuantumRegister(n, name='B')
        ancillas = AncillaRegister(rDKRSCarryLookaheadAdder.get_num_ancilla_qubits(n))
        print("total_qubits",len(A)+len(B)+len(ancillas))
        qc = rCircuit(A, B, ancillas)

        qc.add(A, B, ancillas, mode='lookahead')
        print(f"n: {n}\tcost: {qc.quantum_cost}")

        #print(qc.decompose(gates_to_decompose=["rADD", "rCarry", "rCarry_reverse", "rXOR", 'rNOT'], reps=2).draw(wire_order=wire_order,output='mpl'))
        #plt.show()

        #print(qc.decompose(gates_to_decompose=["rADD", "rCarry", "rCarry_reverse", "rXOR", "rNOT"], reps=2))

        result = circuit_test(qc, lambda x,y: [(x+y)%(2**n), y], [a, b], [A, B], [A, B], verbose=False)

        assert(result)

def test_addc():
    for _ in range(nb_tests):
        n = random.randint(1, 10)
        a = random.getrandbits(n)
        c = random.getrandbits(n)
        ancillas = AncillaRegister(rConstantDKRSCarryLookaheadAdder.get_num_ancilla_qubits(n))

        X1 = QuantumRegister(n)
        
        qc = rCircuit(X1, ancillas)
        X2 = qc.add(X1, c, ancillas)

        result = circuit_test(qc, lambda x: [(x+c)%2**n], [a], [X1], [X2], verbose=True)

        assert(result)

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

    result = circuit_test(qc, lambda x,y,z: [ror(x, r1, n)^y, ror(y, r2, n)^z, z^rol(ror(x, r1, n)^y, r3, n)], [9, 11, 2], [X1, Y1, Z], [X2, Y2, Z], verbose=True)

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
    test_traxl()
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
