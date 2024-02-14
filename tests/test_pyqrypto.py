from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from pyqrypto.rOperations import make_circuit, run_circuit, rCircuit, rDKRSCarryLookaheadAdder, rConstantDKRSCarryLookaheadAdder
from pyqrypto.sparkle import Alzette, Traxl_enc, Traxm_enc_round, c_alzette, c_traxl_genkeys, c_traxl_enc, c_traxm_enc
from itertools import chain
import random

"""Number of times to run each test"""
NB_TESTS = 100
"""Device for the tests"""
DEVICE = "CPU"
"""Simulation method"""
METHOD = "matrix_product_state"

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
    result = run_circuit(final_circuit, device=DEVICE, method=METHOD)

    if verbose:
        print("True result:\t", true_result)
        print("Actual result:\t", result)

    if true_result != result:
        return False
    return True

# Specific tests
def test_prepare():
    for _ in range(NB_TESTS):
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
    for _ in range(NB_TESTS):
        n = random.randint(1, 10)
        a = random.getrandbits(n)
        r = random.randint(1, 3*n)

        X1 = QuantumRegister(n)

        qc = rCircuit(X1)
        X2 = qc.ror(X1, r)

        result = circuit_test(qc, lambda x: [ror(x, r, n)], [a], [X1], [X2])

        assert(result)

def test_xor():
    for _ in range(NB_TESTS):
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
    for _ in range(NB_TESTS):
        n = random.randint(1, 10)
        a = random.getrandbits(n)
        c = random.getrandbits(n)

        X1 = QuantumRegister(n)
        
        qc = rCircuit(X1)
        X2 = qc.xor(X1, c)

        result = circuit_test(qc, lambda x: [x^c], [a], [X1], [X2])

        assert(result)

def test_rorrorxor():
    for _ in range(NB_TESTS):
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
    for _ in range(NB_TESTS):
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
    for _ in range(NB_TESTS):
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

def test_ripple_add():
    for _ in range(NB_TESTS):
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
    for _ in range(NB_TESTS//5):
        n = random.randint(1, 32)
        a = random.getrandbits(n)
        b = random.getrandbits(n)

        A = QuantumRegister(n, name='A')
        B = QuantumRegister(n, name='B')
        ancillas = AncillaRegister(rDKRSCarryLookaheadAdder.get_num_ancilla_qubits(n))

        qc = rCircuit(A, B, ancillas)
        qc.add(A, B, ancillas, mode='lookahead')

        if ancillas:
            result = circuit_test(qc, lambda x,y, _: [(x+y)%(2**n), y, 0], [a, b, 0], [A, B, ancillas], [A, B, ancillas], verbose=False)
        else:
            result = circuit_test(qc, lambda x,y: [(x+y)%(2**n), y], [a, b], [A, B], [A, B], verbose=False)

        assert(result)

def test_addc():
    for _ in range(NB_TESTS):
        n = random.randint(1, 10)
        a = random.getrandbits(n)
        c = random.getrandbits(n)
        ancillas = AncillaRegister(rConstantDKRSCarryLookaheadAdder.get_num_ancilla_qubits(n))

        X1 = QuantumRegister(n)
        
        qc = rCircuit(X1, ancillas)
        X2 = qc.add(X1, c, ancillas)

        if ancillas:
            result = circuit_test(qc, lambda x, _: [(x+c)%2**n, 0], [a, 0], [X1, ancillas], [X2, ancillas])
        else:
            result = circuit_test(qc, lambda x: [(x+c)%2**n], [a], [X1], [X2])

        assert(result)

def test_alzette():
    for _ in range(NB_TESTS):
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

def test_ctraxl():
    x = [0xaea9f4e8, 0xc926a22f, 0x9d1078f8, 0x8a779f98]
    y = [0x24e002dc, 0x8f34f225, 0x13ff3742, 0x510e85ea]

    key = [0x3b9c0bb1, 0xcc2106fb, 0x28bc5755, 0xb146dc0f, 0xe111aad7, 0xca29eea5, 0x612eef46, 0x5ace7bc]
    tweak = [0x7c856a02, 0x62e011b3, 0xc016150f, 0xc0045ae6]

    subkeys = c_traxl_genkeys(key, n=32)
    res_x, res_y = c_traxl_enc(x, y, subkeys, tweak, n=32)
    true_x = [0xef5eabc9, 0xc3deec85, 0x3be9c4fd, 0x7db95c88]
    true_y = [0xf5ba2404, 0xf54fcd43, 0xcbcc4b1a, 0xe796aadf]

    assert(res_x == true_x)
    assert(res_y == true_y)

def test_traxm():
    for _ in range(NB_TESTS//10):
        n = random.choice([16, 32, 64, 128])
        x = [random.getrandbits(n//4) for _ in range(2)]
        y = [random.getrandbits(n//4) for _ in range(2)]
        round_key = [random.getrandbits(n//4) for _ in range(4)]

        X = [QuantumRegister(n//4, name=f'X{i}') for i in range(2)]
        Y = [QuantumRegister(n//4, name=f'Y{i}') for i in range(2)]
        K = [QuantumRegister(n//4, name=f'K{i}') for i in range(4)]
        ancillas = AncillaRegister(Traxm_enc_round.get_num_ancilla_qubits(n))

        qc = rCircuit(*X, *Y, *K, ancillas)
        gate = Traxm_enc_round(X, Y, K, None, ancillas)
        qc.append(gate, list(chain(*gate.inputs)))

        result = circuit_test(qc, lambda *params: chain.from_iterable(c_traxm_enc(list(params[0:2]), list(params[2:4]), list(params[4::]), None, n=n//4, nsteps=1, final_addition=False)), x+y+round_key, gate.inputs[:-1], gate.outputs[:-3])

        assert(result)
        
def test_traxl():
    # This test is really slow so we must run it less times
    for _ in range(1):
        n = random.choice([16, 32, 64, 128, 256])
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

if __name__ == "__main__":
    test_ripple_add()
