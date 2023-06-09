from qiskit import QuantumCircuit, QuantumRegister
import matplotlib.pyplot as plt

from bitwise_operations import ROR, ROL, bXOR, bAppend, make_circuit, run_circuit

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

def rol(x, r, n):
    r = r%n
    return ((x << r) | (x >> (n - r))) & ((2**n)-1)

def ror(x, r, n):
    r = r % n
    return ((x >> r) | (x << (n - r))) & ((2**n) - 1)

if __name__ == '__main__':
    n=4
    r1 = 1
    r2 = 2

    X1 = QuantumRegister(n, name='X')
    Y1 = QuantumRegister(n, name='Y')
    Z = QuantumRegister(n, name='Z')

    qc = QuantumCircuit(X1, Y1, Z)
    X2 = ROR(X1, r1)

    bAppend(qc, bXOR(n), [X2, Y1])
    Y2 = ROR(Y1, r2)
    bAppend(qc, bXOR(n), [Y2, Z])

    result = test_circuit(qc, lambda x,y,z: [ror(x, r1, n)^y, ror(y, r2, n)^z, z], [9, 11, 2], [X1, Y1, Z], [X2, Y2, Z], verbose=True)

    if result:
        print("Test passed!")
    else:
        print("Test failed!")
    
