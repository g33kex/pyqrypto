# These examples showcase how to use the different adders in pyqrypto
from pyqrypto.rOperations import rCircuit, make_circuit, run_circuit, rDKRSCarryLookaheadAdder, rOperation
from qiskit import QuantumRegister, AncillaRegister
import matplotlib.pyplot as plt
import random
random.seed(42)

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

def showcase_dkrs_adder():
    n = 5
    a = random.getrandbits(n)
    b = random.getrandbits(n)
    # Let's put the wires in the same order as the DKRS2004 paper
    wire_order = [0,5,10,1,6,11,2,7,14,12,3,8,13,4,9]

    print(f"Doing operation {a}+{b}={a+b}")

    # Instanciate registers
    A = QuantumRegister(n, name='A')
    B = QuantumRegister(n, name='B')
    # Instanciate enough ancilla qubits
    ancillas = AncillaRegister(rDKRSCarryLookaheadAdder.get_num_ancilla_qubits(n))

    # Build circuit
    qc = rCircuit(A, B, ancillas)
    qc.add(A, B, mode='ripple')

    # Print statistics about the circuit
    print("Circuit stats:", qc.stats)

    # Show circuit as matplotlib graph
    fig = qc.decompose(gates_to_decompose=[rOperation], reps=2).draw(wire_order=wire_order,output='mpl')
    plt.show()

    # Add initialization and measurements
    final_circuit = make_circuit(qc, [a, b], [A, B], [A])

    # Run the circuit
    result = run_circuit(final_circuit)

    print("Result:", result[0])

if __name__ == '__main__':
    showcase_dkrs_adder()
