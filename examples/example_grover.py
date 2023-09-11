# This program builds a quantum circuit to do a Grover search on TRAX
from qiskit_algorithms import AmplificationProblem, Grover
from qiskit.primitives import Sampler
from qiskit import QuantumRegister, AncillaRegister
from pyqrypto.rOperations import make_circuit, rCircuit, rPrepare
from qiskit.circuit.library import ZGate, GroverOperator
from itertools import chain
from pyqrypto.sparkle import Traxl_enc, c_traxl_genkeys, c_traxl_enc
import random
random.seed(42)

def grover_on_trax(simulate:bool=False) -> AmplificationProblem:
    """Construct a circuit that performs a grover search on TRAX with a known plaintext/ciphertext pair.

    :param simulate: Build a toy version of the circuit and simulate it.
    :returns: The circuit as an AmplificationProblem.
    """

    # Set TRAX size, key and tweak
    n = 16
    key = [random.getrandbits(n//8) for _ in range(8)]
    tweak = [random.getrandbits(n//8) for _ in range(4)]

    # Assume we pocess a plaintext/ciphertext pair
    plaintext_x = [random.getrandbits(n//8) for _ in range(4)]
    plaintext_y = [random.getrandbits(n//8) for _ in range(4)]
    subkeys = c_traxl_genkeys(key, n=n//8)
    ciphertext_x, ciphertext_y = c_traxl_enc(plaintext_x, plaintext_y, subkeys, tweak, n=n//8)
    print(f"ciphertext_x: {list(map(hex, ciphertext_x))}")
    print(f"ciphertext_y: {list(map(hex, ciphertext_y))}")

    # Declare quantum registers
    X = [QuantumRegister(n//8, name=f'X{i}') for i in range(4)]
    Y = [QuantumRegister(n//8, name=f'Y{i}') for i in range(4)]
    K = [QuantumRegister(n//8, name=f'K{i}') for i in range(8)]
    ancillas = AncillaRegister(Traxl_enc.get_num_ancilla_qubits(n))

    # Build Grover oracle
    oracle = rCircuit(*X, *Y, *K, ancillas)

    # Prepare X and Y with plaintext
    for i in range(4):
        oracle.compose(rPrepare(X[i], plaintext_x[i]), X[i], inplace=True)
        oracle.compose(rPrepare(Y[i], plaintext_y[i]), Y[i], inplace=True)
    if simulate:
        # Simplify the problem by setting all of the key except n//8 bits
        for i in range(1, 8):
            oracle.compose(rPrepare(K[i], key[i]), K[i], inplace=True)

    # Encrypt X and Y using key K
    gate = Traxl_enc(X, Y, K, tweak, ancillas)
    oracle.append(gate, list(chain(*gate.inputs)))

    # Flip the phase of the solution when all X and Y are 1
    # This is a multi-controlled mixed-polarity Z gate
    for i in range(4):
        # X and Y are all 1 if and only if they're equal to the ciphertext
        oracle.xor(X[i], ciphertext_x[i]^(2**(n//8)-1))
        oracle.xor(Y[i], ciphertext_y[i]^(2**(n//8)-1))
    if simulate:
        # Only test the first n//8 bits of the ciphertext for the simulation
        mcz = ZGate().control(len(list(chain(X[0])))-1)
        oracle.append(mcz, list(chain(X[0])))
    else:
        mcz = ZGate().control(len(list(chain(*X, *Y)))-1)
        oracle.append(mcz, list(chain(*X, *Y)))
    for i in range(4):
        oracle.xor(X[i], ciphertext_x[i]^(2**(n//8)-1))
        oracle.xor(Y[i], ciphertext_y[i]^(2**(n//8)-1))
    
    # Undo the encryption of the plaintext
    oracle.append(gate.inverse(), list(chain(*gate.inputs)))

    # Build the Grover operator
    if simulate:
        grover_operator = GroverOperator(oracle, reflection_qubits=list(map(lambda q: oracle.find_bit(q)[0], chain(K[0]))))
    else:
        grover_operator = GroverOperator(oracle, reflection_qubits=list(map(lambda q: oracle.find_bit(q)[0], chain(*K))))

    print(grover_operator)
    circuit = make_circuit(gro

    # problem = AmplificationProblem(oracle, grover_operator=grover_operator)
    #
    # print(problem.grover_operator.decompose())
    #
    # if simulate:
    #     # Run Grover
    #     grover = Grover(sampler=Sampler())
    #     result = grover.amplify(problem)
    #
    #     print(result)
    #
    # return problem

grover_on_trax(simulate=True)
