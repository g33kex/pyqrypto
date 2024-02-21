"""This program builds a quantum circuit to do a Grover search on TRAX."""
from qiskit_algorithms import AmplificationProblem, Grover
from qiskit import QuantumRegister, AncillaRegister, QuantumCircuit
from pyqrypto.rOperations import make_circuit, rCircuit, rPrepare
from qiskit.circuit.library import ZGate, GroverOperator
from itertools import chain
from pyqrypto.sparkle import Traxl_enc, c_traxl_genkeys, c_traxl_enc
import random
from math import pi, ceil, sqrt

random.seed(42)


def grover_oracle(n, X, Y, K, ancillas, tweak, ciphertext_x, ciphertext_y):
    """Build a phase oracle that encrypts the plaintext and compares it with a given ciphertext."""
    oracle = rCircuit(*X, *Y, *K, ancillas)

    # Encrypt X and Y using key K
    gate = Traxl_enc(X, Y, K, tweak, ancillas)
    oracle.append(gate, list(chain(*gate.inputs)))

    # Flip the phase of the solution when all X and Y are 1
    # This is a multi-controlled mixed-polarity Z gate
    for i in range(4):
        # X and Y are all 1 if and only if they're equal to the ciphertext
        oracle.xor(X[i], ciphertext_x[i] ^ (2 ** (n // 8) - 1))
        oracle.xor(Y[i], ciphertext_y[i] ^ (2 ** (n // 8) - 1))
    mcz = ZGate().control(len(list(chain(*X, *Y))) - 1)
    oracle.append(mcz, list(chain(*X, *Y)))
    for i in range(4):
        oracle.xor(X[i], ciphertext_x[i] ^ (2 ** (n // 8) - 1))
        oracle.xor(Y[i], ciphertext_y[i] ^ (2 ** (n // 8) - 1))

    # Undo the encryption of the plaintext
    oracle.append(gate.inverse(), list(chain(*gate.inputs)))

    return oracle


def grover_on_trax(n=256, iterations=None) -> QuantumCircuit:
    """Construct a circuit that performs a grover search on TRAX with a known plaintext/ciphertext pair.

    :param n: Size of the key.
    :param iterations: Number of iterations of Grover, if None uses the optimal number of iterations.
    :returns: The circuit performing the search.
    """
    # Set key and tweak
    key = [random.getrandbits(n // 8) for _ in range(8)]
    tweak = [random.getrandbits(n // 8) for _ in range(4)]

    # Assume we pocess a plaintext/ciphertext pair
    plaintext_x = [random.getrandbits(n // 8) for _ in range(4)]
    plaintext_y = [random.getrandbits(n // 8) for _ in range(4)]
    subkeys = c_traxl_genkeys(key, n=n // 8)
    ciphertext_x, ciphertext_y = c_traxl_enc(plaintext_x, plaintext_y, subkeys, tweak, n=n // 8)
    print(f"ciphertext_x: {list(map(hex, ciphertext_x))}")
    print(f"ciphertext_y: {list(map(hex, ciphertext_y))}")

    # Declare quantum registers
    X = [QuantumRegister(n // 8, name=f"X{i}") for i in range(4)]
    Y = [QuantumRegister(n // 8, name=f"Y{i}") for i in range(4)]
    K = [QuantumRegister(n // 8, name=f"K{i}") for i in range(8)]
    ancillas = AncillaRegister(Traxl_enc.get_num_ancilla_qubits(n))

    # Build Grover oracle
    oracle = grover_oracle(n, X, Y, K, ancillas, tweak, ciphertext_x, ciphertext_y)

    # State preparation
    state_preparation = rCircuit(*X, *Y, *K, ancillas)

    # Prepare X and Y with plaintext
    for i in range(4):
        state_preparation.compose(rPrepare(X[i], plaintext_x[i]), X[i], inplace=True)
        state_preparation.compose(rPrepare(Y[i], plaintext_y[i]), Y[i], inplace=True)

    # Put key in uniform superposition
    for qubit in chain(*K):
        state_preparation.h(qubit)

    # Build the Grover operator
    grover_operator = GroverOperator(
        oracle, reflection_qubits=list(map(lambda q: oracle.find_bit(q)[0], chain(*K)))
    )

    print(grover_operator.decompose())

    problem = AmplificationProblem(
        oracle,
        state_preparation=state_preparation,
        grover_operator=grover_operator,
        objective_qubits=list(map(lambda q: oracle.find_bit(q)[0], chain(*K))),
    )

    if iterations is None:
        iterations = Grover.optimal_num_iterations(1, n)

    circuit = Grover().construct_circuit(problem, iterations, measurement=True)

    return circuit


if __name__ == "__main__":
    circuit = grover_on_trax(16, 3)
    print(circuit.decompose(reps=2))
