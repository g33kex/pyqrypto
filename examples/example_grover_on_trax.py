"""Example of a quantum circuit to do a Grover search on TRAX-L-17."""
from __future__ import annotations

import random
from itertools import chain

from pyqrypto.register_operations import RegisterCircuit, RegisterPrepare
from pyqrypto.sparkle import TraxlEnc, c_traxl_enc, c_traxl_genkeys
from qiskit import AncillaRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import GroverOperator, ZGate
from qiskit_algorithms import AmplificationProblem, Grover

random.seed(42)


def grover_oracle(
    n: int,
    X: list[QuantumRegister],
    Y: list[QuantumRegister],
    K: list[QuantumRegister],
    ancillas: AncillaRegister,
    tweak: list[int],
    ciphertext_x: list[int],
    ciphertext_y: list[int],
) -> RegisterCircuit:
    """Build a phase oracle that encrypts the plaintext and compares it with a given ciphertext.

    The plaintext stored in X and Y is encrypted and compared to an expected ciphertext. The phase
    of the quantum state corresponding to the key that gives the expected ciphertext is flipped.

    :param n: Size of the key
    :param X: A list of 4 quantum registers of size n/8 storing the first half of the plaintext.
    :param Y: A list of 4 quantum registers of size n/8 storing the second half of the plaintext.
    :param K: A list of 8 quantum registers of size n/8 to storing a superposition of all keys.
    :param ancillas: The ancillas qubits needed for the computation.
    :param tweak: The tweak, a list of 4 integers on n/8 bits.
    :ciphertext_x: A n/2-bit integer corresponding to the first half of the expected ciphertext.
    :ciphertext_y: A n/2-bit integer corresponding to the second half of the expected ciphertext.
    """
    oracle = RegisterCircuit(*X, *Y, *K, ancillas)

    # Encrypt X and Y using key K
    gate = TraxlEnc(X, Y, K, tweak, ancillas)
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


def grover_on_trax(n: int = 256, iterations: int | None = None) -> QuantumCircuit:
    """Construct a circuit performing a grover search on TRAX.

    This function randomly generates a key and a known plaintext/ciphertext pair.

    :param n: Size of the key.
    :param iterations: Number of iterations of Grover. If None use the optimal number of iterations.
    :returns: The circuit performing the search.
    """
    # Set key and tweak
    key = [random.getrandbits(n // 8) for _ in range(8)]
    tweak = [random.getrandbits(n // 8) for _ in range(4)]

    # Assume we possess a plaintext/ciphertext pair
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
    ancillas = AncillaRegister(TraxlEnc.get_num_ancilla_qubits(n))

    # Build Grover oracle
    oracle = grover_oracle(n, X, Y, K, ancillas, tweak, ciphertext_x, ciphertext_y)

    # State preparation
    state_preparation = RegisterCircuit(*X, *Y, *K, ancillas)

    # Prepare X and Y with plaintext
    for i in range(4):
        state_preparation.compose(RegisterPrepare(X[i], plaintext_x[i]), X[i], inplace=True)
        state_preparation.compose(RegisterPrepare(Y[i], plaintext_y[i]), Y[i], inplace=True)

    # Put key in uniform superposition
    for qubit in chain(*K):
        state_preparation.h(qubit)

    # Build the Grover operator
    grover_operator = GroverOperator(
        oracle,
        reflection_qubits=list(map(lambda q: oracle.find_bit(q)[0], chain(*K))),
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
