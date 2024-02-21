"""These examples showcase how to use the different adders in pyqrypto"""
import random
from pyqrypto.register_operations import (
    RegisterCircuit,
    make_circuit,
    run_circuit,
    RegisterDKRSCarryLookaheadAdder,
    RegisterOperation,
)
from qiskit import QuantumRegister, AncillaRegister, QuantumCircuit

random.seed(42)


def example_add() -> None:
    print("\nExample usage of the add operation:\n")
    # Let's create two 8 bit quantum registers
    X1 = QuantumRegister(8, name="X")
    Y = QuantumRegister(8, name="Y")

    qc = RegisterCircuit(X1, Y)

    # X2 <- X1 + Y
    # By default, the adder used is a ripple-carry adder
    X2 = qc.add(X1, Y)

    # Print the circuit
    print(qc.decompose(reps=2))

    # Let's test it by adding 74 and 42
    final_circuit = make_circuit(qc, [74, 42], [X1, Y], [X2])

    result = run_circuit(final_circuit)

    # 74+42 = 116
    print("Result:", result)


def example_dkrs_adder():
    print("\nExample usage of the carry-lookahead adder:\n")
    n = 32
    a = random.getrandbits(n)
    b = random.getrandbits(n)

    print(f"Performing operation {a}+{b}={a+b}")

    # Instanciate registers
    A = QuantumRegister(n, name="A")
    B = QuantumRegister(n, name="B")
    # Instanciate enough ancilla qubits
    ancillas = AncillaRegister(RegisterDKRSCarryLookaheadAdder.get_num_ancilla_qubits(n))

    # Build circuit
    qc = RegisterCircuit(A, B, ancillas)
    qc.add(A, B, ancillas=ancillas, mode="lookahead")

    # Print statistics about the circuit
    print("Circuit stats:", qc.stats)

    # Add initialization and measurements
    final_circuit = make_circuit(qc, [a, b], [A, B], [A])

    # Run the circuit
    result = run_circuit(final_circuit, method="matrix_product_state")

    print("Result:", result[0])


if __name__ == "__main__":
    example_add()
    example_dkrs_adder()
