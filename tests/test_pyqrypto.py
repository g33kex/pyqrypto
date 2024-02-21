"""Test pyqrypto."""
from __future__ import annotations

import random
from itertools import chain
from typing import Callable

from pyqrypto.register_operations import (
    RegisterCircuit,
    RegisterConstantDKRSCarryLookaheadAdder,
    RegisterDKRSCarryLookaheadAdder,
    make_circuit,
    run_circuit,
)
from pyqrypto.sparkle import (
    Alzette,
    TraxlEnc,
    TraxmEncRound,
    TraxsEncRound,
    c_alzette,
    c_traxl_enc,
    c_traxl_genkeys,
    c_traxm_enc,
    c_traxs_enc,
)
from qiskit import AncillaRegister, QuantumCircuit, QuantumRegister

NB_TESTS = 100
"""Number of times to run each test"""
DEVICE = "CPU"
"""Device for the tests"""
METHOD = "matrix_product_state"
"""Simulation method"""


# Utils
def rol(x: int, r: int, n: int) -> int:
    """Left rotation."""
    r = r % n
    return ((x << r) | (x >> (n - r))) & ((2**n) - 1)


def ror(x: int, r: int, n: int) -> int:
    """Right rotation."""
    r = r % n
    return ((x >> r) | (x << (n - r))) & ((2**n) - 1)


# Generic circuit test
def circuit_test(
    circuit: RegisterCircuit,
    classical_function: Callable,
    inputs: list[int],
    input_registers: list[QuantumRegister],
    output_registers: list[QuantumRegister],
    verbose=False,
) -> bool:
    """Test if a circuit's output corresponds to the classical function's output."""
    true_result = classical_function(*inputs)

    if verbose:
        print("Testing the following gate:")
        print(circuit.decompose(reps=1))

    final_circuit = make_circuit(circuit, inputs, input_registers, output_registers)
    if verbose:
        print("Assembling it into the following circuit:")
        print(final_circuit)
    result = run_circuit(final_circuit, device=DEVICE, method=METHOD)

    if verbose:
        print("True result:\t", true_result)
        print("Actual result:\t", result)

    if true_result != result:
        return False
    return True


# Specific tests
def test_prepare() -> None:
    """Test circuit preparation."""
    for _ in range(NB_TESTS):
        n_1 = random.randint(1, 10)
        n_2 = random.randint(1, 10)
        a = random.getrandbits(n_1)
        b = random.getrandbits(n_2)

        X = QuantumRegister(n_1)
        Y = QuantumRegister(n_2)

        circuit = QuantumCircuit(X, Y)

        result = circuit_test(circuit, lambda x, y: [x, y], [a, b], [X, Y], [X, Y])

        assert result


def test_ror() -> None:
    """Test right rotation."""
    for _ in range(NB_TESTS):
        n = random.randint(1, 10)
        a = random.getrandbits(n)
        r = random.randint(1, 3 * n)

        X1 = QuantumRegister(n)

        circuit = RegisterCircuit(X1)
        X2 = circuit.ror(X1, r)

        result = circuit_test(circuit, lambda x: [ror(x, r, n)], [a], [X1], [X2])

        assert result


def test_xor() -> None:
    """Test xor between two quantum registers."""
    for _ in range(NB_TESTS):
        n = random.randint(1, 10)
        a = random.getrandbits(n)
        b = random.getrandbits(n)

        X = QuantumRegister(n)
        Y = QuantumRegister(n)

        circuit = RegisterCircuit(X, Y)
        circuit.xor(X, Y)

        result = circuit_test(circuit, lambda x, y: [x ^ y], [a, b], [X, Y], [X])

        assert result


def test_xorc() -> None:
    """Test xor with a constant."""
    for _ in range(NB_TESTS):
        n = random.randint(1, 10)
        a = random.getrandbits(n)
        c = random.getrandbits(n)

        X1 = QuantumRegister(n)

        circuit = RegisterCircuit(X1)
        X2 = circuit.xor(X1, c)

        result = circuit_test(circuit, lambda x: [x ^ c], [a], [X1], [X2])

        assert result


def test_rorrorxor() -> None:
    """Test chaining ror, ror and xor."""
    for _ in range(NB_TESTS):
        n = random.randint(1, 10)
        a = random.getrandbits(n)
        b = random.getrandbits(n)
        r1 = random.randint(1, n * 3)
        r2 = random.randint(1, n * 3)

        X1 = QuantumRegister(n)
        Y1 = QuantumRegister(n)

        circuit = RegisterCircuit(X1, Y1)
        X2 = circuit.ror(X1, r1)
        Y2 = circuit.ror(Y1, r2)
        circuit.xor(X2, Y2)

        result = circuit_test(
            circuit,
            lambda x, y: [ror(x, r1, n) ^ ror(y, r2, n), y],
            [a, b],
            [X1, Y1],
            [X2, Y1],
        )

        assert result


def test_rorxorrolxor() -> None:
    """Test chaining ror, xor, rol and xor."""
    for _ in range(NB_TESTS):
        n = random.randint(1, 10)
        a = random.getrandbits(n)
        b = random.getrandbits(n)
        r1 = random.randint(1, n * 3)
        r2 = random.randint(1, n * 3)

        X1 = QuantumRegister(n)
        Y1 = QuantumRegister(n)

        circuit = RegisterCircuit(X1, Y1)
        X2 = circuit.xor(circuit.ror(X1, r1), Y1)
        Y2 = circuit.rol(Y1, r2)
        circuit.xor(X2, Y2)

        result = circuit_test(
            circuit,
            lambda x, y: [ror(x, r1, n) ^ y ^ rol(y, r2, n), rol(y, r2, n)],
            [a, b],
            [X1, Y1],
            [X2, Y2],
        )

        assert result


def test_complexxor() -> None:
    """Test chaining ror and xor with 3 registers."""
    for _ in range(NB_TESTS):
        n = random.randint(1, 10)
        a = random.getrandbits(n)
        b = random.getrandbits(n)
        c = random.getrandbits(n)
        r1 = random.randint(0, n * 3)
        r2 = random.randint(0, n * 3)
        r3 = random.randint(0, n * 3)

        X1 = QuantumRegister(n, name="X")
        Y1 = QuantumRegister(n, name="Y")
        Z = QuantumRegister(n, name="Z")

        circuit = RegisterCircuit(X1, Y1, Z)
        X2 = circuit.ror(X1, r1)
        circuit.xor(X2, Y1)
        Y2 = circuit.ror(Y1, r2)
        circuit.xor(Y2, Z)
        circuit.xor(Z, circuit.rol(X2, r3))

        result = circuit_test(
            circuit,
            lambda x, y, z: [
                ror(x, r1, n) ^ y,
                ror(y, r2, n) ^ z,
                z ^ rol(ror(x, r1, n) ^ y, r3, n),
            ],
            [a, b, c],
            [X1, Y1, Z],
            [X2, Y2, Z],
        )

        assert result


def test_ripple_add() -> None:
    """Test the ripple-carry adder."""
    for _ in range(NB_TESTS):
        n = random.randint(1, 10)
        a = random.getrandbits(n)
        b = random.getrandbits(n)

        X1 = QuantumRegister(n, name="X")
        Y = QuantumRegister(n, name="Y")

        circuit = RegisterCircuit(X1, Y)

        X2 = circuit.add(X1, Y, mode="ripple")

        result = circuit_test(circuit, lambda x, y: [(x + y) % (2**n), y], [a, b], [X1, Y], [X2, Y])

        assert result


def test_lookahead_add():
    """Test the carry-lookahead adder."""
    for _ in range(NB_TESTS // 5):
        n = random.randint(1, 32)
        a = random.getrandbits(n)
        b = random.getrandbits(n)

        A = QuantumRegister(n, name="A")
        B = QuantumRegister(n, name="B")
        ancillas = AncillaRegister(RegisterDKRSCarryLookaheadAdder.get_num_ancilla_qubits(n))

        circuit = RegisterCircuit(A, B, ancillas)
        circuit.add(A, B, ancillas, mode="lookahead")

        if ancillas:
            result = circuit_test(
                circuit,
                lambda x, y, _: [(x + y) % (2**n), y, 0],
                [a, b, 0],
                [A, B, ancillas],
                [A, B, ancillas],
                verbose=False,
            )
        else:
            result = circuit_test(
                circuit,
                lambda x, y: [(x + y) % (2**n), y],
                [a, b],
                [A, B],
                [A, B],
                verbose=False,
            )

        assert result


def test_addc() -> None:
    """Test the adder with a constant."""
    for _ in range(NB_TESTS):
        n = random.randint(1, 10)
        a = random.getrandbits(n)
        c = random.getrandbits(n)
        ancillas = AncillaRegister(
            RegisterConstantDKRSCarryLookaheadAdder.get_num_ancilla_qubits(n),
        )

        X1 = QuantumRegister(n)

        circuit = RegisterCircuit(X1, ancillas)
        X2 = circuit.add(X1, c, ancillas)

        if ancillas:
            result = circuit_test(
                circuit,
                lambda x, _: [(x + c) % 2**n, 0],
                [a, 0],
                [X1, ancillas],
                [X2, ancillas],
            )
        else:
            result = circuit_test(circuit, lambda x: [(x + c) % 2**n], [a], [X1], [X2])

        assert result


def test_alzette() -> None:
    """Test Alzette against the classical implementation."""
    for _ in range(NB_TESTS):
        n = random.randint(1, 10)
        a = random.getrandbits(n)
        b = random.getrandbits(n)
        c = random.getrandbits(n)

        X = QuantumRegister(n)
        Y = QuantumRegister(n)

        circuit = QuantumCircuit(X, Y)
        gate = Alzette(X, Y, c)
        circuit.append(gate, chain(*gate.outputs))

        result = circuit_test(circuit, lambda x, y: c_alzette(x, y, c, n), [a, b], [X, Y], [X, Y])

        assert result


def test_ctraxl() -> None:
    """Test the classical TRAX-L against values pre-computed with the C reference implementation."""
    x = [0xAEA9F4E8, 0xC926A22F, 0x9D1078F8, 0x8A779F98]
    y = [0x24E002DC, 0x8F34F225, 0x13FF3742, 0x510E85EA]

    key = [
        0x3B9C0BB1,
        0xCC2106FB,
        0x28BC5755,
        0xB146DC0F,
        0xE111AAD7,
        0xCA29EEA5,
        0x612EEF46,
        0x5ACE7BC,
    ]
    tweak = [0x7C856A02, 0x62E011B3, 0xC016150F, 0xC0045AE6]

    subkeys = c_traxl_genkeys(key, n=32)
    res_x, res_y = c_traxl_enc(x, y, subkeys, tweak, n=32)
    true_x = [0xEF5EABC9, 0xC3DEEC85, 0x3BE9C4FD, 0x7DB95C88]
    true_y = [0xF5BA2404, 0xF54FCD43, 0xCBCC4B1A, 0xE796AADF]

    assert res_x == true_x
    assert res_y == true_y


def test_traxs() -> None:
    """Test TRAX-S against the classical implementation."""
    for _ in range(NB_TESTS // 10):
        n = random.choice([16, 32, 64, 128])
        x = random.getrandbits(n // 2)
        y = random.getrandbits(n // 2)
        round_key = [random.getrandbits(n // 2) for _ in range(2)]

        X = QuantumRegister(n // 2, name="X")
        Y = QuantumRegister(n // 2, name="Y")
        K = [QuantumRegister(n // 2, name=f"K{i}") for i in range(2)]
        ancillas = AncillaRegister(TraxsEncRound.get_num_ancilla_qubits(n))

        circuit = RegisterCircuit(X, Y, *K, ancillas)
        gate = TraxsEncRound(X, Y, K, None, ancillas)
        circuit.append(gate, list(chain(*gate.inputs)))

        result = circuit_test(
            circuit,
            lambda *params: list(
                c_traxs_enc(
                    params[0],
                    params[1],
                    list(params[2::]),
                    None,
                    n=n // 2,
                    nsteps=1,
                    final_addition=False,
                )
            ),
            [x, y, *round_key],
            gate.inputs[:-1],
            gate.outputs[:-3],
        )

        assert result


def test_traxm() -> None:
    """Test TRAX-M against the classical implementation."""
    for _ in range(NB_TESTS // 10):
        n = random.choice([16, 32, 64, 128])
        x = [random.getrandbits(n // 4) for _ in range(2)]
        y = [random.getrandbits(n // 4) for _ in range(2)]
        round_key = [random.getrandbits(n // 4) for _ in range(4)]

        X = [QuantumRegister(n // 4, name=f"X{i}") for i in range(2)]
        Y = [QuantumRegister(n // 4, name=f"Y{i}") for i in range(2)]
        K = [QuantumRegister(n // 4, name=f"K{i}") for i in range(4)]
        ancillas = AncillaRegister(TraxmEncRound.get_num_ancilla_qubits(n))

        circuit = RegisterCircuit(*X, *Y, *K, ancillas)
        gate = TraxmEncRound(X, Y, K, None, ancillas)
        circuit.append(gate, list(chain(*gate.inputs)))

        result = circuit_test(
            circuit,
            lambda *params: list(
                chain.from_iterable(
                    c_traxm_enc(
                        list(params[0:2]),
                        list(params[2:4]),
                        list(params[4::]),
                        None,
                        n=n // 4,
                        nsteps=1,
                        final_addition=False,
                    ),
                )
            ),
            x + y + round_key,
            gate.inputs[:-1],
            gate.outputs[:-5],
            verbose=False,
        )

        assert result


def test_traxl() -> None:
    """Test TRAX-L against the classical implementation."""
    # This test is really slow so we must run it less times
    for _ in range(1):
        n = random.choice([16, 32, 64, 128, 256])
        x = [random.getrandbits(n // 8) for _ in range(4)]
        y = [random.getrandbits(n // 8) for _ in range(4)]
        tweak = [random.getrandbits(n // 8) for _ in range(4)]
        key = [random.getrandbits(n // 8) for _ in range(8)]

        X = [QuantumRegister(n // 8, name=f"X{i}") for i in range(4)]
        Y = [QuantumRegister(n // 8, name=f"Y{i}") for i in range(4)]
        K = [QuantumRegister(n // 8, name=f"K{i}") for i in range(8)]
        ancillas = AncillaRegister(TraxlEnc.get_num_ancilla_qubits(n))

        circuit = RegisterCircuit(*X, *Y, *K, ancillas)
        gate = TraxlEnc(X, Y, K, tweak, ancillas)
        circuit.append(gate, list(chain(*gate.inputs)))

        result = circuit_test(
            circuit,
            lambda *params: [
                a
                for sublist in c_traxl_enc(
                    list(params[0:4]),
                    list(params[4:8]),
                    c_traxl_genkeys(list(params[8::]), n=n // 8),
                    tweak,
                    n=n // 8,
                )
                for a in sublist
            ],
            x + y + key,
            gate.inputs[:-1],
            gate.outputs[:-9],
        )

        assert result
