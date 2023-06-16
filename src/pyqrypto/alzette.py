"""A reversible quantum implementation of n qubits Alzette from [Alzette2020]_.

.. [Alzette2020] Beierle, C., Biryukov, A., Cardoso dos Santos, L., Großschädl, J., Perrin, L., Udovenko, A., ... & Wang, Q. (2020). Alzette: A 64-Bit ARX-box: (Feat. CRAX and TRAX). In Advances in Cryptology–CRYPTO 2020: 40th Annual International Cryptology Conference, CRYPTO 2020, Santa Barbara, CA, USA, August 17–21, 2020, Proceedings, Part III 40 (pp. 419-448). Springer International Publishing.
"""
from pyqrypto.rOperations import rCircuit, rOperation, make_circuit, run_circuit
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit import Gate
from typing import Optional, List
from itertools import chain
import json

def ror(x: int, r: int, n: int) -> int:
    """Classical implementation of the right rotation.

    :param x: The integer to rotate
    :param r: The rotation amount
    :param n: The number of bits on which x is encoded.
    :returns: The integer x rotated right by r bits.
    """
    r = r % n
    return ((x >> r) | (x << (n - r))) & ((2**n) - 1)

def alzette(x: int, y: int, c: int, n: int) -> List[int]:
    """Classical software implementation of Alzette.

    :param x: The integer value of x.
    :param y: The integer value of y.
    :param c: The c constant in Alzette.
    :param n: The number of bits on which x, y and n are encoded.
    :returns: alzette(x, y)
    """
    x = (x + ror(y, 31, n)) % 2**n
    y = y ^ ror(x, 24, n)
    x = x ^ c
    x = (x + ror(y, 17, n)) % 2**n
    y = y ^ ror(x, 17, n)
    x = x ^ c
    x = (x + y) % 2**n
    y = y ^ ror(x, 31, n)
    x = x ^ c
    x = (x + ror(y, 24, n)) % 2**n
    y = y ^ ror(x, 16, n)
    x = x ^ c
    return [x, y]

class Alzette(Gate, rOperation):
    r"""A quantum gate implementing the ARX-box Alzette of two vectors of qubits.
    :param X: X vector.
    :param Y: Y vector.
    :param c: Alzette constant.
    :raises CiruitError: If X and Y have a different size.

    :operation: :math:`(X, Y) \leftarrow \mathrm{Alzette}(X, Y)`

    :math:`\mathrm{Alzette(X, Y)}` is defined in [Alzette2020]_ as the following operations:

    .. math::
        \begin{align*}
            & \mathrm{X} \leftarrow \mathrm{X} + (\mathrm{Y} \ggg 31) \\
            & \mathrm{Y} \leftarrow \mathrm{Y} \oplus (\mathrm{X} \ggg 24) \\
            & \mathrm{X} \leftarrow \mathrm{X} \oplus c \\
            & \mathrm{X} \leftarrow \mathrm{X} + (\mathrm{Y} \ggg 17) \\
            & \mathrm{Y} \leftarrow \mathrm{Y} \oplus (\mathrm{X} \ggg 17) \\
            & \mathrm{X} \leftarrow \mathrm{X} \oplus c \\
            & \mathrm{X} \leftarrow \mathrm{X} + (\mathrm{Y} \ggg 0) \\
            & \mathrm{Y} \leftarrow \mathrm{Y} \oplus (\mathrm{X} \ggg 31) \\
            & \mathrm{X} \leftarrow \mathrm{X} \oplus c \\
            & \mathrm{X} \leftarrow \mathrm{X} + (\mathrm{Y} \ggg 24) \\
            & \mathrm{Y} \leftarrow \mathrm{Y} \oplus (\mathrm{X} \ggg 16) \\
            & \mathrm{X} \leftarrow \mathrm{X} \oplus c
        \end{align*}
    """
    def __init__(self, X: QuantumRegister, Y: QuantumRegister, c: int, label: Optional[str] = None):
        if len(X) != len(Y):
            raise CircuitError("X and Y must have the same size.")
        self.n = len(X)
        self.c = c
        self.inputs = [X, Y]
        super().__init__("Alzette", self.n*2, [], label=label)

        qc = rCircuit(X, Y, name='Alzette')

        X = qc.add(X, qc.ror(Y, 31))
        Y = qc.xor(Y, qc.ror(X, 24))
        X = qc.xor(X, self.c)
        X = qc.add(X, qc.ror(Y, 17))
        Y = qc.xor(Y, qc.ror(X, 17))
        X = qc.xor(X, self.c)
        X = qc.add(X, Y)
        Y = qc.xor(Y, qc.ror(X, 31))
        X = qc.xor(X, self.c)
        X = qc.add(X, qc.ror(Y, 24))
        Y = qc.xor(Y, qc.ror(X, 16))
        X = qc.xor(X, self.c)

        self._definition = qc
        self.outputs = [X, Y]

if __name__ == '__main__':

    n = 32
    X = QuantumRegister(32, name='X')
    Y = QuantumRegister(32, name='Y')
    a = 384973
    b = 1238444859
    c = 0xb7e15162
    print(f"Running alzette_{c}({a}, {b})")


    qc = QuantumCircuit(X, Y)
    gate = Alzette(X, Y, c)
    qc.append(gate, list(chain(*gate.outputs)))

    decomposition_reps = 2
    circuit_depth = qc.decompose(reps=decomposition_reps).depth()
    gate_counts = qc.decompose(reps=decomposition_reps).count_ops()

    final_circuit = make_circuit(qc, [a, b], [X, Y], [X, Y])

    final_circuit.decompose(reps=4).qasm(filename='alzette.qasm')

    result = run_circuit(final_circuit)

    print(f"Classical result: {alzette(a, b, c, n)}")
    print(f"Quantum simulated result: {result}")
    print("Circuit depth:", circuit_depth)
    print("Circuit gate count:", json.dumps(gate_counts))
