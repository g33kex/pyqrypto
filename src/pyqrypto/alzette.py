"""A reversible quantum implementation of n qubits Alzette from [Alzette2020]_.

.. [Alzette2020] Beierle, C., Biryukov, A., Cardoso dos Santos, L., Großschädl, J., Perrin, L., Udovenko, A., ... & Wang, Q. (2020). Alzette: A 64-Bit ARX-box: (Feat. CRAX and TRAX). In Advances in Cryptology–CRYPTO 2020: 40th Annual International Cryptology Conference, CRYPTO 2020, Santa Barbara, CA, USA, August 17–21, 2020, Proceedings, Part III 40 (pp. 419-448). Springer International Publishing.
"""
from pyqrypto.rOperations import rCircuit, rOperation, make_circuit, run_circuit
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit import Gate
from typing import Optional, List, Final
from itertools import chain
import json

TRAX_NSTEPS: Final[int] = 17
"""The number of rounds in TRAX."""

RCON: Final[List[int]] = [0xB7E15162, 0xBF715880, 0x38B4DA56, 0x324E7738, 0xBB1185EB, 0x4F7C7B57, 0xCFBFA1C8, 0xC2B3293D]
"""The round constants for TRAX."""

def c_ror(x: int, r: int, n: int) -> int:
    """Classical implementation of the right rotation.

    :param x: The integer to rotate
    :param r: The rotation amount
    :param n: The number of bits on which x is encoded.
    :returns: The integer x rotated right by r bits.
    """
    r = r % n
    return ((x >> r) | (x << (n - r))) & ((2**n) - 1)

def c_rol(x, r, n):
    """Classical implementation of the left rotation.

    :param x: The integer to rotate
    :param r: The rotation amount
    :param n: The number of bits on which x is encoded.
    :returns: The integer x rotated right by r bits.
    """
    r = r % n
    return ((x << r) | (x >> (n - r))) & ((2**n) - 1)

def c_alzette(x: int, y: int, c: int, n: int) -> List[int]:
    """Classical software implementation of Alzette.

    :param x: The integer value of x.
    :param y: The integer value of y.
    :param c: The c constant in Alzette.
    :param n: The number of bits on which x, y and n are encoded.
    :returns: alzette(x, y)
    """
    x = (x + c_ror(y, 31, n)) % 2**n
    y = y ^ c_ror(x, 24, n)
    x = x ^ c
    x = (x + c_ror(y, 17, n)) % 2**n
    y = y ^ c_ror(x, 17, n)
    x = x ^ c
    x = (x + y) % 2**n
    y = y ^ c_ror(x, 31, n)
    x = x ^ c
    x = (x + c_ror(y, 24, n)) % 2**n
    y = y ^ c_ror(x, 16, n)
    x = x ^ c
    return [x, y]

def c_ell(x, n):
    return (c_ror(((x) ^ (((x) << n//2)) % 2**n), n//2, n))

def c_traxl_genkeys(key: List[int], n: int=32) -> List[int]:
    """Classical implementation of TRAX-L key generation.

    :param key: A list of the 8 n-bit integer key parts
    :returns: A list of 8*( :py:data:`TRAX_NSTEPS` +1) subkeys
    """
    subkeys = [0]*8*(TRAX_NSTEPS+1)
    key_ = key.copy()
    for s in range(TRAX_NSTEPS+1):
        # assign 8 sub-keys
        for b in range(8):
            subkeys[8*s+b] = key_[b]
        # update master-key
        key_[0] = (key_[0] + key_[1] + (RCON[(2*s)%8] % 2**n)) % 2**n
        key_[0] = (key_[0] + key_[1]) % 2**n
        key_[2] ^= key_[3] ^ (s%2**n)
        key_[4] = (key_[4] + key_[5] + (RCON[(2*s+1)%8] % 2**n)) % 2**n
        key_[4] = (key_[4] + key_[5]) % 2**n
        key_[6] ^= key_[7] ^ ((s << n//2) % 2**n)
        # rotate master-key
        key_.append(key_.pop(0))
    return subkeys

def c_traxl_enc(x, y, subkeys, tweak, n:int =32):
    x = x.copy()
    y = y.copy()

    for s in range(TRAX_NSTEPS):
        # Add tweak if step counter is odd
        if ((s % 2) == 1):
            x[0] ^= tweak[0]
            y[0] ^= tweak[1]
            x[1] ^= tweak[2]
            y[1] ^= tweak[3]

        # Add subkeys to state and execute ALZETTEs
        for b in range(4):
            x[b] ^= subkeys[8*s+2*b]
            y[b] ^= subkeys[8*s+2*b+1]
            x[b], y[b] = c_alzette(x[b], y[b], (RCON[(4*s+b)%8] % 2**n), n)

        # Linear layer (Sparkle256 permutation)
        tmpx = c_ell(x[2]^x[3], n)
        y[0] ^= tmpx
        y[1] ^= tmpx
       # 
        tmpy = c_ell(y[2] ^ y[3], n)
        x[0] ^= tmpy
        x[1] ^= tmpy
       #  
        x[0], x[1], x[2], x[3] = x[3], x[2], x[0], x[1]
        y[0], y[1], y[2], y[3] = y[3], y[2], y[0], y[1]

    # Add subkeys to state for final key addition
    for b in range(4):
        x[b] ^= subkeys[8*TRAX_NSTEPS+2*b]
        y[b] ^= subkeys[8*TRAX_NSTEPS+2*b+1]

    return x, y

class Traxl_enc(Gate, rOperation):
    def __init__(self, x: List[QuantumRegister], y: List[QuantumRegister], key: List[QuantumRegister], tweak: List[int], label: Optional[str] = None):
        if len(x) != 4 or len(y) != 4 or len(key) != 8:
            raise CircuitError("Wrong number of inputs.")
        self._inputs = x+y+key
        self.n = len(x[0])
        super().__init__("Traxl_enc", self.n * 16, [], label=label)

        x = x.copy()
        y = y.copy()
        key = key.copy()

        qc = rCircuit(*self.inputs, name='Traxl_enc')
        s = 0
        for s in range(TRAX_NSTEPS):
            # Add tweak if step counter is odd
            if ((s % 2) == 1):
                qc.xor(x[0], tweak[0])
                qc.xor(y[0], tweak[1])
                qc.xor(x[1], tweak[2])
                qc.xor(y[1], tweak[3])

            # Add subkeys and execute ALZETTEs
            for b in range(4):
                qc.xor(x[b], key[2*b])
                qc.xor(y[b], key[2*b+1])
                qc.append(Alzette(x[b], y[b], (RCON[(4*s+b)%8]%2**self.n)), list(chain(x[b], y[b])))

            # Linear layer (Sparkle256 permutation)
            half = self.n//2
            for b in range(2):
                Yb_0 = QuantumRegister(bits=y[b][0:half])
                Yb_1 = QuantumRegister(bits=y[b][half:])
                X2_0 = QuantumRegister(bits=x[2][0:half])
                X2_1 = QuantumRegister(bits=x[2][half:])
                X3_0 = QuantumRegister(bits=x[3][0:half])
                X3_1 = QuantumRegister(bits=x[3][half:])

                # Here we decompose the sparkle permutation into XOR
                qc.xor(Yb_0, X2_1)
                qc.xor(Yb_0, X2_0)
                qc.xor(Yb_1, X2_0)
                qc.xor(Yb_0, X3_1)
                qc.xor(Yb_0, X3_0)
                qc.xor(Yb_1, X3_0)

            for b in range(2):
                Xb_0 = QuantumRegister(bits=x[b][0:half])
                Xb_1 = QuantumRegister(bits=x[b][half:])
                Y2_0 = QuantumRegister(bits=y[2][0:half])
                Y2_1 = QuantumRegister(bits=y[2][half:])
                Y3_0 = QuantumRegister(bits=y[3][0:half])
                Y3_1 = QuantumRegister(bits=y[3][half:])

                # Here we decompose the sparkle permutation into XOR
                qc.xor(Xb_0, Y2_1)
                qc.xor(Xb_0, Y2_0)
                qc.xor(Xb_1, Y2_0)
                qc.xor(Xb_0, Y3_1)
                qc.xor(Xb_0, Y3_0)
                qc.xor(Xb_1, Y3_0)
       
            x[0], x[1], x[2], x[3] = x[3], x[2], x[0], x[1]
            y[0], y[1], y[2], y[3] = y[3], y[2], y[0], y[1]

            # Compute key schedule
            qc.add(key[0], key[1])
            qc.add(key[0], (RCON[(2*s)%8]%2**self.n))
            qc.xor(key[2], key[3])
            qc.xor(key[2], s%2**self.n)
            qc.add(key[4], key[5])
            qc.add(key[4], (RCON[(2*s+1)%8]%2**self.n))
            qc.xor(key[6], key[7])
            qc.xor(key[6], ((s << self.n//2) % 2**self.n))

            key.append(key.pop(0))

        # Add last round subkeys
        for b in range(4):
            qc.xor(x[b], key[2*b])
            qc.xor(y[b], key[2*b+1])

        self._definition = qc
        self._outputs = x+y+key
        self._quantum_cost = qc.quantum_cost

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
        self._inputs = [X, Y]
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
        self._outputs = [X, Y]
        self._quantum_cost = qc.quantum_cost

if __name__ == '__main__':

    n = 32
    X = QuantumRegister(32, name='X')
    Y = QuantumRegister(32, name='Y')
    a = 384973
    b = 1238444859
    c = 0xb7e15162
    print(f"Running alzette_{c}({a}, {b})")


    qc = rCircuit(X, Y)
    gate = Alzette(X, Y, c)
    qc.append(gate, list(chain(*gate.outputs)))

    decomposition_reps = 2
    qc.decompose(reps=decomposition_reps).qasm(filename='alzette_only.qasm')

    final_circuit = make_circuit(qc, [a, b], [X, Y], [X, Y])

    final_circuit.decompose(reps=2).qasm(filename='alzette.qasm')

    result = run_circuit(final_circuit)

    circuit_depth = qc.decompose(reps=decomposition_reps).depth()
    gate_counts = qc.decompose(reps=decomposition_reps).count_ops()
    print(f"Classical result: {c_alzette(a, b, c, n)}")
    print(f"Quantum simulated result: {result}")
    print("Circuit depth:", circuit_depth)
    print("Circuit gate count:", json.dumps(gate_counts))
    print("Circuit quantum cost:", qc.quantum_cost)
