"""A reversible quantum implementation of the n-qubits Alzette ARX box and the TRAX cipher from the Sparkle-suite as defined in [Alzette2020]_.

.. [Alzette2020] Beierle, C., Biryukov, A., Cardoso dos Santos, L., Großschädl, J., Perrin, L., Udovenko, A., ... & Wang, Q. (2020). Alzette: A 64-Bit ARX-box: (Feat. CRAX and TRAX). In Advances in Cryptology–CRYPTO 2020: 40th Annual International Cryptology Conference, CRYPTO 2020, Santa Barbara, CA, USA, August 17–21, 2020, Proceedings, Part III 40 (pp. 419-448). Springer International Publishing.
"""
from pyqrypto.rOperations import rCircuit, rOperation, rDKRSCarryLookaheadAdder 
from qiskit import QuantumRegister, AncillaRegister
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit import Gate
from typing import Optional, List, Tuple, Final
from itertools import chain

RCON: Final[List[int]] = [0xB7E15162, 0xBF715880, 0x38B4DA56, 0x324E7738, 0xBB1185EB, 0x4F7C7B57, 0xCFBFA1C8, 0xC2B3293D]
"""The round constants for TRAX."""

def c_ror(x: int, r: int, n: int) -> int:
    """Classical implementation of the right rotation.

    :param x: The integer to rotate.
    :param r: The rotation amount.
    :param n: The number of bits on which x is encoded.
    :returns: The integer x rotated right by r bits.
    """
    r = r % n
    return ((x >> r) | (x << (n - r))) & ((2**n) - 1)

def c_rol(x, r, n):
    """Classical implementation of the left rotation.

    :param x: The integer to rotate.
    :param r: The rotation amount.
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
    :returns: alzette(x, y).
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

def c_traxl_genkeys(key: List[int], n: int=32, nsteps: int=17) -> List[int]:
    """Classical implementation of TRAX-L key generation.

    :param key: A list of the 8 :py:data:`n`-bit key parts.
    :param n: Size of each key part.
    :nsteps: Number of rounds.
    :returns: A list of 8*( :py:data:`nsteps` +1) :py:data:`n`-bit subkey parts.
    """
    subkeys = [0]*8*(nsteps+1)
    key_ = key.copy()
    for s in range(nsteps+1):
        # assign 8 sub-keys
        for b in range(8):
            subkeys[8*s+b] = key_[b]
        # update master-key
        key_[0] = (key_[0] + key_[1] + (RCON[(2*s)%8] % 2**n)) % 2**n
        key_[2] ^= key_[3] ^ (s%2**n)
        key_[4] = (key_[4] + key_[5] + (RCON[(2*s+1)%8] % 2**n)) % 2**n
        key_[6] ^= key_[7] ^ ((s << n//2) % 2**n)
        # rotate master-key
        key_.append(key_.pop(0))
    return subkeys

def c_traxs_enc(x, y, subkeys, tweak, n:int=32, nsteps: int=10):
    """Classical implementation of TRAX-S.

    :param x: A :py:data:`n`-bit integer.
    :param y: A :py:data:`n`-bit integer.
    :param subkeys: A list of 2*( :py:data:`nsteps` +1) :py:data:`n`-bit subkey parts.
    :param tweak: A list of the 2 :py:data:`n`-bit tweak parts.
    :param n: Size of each key part.
    :param nsteps: Number of rounds.
    :returns: Encrypted (x,y).
    """
    x = x.copy()
    y = y.copy()

    for s in range(nsteps):
        # Add tweak if step counter is odd
        if ((s % 2) == 1):
            x ^= tweak[0]
            y ^= tweak[1]

        # Add subkeys to state and execute ALZETTEs
        x ^= subkeys[2*s]
        y ^= subkeys[2*s+1]
        x, y = c_alzette(x, y, (RCON[s%8] % 2**n), n)

    # Add subkeys to state for final key addition
    x ^= subkeys[2*nsteps]
    y ^= subkeys[2*nsteps+1]

    return x, y

def c_traxm_enc(x, y, subkeys, tweak, n:int=32, nsteps: int=12, final_addition=True):
    """Classical implementation of TRAX-M.

    :param x: [:math:`x_0`, :math:`x_1`] where :math:`x_i` is a :py:data:`n`-bit integer.
    :param y: [:math:`y_0`, :math:`y_1`] where :math:`y_i` is a :py:data:`n`-bit integer.
    :param subkeys: A list of 4*( :py:data:`nsteps` +1) :py:data:`n`-bit subkey parts.
    :param tweak: A list of the 2 :py:data:`n`-bit tweak parts.
    :param n: Size of each key part.
    :param nsteps: Number of rounds.
    :param final_addition: Whether to do the final key addition
    :returns: Encrypted (x,y).
    """
    x = x.copy()
    y = y.copy()

    for s in range(nsteps):
        # Add tweak if step counter is odd
        if ((s % 2) == 1):
            x[0] ^= tweak[0]
            y[0] ^= tweak[1]

        # Add subkeys to state and execute ALZETTEs
        for b in range(2):
            x[b] ^= subkeys[4*s+2*b]
            y[b] ^= subkeys[4*s+2*b+1]
            x[b], y[b] = c_alzette(x[b], y[b], (RCON[(2*s+b)%8] % 2**n), n)

        # Linear layer
        x[0], y[0], x[1], y[1] = (x[0] ^ x[1], y[0] ^ y[1], x[0], y[0])

    # Add subkeys to state for final key addition
    if final_addition:
        for b in range(2):
            x[b] ^= subkeys[4*nsteps+2*b]
            y[b] ^= subkeys[4*nsteps+2*b+1]

    return x, y


def c_traxl_enc(x: List[int], y: List[int], subkeys: List[int], tweak: List[int], n:int=32, nsteps: int=17) -> Tuple[List[int], List[int]]:
    """Classical implementation of TRAX-L.

    :param x: [:math:`x_0`, :math:`x_1`, :math:`x_2`, :math:`x_3`] where :math:`x_i` is a :py:data:`n`-bit integer.
    :param y: [:math:`y_0`, :math:`y_1`, :math:`y_2`, :math:`y_3`] where :math:`y_i` is a :py:data:`n`-bit integer.
    :param subkeys: A list of 8*( :py:data:`nsteps` +1) :py:data:`n`-bit subkey parts.
    :param tweak: A list of the 4 :py:data:`n`-bit tweak parts.
    :param n: Size of each key part.
    :param nsteps: Number of rounds.
    :returns: Encrypted (x,y).
    """
    x = x.copy()
    y = y.copy()

    for s in range(nsteps):
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

        # Linear layer (see Sparkle256 permutation)
        tmpx = c_ell(x[2]^x[3], n)
        y[0] ^= tmpx
        y[1] ^= tmpx

        tmpy = c_ell(y[2] ^ y[3], n)
        x[0] ^= tmpy
        x[1] ^= tmpy

        x[0], x[1], x[2], x[3] = x[3], x[2], x[0], x[1]
        y[0], y[1], y[2], y[3] = y[3], y[2], y[0], y[1]

    # Add subkeys to state for final key addition
    for b in range(4):
        x[b] ^= subkeys[8*nsteps+2*b]
        y[b] ^= subkeys[8*nsteps+2*b+1]

    return x, y

class Alzette(Gate, rOperation):
    r"""A quantum gate implementing the ARX-box Alzette of two vectors of qubits.

    :param X: X vector.
    :param Y: Y vector.
    :param c: Alzette constant.
    :param ancillas: The anquilla register needed if :py:obj:`adder_mode` is :py:data:`lookahead`.
    :param adder_mode: See :py:func:`pyqrypto.rOperations.rCircuit.add`.
    :param label: An optional label for the gate.
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
    @staticmethod
    def get_num_ancilla_qubits(n: int=32, adder_mode='ripple') -> int:
        """Get the number of required ancilla qubits without instantiating the class.

        :param n: The size of the two inputs of Alzette.
        :param adder_mode: See :py:func:`pyqrypto.rOperations.rCircuit.add`.
        :returns: The number of ancilla qubits needed for the computation.
        """
        if adder_mode=='ripple':
            return 0
        if adder_mode=='lookahead':
            return rDKRSCarryLookaheadAdder.get_num_ancilla_qubits(n)
        else:
            raise CircuitError(f"Unknown adder mode {adder_mode}.")

    def __init__(self, X: QuantumRegister, Y: QuantumRegister, c: int, ancillas: Optional[AncillaRegister]=None, adder_mode='ripple', label: Optional[str] = None):
        if len(X) != len(Y):
            raise CircuitError("X and Y must have the same size.")
        self.n = len(X)
        self.c = c
        self._inputs = [X, Y]
        self._outputs = [X, Y]
        if (len(ancillas) if ancillas else 0) != self.get_num_ancilla_qubits(self.n, adder_mode=adder_mode):
            raise CircuitError(f"Circuit needs {self.get_num_ancilla_qubits(self.n, adder_mode=adder_mode)} ancilla qubits but {len(ancillas) if ancillas else 0} were given.")
        if ancillas is not None:
            self._inputs.append(ancillas)
            self._outputs.append(ancillas)
            super().__init__("Alzette", self.n*2+len(ancillas), [], label=label)
            qc = rCircuit(X, Y, ancillas, name='Alzette')
        else:
            super().__init__("Alzette", self.n*2, [], label=label)
            qc = rCircuit(X, Y, name='Alzette')

        qc.add(X, qc.ror(Y, 31), ancillas, mode=adder_mode)
        qc.xor(Y, qc.ror(X, 24))
        qc.xor(X, self.c)
        qc.add(X, qc.ror(Y, 17), ancillas, mode=adder_mode)
        qc.xor(Y, qc.ror(X, 17))
        qc.xor(X, self.c)
        qc.add(X, Y, ancillas, mode=adder_mode)
        qc.xor(Y, qc.ror(X, 31))
        qc.xor(X, self.c)
        qc.add(X, qc.ror(Y, 24), ancillas, mode=adder_mode)
        qc.xor(Y, qc.ror(X, 16))
        qc.xor(X, self.c)

        self._definition = qc
        self._outputs = [X, Y]

class Traxm_enc_round(Gate, rOperation):
    """A quantum implementation of a round of the encryption step of the TRAX-M as defined in [Alzette2020] with a block size of n bits.
    The default block size for this cipher is 128 bits.
    Only a single round is implemented because there is no standard key schedule for this cipher.


    The Alzette rounds of TRAX can be implemented using different techniques:

    - :py:data:`lookahead-parallel`: Run the 2 instances of Alzette in parallel using carry lookahead adders.
    - :py:data:`lookahead-sequential`: Run the 2 instances of Alzette sequentially using carry lookahead adders.
    - :py:data:`ripple`: Run the 2 instances of Alzette in parallel using ripple carry adders.

    :param x: A list of 2 quantum registers of size n/4.
    :param y: A list of 2 quantum registers of size n/4.
    :param key: The round key, a list of 4 quantum registers of size n/4.
    :param tweak: The tweak, a list of 2 integers on n/4 bits. None if the tweak souldn't be added at that round.
    :param ancillas: The ancillas qubits needed for the computation.
    :param alzette_mode: The method to use to compute the Alzette rounds.
    :param round_constants: A list of the two round constants to use for this round.
    :param label: An optional label for the gate.
    """

    @staticmethod
    def _get_num_ancilla_registers(alzette_mode: str='lookahead-parallel'):
        """Returns the numbers of ancilla registers used internally by Traxm_enc_round."""

        if alzette_mode == 'lookahead-parallel':
            num_ancilla_registers = 2
        elif alzette_mode == 'lookahead-sequential' or alzette_mode == 'ripple':
            num_ancilla_registers = 1
        else:
            raise CircuitError(f"Unknown alzette mode {alzette_mode}.")

        return num_ancilla_registers

    @staticmethod
    def get_num_ancilla_qubits(n: int=128, alzette_mode:str ='lookahead-parallel') -> int:
        """Get the number of required ancilla qubits without instantiating the class.

        :param n: The size of the block size the cipher is operating on.
        :param alzette_mode: The method to use to compute the Alzette rounds.
        :returns: The number of ancilla qubits needed for the computation.
        """
        return rDKRSCarryLookaheadAdder.get_num_ancilla_qubits(n//4)*Traxm_enc_round._get_num_ancilla_registers(alzette_mode)

    def __init__(self, x: List[QuantumRegister], y: List[QuantumRegister], round_key: List[QuantumRegister], tweak: Optional[List[int]], ancillas: AncillaRegister, alzette_mode: str='lookahead-parallel', round_constants: List[int] = [RCON[0], RCON[1]], label: Optional[str] = None):
        if len(x) != 2 or len(y) != 2 or len(round_key) != 4 or (tweak is not None and len(tweak) != 2):
            raise CircuitError("Wrong number of inputs.")
        self._inputs = x+y+round_key+[ancillas]
        self.n = len(x[0]) # Number of qubits in each register (by default 32)

        super().__init__("Traxm_enc_round", self.n * 8 + len(ancillas) , [], label=label)

        x = x.copy()
        y = y.copy()
        round_key = round_key.copy()
        num_ancilla_registers = self._get_num_ancilla_registers(alzette_mode)
        ancilla_registers = [AncillaRegister(bits=ancillas[i*len(ancillas)//num_ancilla_registers:(i+1)*len(ancillas)//num_ancilla_registers]) for i in range(num_ancilla_registers)]

        qc = rCircuit(*self.inputs, name='Trax_enc_round')

        # Add tweak if it exists
        if tweak is not None:
            qc.xor(x[0], tweak[0])
            qc.xor(y[0], tweak[1])

        # Add subkeys and execute ALZETTEs
        for b in range(2):
            qc.xor(x[b], round_key[2*b])
            qc.xor(y[b], round_key[2*b+1])

            if alzette_mode == 'ripple':
                qc.append(Alzette(x[b], y[b], (round_constants[b]%2**self.n), adder_mode='ripple'), list(chain(x[b], y[b])))
            elif alzette_mode == 'lookahead-sequential':
                qc.append(Alzette(x[b], y[b], (round_constants[b]%2**self.n), ancilla_registers[0], adder_mode='lookahead'), list(chain(x[b], y[b], ancilla_registers[0])))
            elif alzette_mode == 'lookahead-parallel':
                qc.append(Alzette(x[b], y[b], (round_constants[b]%2**self.n), ancilla_registers[b], adder_mode='lookahead'), list(chain(x[b], y[b], ancilla_registers[b])))
            else:
                raise CircuitError(f"Unknown alzette mode {alzette_mode}.")
    
        # Linear layer
        qc.xor(x[1], x[0])
        qc.xor(y[1], y[0])
        x[0], y[0], x[1], y[1] = x[1], y[1], x[0], y[0]

        self._outputs = x+y+round_key+[ancillas]
        self._definition = qc

class Traxl_enc(Gate, rOperation):
    """A quantum implementation of the encryption step of the TRAX-L cipher as defined in [Alzette2020]_ with a block size of n bits.
    The default block size for this cipher is 256 bits.

    The Alzette rounds of TRAX can be implemented using different techniques:

    - :py:data:`lookahead-parallel`: Run the 4 instances of Alzette in parallel using carry lookahead adders.
    - :py:data:`lookahead-half-parallel`: Run 2 instances of Alzette in parallel using carry lookahead adders.
    - :py:data:`lookahead-sequential`: Run the 4 instances of Alzette sequentially using carry lookahead adders.
    - :py:data:`ripple`: Run the 4 instances of Alzette in parallel using ripple carry adders.

    The key schedule can be implemented using different techniques:

    - :py:data:`ripple`: Compute the key schedule using ripple carry adders for the non constant adders.
    - :py:data:`lookahead`: Compute the key schedule using only carry lookahead adders.

    Each combinaison of techniques involves a tradeoff between the circuit depth, the quantum cost and the number of needed ancilla qubits. Please read the `Grover on TRAX` paper for more information about this.

    :param x: A list of 4 quantum registers of size n/8.
    :param y: A list of 4 quantum registers of size n/8.
    :param key: The encryption key, a list of 8 quantum registers of size n/8.
    :param tweak: The tweak, a list of 4 integers on n/8 bits.
    :param ancillas: The ancillas qubits needed for the computation.
    :param alzette_mode: The method to use to compute the Alzette rounds.
    :param schedule_mode: The method to use to compute the key schedule.
    :param nsteps: The number of rounds of TRAX.
    :param label: An optional label for the gate.
    """
    @staticmethod
    def _get_num_ancilla_registers(alzette_mode: str='lookahead-parallel', schedule_mode: str='lookahead'):
        """Returns the numbers of ancilla registers used internally by Traxl_enc."""

        if alzette_mode == 'lookahead-parallel':
            num_ancilla_registers = 4
        elif alzette_mode == 'lookahead-half-parallel':
            num_ancilla_registers = 2
        elif alzette_mode == 'lookahead-sequential' or alzette_mode == 'ripple':
            num_ancilla_registers = 1
        else:
            raise CircuitError(f"Unknown alzette mode {alzette_mode}.")
        if schedule_mode == 'lookahead':
            num_ancilla_registers += 1
        elif schedule_mode != 'ripple':
            raise CircuitError(f"Unknown schedule mode {schedule_mode}.")

        return num_ancilla_registers

    @staticmethod
    def get_num_ancilla_qubits(n: int=256, alzette_mode:str ='lookahead-parallel', schedule_mode: str='lookahead') -> int:
        """Get the number of required ancilla qubits without instantiating the class.

        :param n: The size of the block size the cipher is operating on.
        :param alzette_mode: The method to use to compute the Alzette rounds.
        :returns: The number of ancilla qubits needed for the computation.
        """
        return rDKRSCarryLookaheadAdder.get_num_ancilla_qubits(n//8)*Traxl_enc._get_num_ancilla_registers(alzette_mode, schedule_mode)

    def __init__(self, x: List[QuantumRegister], y: List[QuantumRegister], key: List[QuantumRegister], tweak: List[int], ancillas: AncillaRegister, alzette_mode: str='lookahead-parallel', schedule_mode: str='lookahead', nsteps: int=17, label: Optional[str] = None):
        if len(x) != 4 or len(y) != 4 or len(key) != 8 or len(tweak) != 4:
            raise CircuitError("Wrong number of inputs.")
        self._inputs = x+y+key+[ancillas]
        self.n = len(x[0]) # Number of qubits in each register (by default 32)

        super().__init__("Traxl_enc", self.n * 16 + len(ancillas) , [], label=label)

        if len(ancillas) != self.get_num_ancilla_qubits(self.n*8, alzette_mode, schedule_mode):
            raise CircuitError(f"Circuit needs {self.get_num_ancilla_qubits(self.n*8, alzette_mode, schedule_mode)} ancilla qubits but {len(ancillas)} were given.")

        x = x.copy()
        y = y.copy()
        key = key.copy()
        num_ancilla_registers = self._get_num_ancilla_registers(alzette_mode, schedule_mode)
        ancilla_registers = [AncillaRegister(bits=ancillas[i*len(ancillas)//num_ancilla_registers:(i+1)*len(ancillas)//num_ancilla_registers]) for i in range(num_ancilla_registers)]

        qc = rCircuit(*self.inputs, name='Traxl_enc')

        for s in range(nsteps):

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
                if alzette_mode == 'ripple':
                    qc.append(Alzette(x[b], y[b], (RCON[(4*s+b)%8]%2**self.n), adder_mode='ripple'), list(chain(x[b], y[b])))
                elif alzette_mode == 'lookahead-sequential':
                    qc.append(Alzette(x[b], y[b], (RCON[(4*s+b)%8]%2**self.n), ancilla_registers[0], adder_mode='lookahead'), list(chain(x[b], y[b], ancilla_registers[0])))
                elif alzette_mode == 'lookahead-half-parallel':
                    qc.append(Alzette(x[b], y[b], (RCON[(4*s+b)%8]%2**self.n), ancilla_registers[b//2], adder_mode='lookahead'), list(chain(x[b], y[b], ancilla_registers[b//2])))
                elif alzette_mode == 'lookahead-parallel':
                    qc.append(Alzette(x[b], y[b], (RCON[(4*s+b)%8]%2**self.n), ancilla_registers[b], adder_mode='lookahead'), list(chain(x[b], y[b], ancilla_registers[b])))
                else:
                    raise CircuitError(f"Unknown alzette mode {alzette_mode}.")
            

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
            if schedule_mode == 'lookahead':
                qc.add(key[0], key[1], ancilla_registers[-1], mode='lookahead')
                qc.add(key[0], (RCON[(2*s)%8]%2**self.n), ancilla_registers[-1])
            elif schedule_mode == 'ripple':
                qc.add(key[0], key[1], mode='ripple')
                qc.add(key[0], (RCON[(2*s)%8]%2**self.n), ancilla_registers[0])
            else:
                raise CircuitError(f"Unknown schedule mode {schedule_mode}.")

            qc.xor(key[2], key[3])
            qc.xor(key[2], s%2**self.n)

            if schedule_mode == 'lookahead':
                qc.add(key[4], key[5], ancilla_registers[-1], mode='lookahead')
                qc.add(key[4], (RCON[(2*s+1)%8]%2**self.n), ancilla_registers[-1])
            elif schedule_mode == 'ripple':
                qc.add(key[4], key[5], mode='ripple')
                qc.add(key[4], (RCON[(2*s+1)%8]%2**self.n), ancilla_registers[-1])
            else:
                raise CircuitError(f"Unknown schedule mode {schedule_mode}.")

            qc.xor(key[6], key[7])
            qc.xor(key[6], ((s << self.n//2) % 2**self.n))

            key.append(key.pop(0))


        # Add last round subkeys
        for b in range(4):
            qc.xor(x[b], key[2*b])
            qc.xor(y[b], key[2*b+1])

        self._outputs = x+y+key+[ancillas]
        self._definition = qc
