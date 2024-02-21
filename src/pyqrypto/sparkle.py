"""A reversible quantum implementation of ciphers from the Sparkle-suite.

It implements the n-qubits Alzette ARX box and the TRAX ciphers defined in [Alzette2020]_.

.. [Alzette2020]
    Beierle, C., Biryukov, A., Cardoso dos Santos, L., Großschädl, J., Perrin, L., Udovenko, A.,
    ... & Wang, Q. (2020). Alzette: A 64-Bit ARX-box: (Feat. CRAX and TRAX). In Advances in
    Cryptology-CRYPTO 2020: 40th Annual International Cryptology Conference, CRYPTO 2020,
    Santa Barbara, CA, USA, August 17-21, 2020, Proceedings, Part III 40 (pp. 419-448).
    Springer International Publishing.
"""
from __future__ import annotations

from itertools import chain
from typing import Final

from qiskit import AncillaRegister, QuantumRegister
from qiskit.circuit import Gate
from qiskit.circuit.exceptions import CircuitError

from pyqrypto.register_operations import (
    RegisterCircuit,
    RegisterDKRSCarryLookaheadAdder,
    RegisterOperation,
)

RCON: Final[list[int]] = [
    0xB7E15162,
    0xBF715880,
    0x38B4DA56,
    0x324E7738,
    0xBB1185EB,
    0x4F7C7B57,
    0xCFBFA1C8,
    0xC2B3293D,
]
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


def c_rol(x: int, r: int, n: int) -> int:
    """Classical implementation of the left rotation.

    :param x: The integer to rotate.
    :param r: The rotation amount.
    :param n: The number of bits on which x is encoded.
    :returns: The integer x rotated left by r bits.
    """
    r = r % n
    return ((x << r) | (x >> (n - r))) & ((2**n) - 1)


def c_alzette(x: int, y: int, c: int, n: int) -> list[int]:
    r"""Classical software implementation of Alzette.

    :param x: The integer value of x.
    :param y: The integer value of y.
    :param c: The c constant in Alzette.
    :param n: The number of bits on which x, y and n are encoded.
    :returns: :math:`\text{ALZETTE}(x, y)`.
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


def c_ell(x: int, n: int) -> int:
    r"""Classical implementation of Sparkle's ELL transformation.

    :param x: The integer value of x.
    :param n: The number of bits on which x is encoded.
    :returns: :math:`\text{ELL(x)}`.
    """
    return c_ror(((x) ^ ((x) << n // 2) % 2**n), n // 2, n)


def c_traxl_genkeys(key: list[int], n: int = 32, nsteps: int = 17) -> list[int]:
    """Classical implementation of TRAX-L key generation.

    :param key: A list of the 8 :py:data:`n`-bit key parts.
    :param n: Size of each key part.
    :nsteps: Number of rounds.
    :returns: A list of 8*( :py:data:`nsteps` +1) :py:data:`n`-bit subkey parts.
    """
    subkeys = [0] * 8 * (nsteps + 1)
    key_ = key.copy()
    for s in range(nsteps + 1):
        # assign 8 sub-keys
        for b in range(8):
            subkeys[8 * s + b] = key_[b]
        # update master-key
        key_[0] = (key_[0] + key_[1] + (RCON[(2 * s) % 8] % 2**n)) % 2**n
        key_[2] ^= key_[3] ^ (s % 2**n)
        key_[4] = (key_[4] + key_[5] + (RCON[(2 * s + 1) % 8] % 2**n)) % 2**n
        key_[6] ^= key_[7] ^ ((s << n // 2) % 2**n)
        # rotate master-key
        key_.append(key_.pop(0))
    return subkeys


def c_traxs_enc(
    x: int,
    y: int,
    subkeys: list[int],
    tweak: list[int],
    n: int = 32,
    nsteps: int = 10,
    final_addition=True,
) -> tuple[int, int]:
    """Classical implementation of TRAX-S.

    :param x: A :py:data:`n`-bit integer.
    :param y: A :py:data:`n`-bit integer.
    :param subkeys: A list of 2*( :py:data:`nsteps` +1) :py:data:`n`-bit subkey parts.
    :param tweak: A list of the 2 :py:data:`n`-bit tweak parts.
    :param n: Size of each key part.
    :param nsteps: Number of rounds.
    :param final_addition: Whether to do the final key addition.
    :returns: Encrypted (x,y).
    """
    for s in range(nsteps):
        # Add tweak if step counter is odd
        if (s % 2) == 1:
            x ^= tweak[0]
            y ^= tweak[1]

        # Add subkeys to state and execute ALZETTEs
        x ^= subkeys[2 * s]
        y ^= subkeys[2 * s + 1]
        x, y = c_alzette(x, y, (RCON[s % 8] % 2**n), n)

    # Add subkeys to state for final key addition
    if final_addition:
        x ^= subkeys[2 * nsteps]
        y ^= subkeys[2 * nsteps + 1]

    return x, y


def c_traxm_enc(
    x: list[int],
    y: list[int],
    subkeys: list[int],
    tweak: list[int],
    n: int = 32,
    nsteps: int = 12,
    final_addition=True,
) -> tuple[list[int], list[int]]:
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
        if (s % 2) == 1:
            x[0] ^= tweak[0]
            y[0] ^= tweak[1]

        # Add subkeys to state and execute ALZETTEs
        for b in range(2):
            x[b] ^= subkeys[4 * s + 2 * b]
            y[b] ^= subkeys[4 * s + 2 * b + 1]
            x[b], y[b] = c_alzette(x[b], y[b], (RCON[(2 * s + b) % 8] % 2**n), n)

        # Linear layer
        x[0], y[0], x[1], y[1] = (x[0] ^ x[1], y[0] ^ y[1], x[0], y[0])

    # Add subkeys to state for final key addition
    if final_addition:
        for b in range(2):
            x[b] ^= subkeys[4 * nsteps + 2 * b]
            y[b] ^= subkeys[4 * nsteps + 2 * b + 1]

    return x, y


def c_traxl_enc(
    x: list[int],
    y: list[int],
    subkeys: list[int],
    tweak: list[int],
    n: int = 32,
    nsteps: int = 17,
) -> tuple[list[int], list[int]]:
    """Classical implementation of TRAX-L.

    :param x: [:math:`x_0`, :math:`x_1`, :math:`x_2`, :math:`x_3`]
      where :math:`x_i` is a :py:data:`n`-bit integer.
    :param y: [:math:`y_0`, :math:`y_1`, :math:`y_2`, :math:`y_3`]
      where :math:`y_i` is a :py:data:`n`-bit integer.
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
        if (s % 2) == 1:
            x[0] ^= tweak[0]
            y[0] ^= tweak[1]
            x[1] ^= tweak[2]
            y[1] ^= tweak[3]

        # Add subkeys to state and execute ALZETTEs
        for b in range(4):
            x[b] ^= subkeys[8 * s + 2 * b]
            y[b] ^= subkeys[8 * s + 2 * b + 1]
            x[b], y[b] = c_alzette(x[b], y[b], (RCON[(4 * s + b) % 8] % 2**n), n)

        # Linear layer (see Sparkle256 permutation)
        tmpx = c_ell(x[2] ^ x[3], n)
        y[0] ^= tmpx
        y[1] ^= tmpx

        tmpy = c_ell(y[2] ^ y[3], n)
        x[0] ^= tmpy
        x[1] ^= tmpy

        x[0], x[1], x[2], x[3] = x[3], x[2], x[0], x[1]
        y[0], y[1], y[2], y[3] = y[3], y[2], y[0], y[1]

    # Add subkeys to state for final key addition
    for b in range(4):
        x[b] ^= subkeys[8 * nsteps + 2 * b]
        y[b] ^= subkeys[8 * nsteps + 2 * b + 1]

    return x, y


class Alzette(Gate, RegisterOperation):
    r"""A quantum gate implementing the ARX-box Alzette of two vectors of qubits.

    :param X: X vector.
    :param Y: Y vector.
    :param c: Alzette constant.
    :param ancillas: The anquilla register needed if :py:obj:`adder_mode` is :py:data:`lookahead`.
    :param adder_mode: See
      :py:func:`RegisterCircuit.add <pyqrypto.register_operations.RegisterCircuit.add>`.
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
    def get_num_ancilla_qubits(n: int = 32, adder_mode: str = "ripple") -> int:
        """Get the number of required ancilla qubits without instantiating the class.

        :param n: The size of the two inputs of Alzette.
        :param adder_mode: See
          :py:func:`RegisterCircuit.add <pyqrypto.register_operations.RegisterCircuit.add>`.
        :returns: The number of ancilla qubits needed for the computation.
        """
        if adder_mode == "ripple":
            return 0
        if adder_mode == "lookahead":
            return RegisterDKRSCarryLookaheadAdder.get_num_ancilla_qubits(n)
        msg = f"Unknown adder mode {adder_mode}."
        raise CircuitError(msg)

    def __init__(
        self: Alzette,
        X: QuantumRegister,
        Y: QuantumRegister,
        c: int,
        ancillas: AncillaRegister | None = None,
        adder_mode: str = "ripple",
        label: str | None = None,
    ) -> None:
        """Initialize Alzette."""
        if len(X) != len(Y):
            msg = "X and Y must have the same size."
            raise CircuitError(msg)
        RegisterOperation.__init__(self)
        self.n = len(X)
        self.c = c
        self._inputs = [X, Y]
        self._outputs = [X, Y]
        if (len(ancillas) if ancillas else 0) != self.get_num_ancilla_qubits(
            self.n,
            adder_mode=adder_mode,
        ):
            msg = (
                f"Circuit needs {self.get_num_ancilla_qubits(self.n, adder_mode=adder_mode)} "
                f"ancilla qubits but {len(ancillas) if ancillas else 0} were given."
            )
            raise CircuitError(msg)
        if ancillas is not None:
            self._inputs.append(ancillas)
            self._outputs.append(ancillas)
            Gate.__init__(self, "Alzette", self.n * 2 + len(ancillas), [], label=label)
            circuit = RegisterCircuit(X, Y, ancillas, name="Alzette")
        else:
            Gate.__init__(self, "Alzette", self.n * 2, [], label=label)
            circuit = RegisterCircuit(X, Y, name="Alzette")

        circuit.add(X, circuit.ror(Y, 31), ancillas, mode=adder_mode)
        circuit.xor(Y, circuit.ror(X, 24))
        circuit.xor(X, self.c)
        circuit.add(X, circuit.ror(Y, 17), ancillas, mode=adder_mode)
        circuit.xor(Y, circuit.ror(X, 17))
        circuit.xor(X, self.c)
        circuit.add(X, Y, ancillas, mode=adder_mode)
        circuit.xor(Y, circuit.ror(X, 31))
        circuit.xor(X, self.c)
        circuit.add(X, circuit.ror(Y, 24), ancillas, mode=adder_mode)
        circuit.xor(Y, circuit.ror(X, 16))
        circuit.xor(X, self.c)

        self._definition = circuit
        self._outputs = [X, Y]


class TraxsEncRound(Gate, RegisterOperation):
    """A quantum implementation of a round of the encryption step of TRAX-S.

    The default block size for this cipher is 64 bits.
    Only a single round is implemented because there is no standard key schedule for this cipher.

    The adder in the Alzette rounds of TRAX can be implemented with two methods:

    - :py:data:`lookahead`: Use a carry-lookahead adder.
    - :py:data:`ripple`: Use a ripple-carry adder.

    :param x: A quantum register of size n/2.
    :param y: A quantum register of size n/2.
    :param key: The round key, a list of 2 quantum registers of size n/2.
    :param tweak: The tweak, a list of 2 integers on n/4 bits.
      None if the tweak souldn't be added at that round.
    :param ancillas: The ancillas qubits needed for the computation.
    :param alzette_mode: The method to use to compute the Alzette rounds.
    :param round_constant: The round constant to use for this round.
    :param label: An optional label for the gate.
    """

    @staticmethod
    def _get_num_ancilla_registers(alzette_mode: str = "lookahead") -> int:
        """Get the numbers of ancilla registers used internally by TraxmEncRound."""
        if alzette_mode in ("lookahead", "ripple"):
            num_ancilla_registers = 1
        else:
            msg = f"Unknown alzette mode {alzette_mode}."
            raise CircuitError(msg)

        return num_ancilla_registers

    @staticmethod
    def get_num_ancilla_qubits(n: int = 64, alzette_mode: str = "lookahead") -> int:
        """Get the number of required ancilla qubits without instantiating the class.

        :param n: The block size of the cipher.
        :param alzette_mode: The method to use to compute the Alzette rounds.
        :returns: The number of ancilla qubits needed for the computation.
        """
        return RegisterDKRSCarryLookaheadAdder.get_num_ancilla_qubits(
            n // 2,
        ) * TraxsEncRound._get_num_ancilla_registers(alzette_mode)

    def __init__(
        self: TraxmEncRound,
        x: QuantumRegister,
        y: QuantumRegister,
        round_key: list[QuantumRegister],
        tweak: list[int] | None,
        ancillas: AncillaRegister,
        alzette_mode: str = "lookahead",
        round_constant: int = RCON[0],
        label: str | None = None,
    ) -> None:
        """Initialize TraxmEncRound."""
        RegisterOperation.__init__(self)
        if len(round_key) != 2 or (tweak is not None and len(tweak) != 2):
            msg = "Wrong number of inputs."
            raise CircuitError(msg)
        self._inputs = [x, y, *round_key, ancillas]
        self.n = len(x)  # Number of qubits in each register (by default 32)

        Gate.__init__(self, "Traxs_enc_round", self.n * 4 + len(ancillas), [], label=label)

        round_key = round_key.copy()
        num_ancilla_registers = self._get_num_ancilla_registers(alzette_mode)
        ancilla_registers = [
            AncillaRegister(
                bits=ancillas[
                    i * len(ancillas) // num_ancilla_registers : (i + 1)
                    * len(ancillas)
                    // num_ancilla_registers
                ],
            )
            for i in range(num_ancilla_registers)
        ]

        circuit = RegisterCircuit(*self.inputs, name="Traxs_enc_round")

        # Add tweak if it exists
        if tweak is not None:
            circuit.xor(x, tweak[0])
            circuit.xor(y, tweak[1])

        # Add subkeys and execute ALZETTEs
        circuit.xor(x, round_key[0])
        circuit.xor(y, round_key[1])

        if alzette_mode == "ripple":
            circuit.append(
                Alzette(x, y, (round_constant % 2**self.n), adder_mode="ripple"),
                list(chain(x, y)),
            )
        elif alzette_mode == "lookahead":
            circuit.append(
                Alzette(
                    x,
                    y,
                    (round_constant % 2**self.n),
                    ancilla_registers[0],
                    adder_mode="lookahead",
                ),
                list(chain(x, y, ancilla_registers[0])),
            )
        else:
            msg = f"Unknown alzette mode {alzette_mode}."
            raise CircuitError(msg)

        self._outputs = [x, y, *round_key, ancillas]
        self._definition = circuit


class TraxmEncRound(Gate, RegisterOperation):
    """A quantum implementation of a round of the encryption step of TRAX-M.

    The default block size for this cipher is 128 bits.
    Only a single round is implemented because there is no standard key schedule for this cipher.

    The Alzette rounds of TRAX can be implemented using different techniques:

    - :py:data:`lookahead-parallel`: Run the 2 instances of Alzette in parallel using
      carry lookahead adders.
    - :py:data:`lookahead-sequential`: Run the 2 instances of Alzette sequentially using
      carry lookahead adders.
    - :py:data:`ripple`: Run the 2 instances of Alzette in parallel using ripple carry adders.

    :param x: A list of 2 quantum registers of size n/4.
    :param y: A list of 2 quantum registers of size n/4.
    :param key: The round key, a list of 4 quantum registers of size n/4.
    :param tweak: The tweak, a list of 2 integers on n/4 bits.
      None if the tweak souldn't be added at that round.
    :param ancillas: The ancillas qubits needed for the computation.
    :param alzette_mode: The method to use to compute the Alzette rounds.
    :param round_constants: A list of the two round constants to use for this round.
    :param label: An optional label for the gate.
    """

    @staticmethod
    def _get_num_ancilla_registers(alzette_mode: str = "lookahead-parallel") -> int:
        """Get the numbers of ancilla registers used internally by TraxmEncRound."""
        if alzette_mode == "lookahead-parallel":
            num_ancilla_registers = 2
        elif alzette_mode in ("lookahead-sequential", "ripple"):
            num_ancilla_registers = 1
        else:
            msg = f"Unknown alzette mode {alzette_mode}."
            raise CircuitError(msg)

        return num_ancilla_registers

    @staticmethod
    def get_num_ancilla_qubits(n: int = 128, alzette_mode: str = "lookahead-parallel") -> int:
        """Get the number of required ancilla qubits without instantiating the class.

        :param n: The block size of the cipher.
        :param alzette_mode: The method to use to compute the Alzette rounds.
        :returns: The number of ancilla qubits needed for the computation.
        """
        return RegisterDKRSCarryLookaheadAdder.get_num_ancilla_qubits(
            n // 4,
        ) * TraxmEncRound._get_num_ancilla_registers(alzette_mode)

    def __init__(
        self: TraxmEncRound,
        x: list[QuantumRegister],
        y: list[QuantumRegister],
        round_key: list[QuantumRegister],
        tweak: list[int] | None,
        ancillas: AncillaRegister,
        alzette_mode: str = "lookahead-parallel",
        round_constants: tuple[int, int] = (RCON[0], RCON[1]),
        label: str | None = None,
    ) -> None:
        """Initialize TraxmEncRound."""
        RegisterOperation.__init__(self)
        if (
            len(x) != 2
            or len(y) != 2
            or len(round_key) != 4
            or (tweak is not None and len(tweak) != 2)
        ):
            msg = "Wrong number of inputs."
            raise CircuitError(msg)
        self._inputs = x + y + round_key + [ancillas]
        self.n = len(x[0])  # Number of qubits in each register (by default 32)

        Gate.__init__(self, "Traxm_enc_round", self.n * 8 + len(ancillas), [], label=label)

        x = x.copy()
        y = y.copy()
        round_key = round_key.copy()
        num_ancilla_registers = self._get_num_ancilla_registers(alzette_mode)
        ancilla_registers = [
            AncillaRegister(
                bits=ancillas[
                    i * len(ancillas) // num_ancilla_registers : (i + 1)
                    * len(ancillas)
                    // num_ancilla_registers
                ],
            )
            for i in range(num_ancilla_registers)
        ]

        circuit = RegisterCircuit(*self.inputs, name="Traxm_enc_round")

        # Add tweak if it exists
        if tweak is not None:
            circuit.xor(x[0], tweak[0])
            circuit.xor(y[0], tweak[1])

        # Add subkeys and execute ALZETTEs
        for b in range(2):
            circuit.xor(x[b], round_key[2 * b])
            circuit.xor(y[b], round_key[2 * b + 1])

            if alzette_mode == "ripple":
                circuit.append(
                    Alzette(x[b], y[b], (round_constants[b] % 2**self.n), adder_mode="ripple"),
                    list(chain(x[b], y[b])),
                )
            elif alzette_mode == "lookahead-sequential":
                circuit.append(
                    Alzette(
                        x[b],
                        y[b],
                        (round_constants[b] % 2**self.n),
                        ancilla_registers[0],
                        adder_mode="lookahead",
                    ),
                    list(chain(x[b], y[b], ancilla_registers[0])),
                )
            elif alzette_mode == "lookahead-parallel":
                circuit.append(
                    Alzette(
                        x[b],
                        y[b],
                        (round_constants[b] % 2**self.n),
                        ancilla_registers[b],
                        adder_mode="lookahead",
                    ),
                    list(chain(x[b], y[b], ancilla_registers[b])),
                )
            else:
                msg = f"Unknown alzette mode {alzette_mode}."
                raise CircuitError(msg)

        # Linear layer
        circuit.xor(x[1], x[0])
        circuit.xor(y[1], y[0])
        x[0], y[0], x[1], y[1] = x[1], y[1], x[0], y[0]

        self._outputs = x + y + round_key + [ancillas]
        self._definition = circuit


class TraxlEnc(Gate, RegisterOperation):
    """A quantum implementation of the encryption step of the TRAX-L cipher.

    The default block size for this cipher is 256 bits.

    The Alzette rounds of TRAX can be implemented using different techniques:

    - :py:data:`lookahead-parallel`: Run the 4 instances of Alzette in parallel
      using carry lookahead adders.
    - :py:data:`lookahead-half-parallel`: Run 2 instances of Alzette in parallel
      using carry lookahead adders.
    - :py:data:`lookahead-sequential`: Run the 4 instances of Alzette sequentially
      using carry lookahead adders.
    - :py:data:`ripple`: Run the 4 instances of Alzette in parallel
      using ripple carry adders.

    The key schedule can be implemented using different techniques:

    - :py:data:`ripple`: Compute the key schedule
      using ripple carry adders for the non constant adders.
    - :py:data:`lookahead`: Compute the key schedule using only carry lookahead adders.

    Each combinaison of techniques involves a tradeoff between the circuit depth, the quantum cost
    and the number of needed ancilla qubits. Please read the `Grover on TRAX` paper
    for more information about this.

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
    def _get_num_ancilla_registers(
        alzette_mode: str = "lookahead-parallel",
        schedule_mode: str = "lookahead",
    ) -> int:
        """Get the numbers of ancilla registers used internally by TraxlEnc."""
        if alzette_mode == "lookahead-parallel":
            num_ancilla_registers = 4
        elif alzette_mode == "lookahead-half-parallel":
            num_ancilla_registers = 2
        elif alzette_mode in ("lookahead-sequential", "ripple"):
            num_ancilla_registers = 1
        else:
            msg = f"Unknown alzette mode {alzette_mode}."
            raise CircuitError(msg)
        if schedule_mode == "lookahead":
            num_ancilla_registers += 1
        elif schedule_mode != "ripple":
            msg = f"Unknown schedule mode {schedule_mode}."
            raise CircuitError(msg)

        return num_ancilla_registers

    @staticmethod
    def get_num_ancilla_qubits(
        n: int = 256,
        alzette_mode: str = "lookahead-parallel",
        schedule_mode: str = "lookahead",
    ) -> int:
        """Get the number of required ancilla qubits without instantiating the class.

        :param n: The block size of the cipher.
        :param alzette_mode: The method to use to compute the Alzette rounds.
        :returns: The number of ancilla qubits needed for the computation.
        """
        return RegisterDKRSCarryLookaheadAdder.get_num_ancilla_qubits(
            n // 8,
        ) * TraxlEnc._get_num_ancilla_registers(alzette_mode, schedule_mode)

    def __init__(
        self: TraxlEnc,
        x: list[QuantumRegister],
        y: list[QuantumRegister],
        key: list[QuantumRegister],
        tweak: list[int],
        ancillas: AncillaRegister,
        alzette_mode: str = "lookahead-parallel",
        schedule_mode: str = "lookahead",
        nsteps: int = 17,
        label: str | None = None,
    ) -> None:
        """Initialize TraxlEnc."""
        RegisterOperation.__init__(self)
        if len(x) != 4 or len(y) != 4 or len(key) != 8 or len(tweak) != 4:
            msg = "Wrong number of inputs."
            raise CircuitError(msg)
        self._inputs = x + y + key + [ancillas]
        self.n = len(x[0])  # Number of qubits in each register (by default 32)

        Gate.__init__(self, "Traxl_enc", self.n * 16 + len(ancillas), [], label=label)

        if len(ancillas) != self.get_num_ancilla_qubits(self.n * 8, alzette_mode, schedule_mode):
            msg = (
                "Circuit needs "
                f"{self.get_num_ancilla_qubits(self.n*8, alzette_mode, schedule_mode)} "
                f"ancilla qubits but {len(ancillas)} were given."
            )
            raise CircuitError(msg)

        x = x.copy()
        y = y.copy()
        key = key.copy()
        num_ancilla_registers = self._get_num_ancilla_registers(alzette_mode, schedule_mode)
        ancilla_registers = [
            AncillaRegister(
                bits=ancillas[
                    i * len(ancillas) // num_ancilla_registers : (i + 1)
                    * len(ancillas)
                    // num_ancilla_registers
                ],
            )
            for i in range(num_ancilla_registers)
        ]

        circuit = RegisterCircuit(*self.inputs, name="Traxl_enc")

        for s in range(nsteps):
            # Add tweak if step counter is odd
            if (s % 2) == 1:
                circuit.xor(x[0], tweak[0])
                circuit.xor(y[0], tweak[1])
                circuit.xor(x[1], tweak[2])
                circuit.xor(y[1], tweak[3])

            # Add subkeys and execute ALZETTEs
            for b in range(4):
                circuit.xor(x[b], key[2 * b])
                circuit.xor(y[b], key[2 * b + 1])
                if alzette_mode == "ripple":
                    circuit.append(
                        Alzette(
                            x[b],
                            y[b],
                            (RCON[(4 * s + b) % 8] % 2**self.n),
                            adder_mode="ripple",
                        ),
                        list(chain(x[b], y[b])),
                    )
                elif alzette_mode == "lookahead-sequential":
                    circuit.append(
                        Alzette(
                            x[b],
                            y[b],
                            (RCON[(4 * s + b) % 8] % 2**self.n),
                            ancilla_registers[0],
                            adder_mode="lookahead",
                        ),
                        list(chain(x[b], y[b], ancilla_registers[0])),
                    )
                elif alzette_mode == "lookahead-half-parallel":
                    circuit.append(
                        Alzette(
                            x[b],
                            y[b],
                            (RCON[(4 * s + b) % 8] % 2**self.n),
                            ancilla_registers[b // 2],
                            adder_mode="lookahead",
                        ),
                        list(chain(x[b], y[b], ancilla_registers[b // 2])),
                    )
                elif alzette_mode == "lookahead-parallel":
                    circuit.append(
                        Alzette(
                            x[b],
                            y[b],
                            (RCON[(4 * s + b) % 8] % 2**self.n),
                            ancilla_registers[b],
                            adder_mode="lookahead",
                        ),
                        list(chain(x[b], y[b], ancilla_registers[b])),
                    )
                else:
                    msg = f"Unknown alzette mode {alzette_mode}."
                    raise CircuitError(msg)

            # Linear layer (Sparkle256 permutation)
            half = self.n // 2
            for b in range(2):
                yb_0 = QuantumRegister(bits=y[b][0:half])
                yb_1 = QuantumRegister(bits=y[b][half:])
                x2_0 = QuantumRegister(bits=x[2][0:half])
                x2_1 = QuantumRegister(bits=x[2][half:])
                x3_0 = QuantumRegister(bits=x[3][0:half])
                x3_1 = QuantumRegister(bits=x[3][half:])

                # Here we decompose the sparkle permutation into XOR
                circuit.xor(yb_0, x2_1)
                circuit.xor(yb_0, x2_0)
                circuit.xor(yb_1, x2_0)
                circuit.xor(yb_0, x3_1)
                circuit.xor(yb_0, x3_0)
                circuit.xor(yb_1, x3_0)

            for b in range(2):
                xb_0 = QuantumRegister(bits=x[b][0:half])
                xb_1 = QuantumRegister(bits=x[b][half:])
                y2_0 = QuantumRegister(bits=y[2][0:half])
                y2_1 = QuantumRegister(bits=y[2][half:])
                y3_0 = QuantumRegister(bits=y[3][0:half])
                y3_1 = QuantumRegister(bits=y[3][half:])

                # Here we decompose the sparkle permutation into XOR
                circuit.xor(xb_0, y2_1)
                circuit.xor(xb_0, y2_0)
                circuit.xor(xb_1, y2_0)
                circuit.xor(xb_0, y3_1)
                circuit.xor(xb_0, y3_0)
                circuit.xor(xb_1, y3_0)

            x[0], x[1], x[2], x[3] = x[3], x[2], x[0], x[1]
            y[0], y[1], y[2], y[3] = y[3], y[2], y[0], y[1]

            # Compute key schedule
            if schedule_mode == "lookahead":
                circuit.add(key[0], key[1], ancilla_registers[-1], mode="lookahead")
                circuit.add(key[0], (RCON[(2 * s) % 8] % 2**self.n), ancilla_registers[-1])
            elif schedule_mode == "ripple":
                circuit.add(key[0], key[1], mode="ripple")
                circuit.add(key[0], (RCON[(2 * s) % 8] % 2**self.n), ancilla_registers[0])
            else:
                msg = f"Unknown schedule mode {schedule_mode}."
                raise CircuitError(msg)

            circuit.xor(key[2], key[3])
            circuit.xor(key[2], s % 2**self.n)

            if schedule_mode == "lookahead":
                circuit.add(key[4], key[5], ancilla_registers[-1], mode="lookahead")
                circuit.add(key[4], (RCON[(2 * s + 1) % 8] % 2**self.n), ancilla_registers[-1])
            elif schedule_mode == "ripple":
                circuit.add(key[4], key[5], mode="ripple")
                circuit.add(key[4], (RCON[(2 * s + 1) % 8] % 2**self.n), ancilla_registers[-1])
            else:
                msg = f"Unknown schedule mode {schedule_mode}."
                raise CircuitError(msg)

            circuit.xor(key[6], key[7])
            circuit.xor(key[6], ((s << self.n // 2) % 2**self.n))

            key.append(key.pop(0))

        # Add last round subkeys
        for b in range(4):
            circuit.xor(x[b], key[2 * b])
            circuit.xor(y[b], key[2 * b + 1])

        self._outputs = x + y + key + [ancillas]
        self._definition = circuit
