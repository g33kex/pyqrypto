## Here is a python script to reproduce the experiments showcased in the Grover on TRAX paper.
from qiskit import QuantumRegister
from qiskit.circuit.quantumregister import AncillaRegister
from pyqrypto.rOperations import rCircuit, rDKRSCarryLookaheadAdder, rConstantDKRSCarryLookaheadAdder
from pyqrypto.sparkle import Traxl_enc, Alzette, RCON
from itertools import chain, product
import random
from math import log2, floor
random.seed(42)

def w(s):
    """Returns the Hadamard weight of s."""
    return bin(s).count('1')

def w2(s, n):
    """Returns the Hadamard weight of s, not taking into account the first and last bit."""
    return w(s >> 1 & (2**(n-2)-1))

def compare_adders(n):
    """This function compares the depths of the ripple carry and carry lookahead adders on n bits"""
    A = QuantumRegister(n)
    B = QuantumRegister(n)

    qc1 = rCircuit(A, B)
    qc1.add(A, B, mode='ripple')
    print("Ripple:\t\t", qc1.stats)

    ancillas = AncillaRegister(rDKRSCarryLookaheadAdder.get_num_ancilla_qubits(n))
    qc2 = rCircuit(A, B, ancillas)
    qc2.add(A, B, ancillas, mode='lookahead')
    print("Lookahead:\t", qc2.stats)

def compare_trax(n=256, trax_nsteps=4):
    """This function compares different implementations of TRAX"""
    X = [QuantumRegister(n//8, name=f'X{i}') for i in range(4)]
    Y = [QuantumRegister(n//8, name=f'Y{i}') for i in range(4)]
    K = [QuantumRegister(n//8, name=f'K{i}') for i in range(8)]
    tweak = [2**(n//8)-1 for _ in range(4)]

    for alzette_mode, schedule_mode in product(['lookahead-parallel', 'lookahead-half-parallel', 'lookahead-sequential', 'ripple'], ['ripple']):
        ancillas = AncillaRegister(Traxl_enc.get_num_ancilla_qubits(n, alzette_mode, schedule_mode))
        qc = rCircuit(*X, *Y, *K, ancillas)
        gate = Traxl_enc(X, Y, K, tweak, ancillas, alzette_mode=alzette_mode, schedule_mode=schedule_mode, trax_nsteps=trax_nsteps)
        qc.append(gate, list(chain(*gate.inputs)))
        print(f"TRAX {alzette_mode}/{schedule_mode}:", qc.stats)

def compare_single_trax(n=256):
#     """Compute stats for a single round of trax"""
    X = [QuantumRegister(n//8, name=f'X{i}') for i in range(4)]
    Y = [QuantumRegister(n//8, name=f'Y{i}') for i in range(4)]
    K = [QuantumRegister(n//8, name=f'K{i}') for i in range(8)]
    tweak = [0 for _ in range(4)]
    for alzette_mode in ['lookahead-parallel', 'lookahead-half-parallel', 'lookahead-sequential', 'ripple']:
        for schedule_mode in ['ripple', 'lookahead']:
            ancillas = AncillaRegister(Traxl_enc.get_num_ancilla_qubits(n, alzette_mode, schedule_mode))
            qc = rCircuit(*X, *Y, *K, ancillas)
            gate = Traxl_enc(X, Y, K, tweak, ancillas, alzette_mode=alzette_mode, schedule_mode=schedule_mode, trax_nsteps=1)
            qc.append(gate, list(chain(*gate.inputs)))
            print(f"Single TRAX {alzette_mode}/{schedule_mode}", qc.stats)
#
def linear_layer(n=32, print_circuit=False):
        # Linear layer (Sparkle256 permutation)
        x = [QuantumRegister(n, name=f'X{i}') for i in range(4)]
        y = [QuantumRegister(n, name=f'Y{i}') for i in range(4)]
        qc = rCircuit(*x, *y)
        half = n//2
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
        if print_circuit:
            print(qc)
        print("Linear layer:", qc.stats)

def compare_sparkle(n=32):
    """Evaluate a sparkle round composed of 4 alzette and a linear layer"""
    x = [QuantumRegister(n, name=f'X{i}') for i in range(4)]
    y = [QuantumRegister(n, name=f'Y{i}') for i in range(4)]
    for alzette_mode in ['ripple', 'lookahead-sequential', 'lookahead-half-parallel', 'lookahead-parallel']:
        num_ancilla_registers = 0
        if alzette_mode == 'lookahead-sequential':
            num_ancilla_registers = 1
        elif alzette_mode == 'lookahead-half-parallel':
            num_ancilla_registers = 2
        elif alzette_mode == 'lookahead-parallel':
            num_ancilla_registers = 4
        ancilla_registers = [AncillaRegister(rDKRSCarryLookaheadAdder.get_num_ancilla_qubits(n)) for _ in range(num_ancilla_registers)]
        qc = rCircuit(*x, *y, *ancilla_registers)
         # Add subkeys and execute ALZETTEs
        s = 0
        for b in range(4):
             if alzette_mode == 'ripple':
                 qc.append(Alzette(x[b], y[b], (RCON[(4*s+b)%8]%2**n), adder_mode='ripple'), list(chain(x[b], y[b])))
             elif alzette_mode == 'lookahead-sequential':
                 qc.append(Alzette(x[b], y[b], (RCON[(4*s+b)%8]%2**n), ancilla_registers[0], adder_mode='lookahead'), list(chain(x[b], y[b], ancilla_registers[0])))
             elif alzette_mode == 'lookahead-half-parallel':
                 qc.append(Alzette(x[b], y[b], (RCON[(4*s+b)%8]%2**n), ancilla_registers[b//2], adder_mode='lookahead'), list(chain(x[b], y[b], ancilla_registers[b//2])))
             elif alzette_mode == 'lookahead-parallel':
                 qc.append(Alzette(x[b], y[b], (RCON[(4*s+b)%8]%2**n), ancilla_registers[b], adder_mode='lookahead'), list(chain(x[b], y[b], ancilla_registers[b])))
        

        print(f"Alzette {alzette_mode}:", qc.stats)
        # Linear layer (Sparkle256 permutation)
        half = n//2
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
        print(f"Sparkle {alzette_mode}:", qc.stats)

# TODO: add compute keyschedule with same ancilla qubits or not
def compare_keyschedule(n=32):
    """Evaluates the key schedule of TRAX"""
    key = [QuantumRegister(n, name=f'K{i}') for i in range(8)]
    s = 0
    for schedule_mode in ['ripple']:#, 'lookahead']:
        for parallel in [True, False]:
            if parallel:
                num_ancilla_registers = 2
            else:
                num_ancilla_registers = 1
            ancilla_registers = [AncillaRegister(rDKRSCarryLookaheadAdder.get_num_ancilla_qubits(n)) for _ in range(num_ancilla_registers)]
            qc = rCircuit(*key, *ancilla_registers)

            if schedule_mode == 'lookahead':
                qc.add(key[0], key[1], ancilla_registers[0], mode='lookahead')
                qc.add(key[0], (RCON[(2*s)%8]%2**n), ancilla_registers[0])
            elif schedule_mode == 'ripple':
                # swap these two lines for optimal/suboptimal order of key schedule
                qc.add(key[0], key[1], mode='ripple')
                qc.add(key[0], (RCON[(2*s)%8]%2**n), ancilla_registers[0])

            qc.xor(key[2], key[3])
            qc.xor(key[2], s%2**n)

            if schedule_mode == 'lookahead':
                qc.add(key[4], key[5], ancilla_registers[-1], mode='lookahead')
                qc.add(key[4], (RCON[(2*s+1)%8]%2**n), ancilla_registers[-1])
            elif schedule_mode == 'ripple':
                qc.add(key[4], key[5], mode='ripple')
                qc.add(key[4], (RCON[(2*s+1)%8]%2**n), ancilla_registers[-1])

            qc.xor(key[6], key[7])
            qc.xor(key[6], ((s << n//2) % 2**n))

            print(f"Key schedule {schedule_mode}-{'parallel' if parallel else 'sequential'}:", qc.stats)

def compare_alzette(n=32):
    """This function compares different implementations of Alzette"""
    c = 0xffffffff

    for adder_mode in ['ripple', 'lookahead']:
        X = QuantumRegister(n, name='X')
        Y = QuantumRegister(n, name='Y')
        ancillas = AncillaRegister(Alzette.get_num_ancilla_qubits(n, adder_mode=adder_mode))
        qc = rCircuit(X, Y, ancillas)
        gate = Alzette(X, Y, c, ancillas, adder_mode=adder_mode)
        qc.append(gate, list(chain(*gate.inputs)))

        print(f"Alzette {adder_mode}:", qc.stats)

def test_trax(n=32):
    x = [QuantumRegister(n, name=f'X{i}') for i in range(4)]
    y = [QuantumRegister(n, name=f'Y{i}') for i in range(4)]
    key = [QuantumRegister(n, name=f'K{i}') for i in range(8)]
    tweak = [2**n-1 for _ in range(4)]
    num_ancilla_registers = 4
    ancillas = AncillaRegister(rDKRSCarryLookaheadAdder.get_num_ancilla_qubits(n)*num_ancilla_registers)

    alzette_mode = 'lookahead-parallel'
    do_alzette = True
    do_schedule = True
    do_linear = True
    
    ancillas_parallel = False
    
    extra_registers_start = 0
    if ancillas_parallel:
        if alzette_mode == 'lookahead-parallel':
            extra_registers_start=4
        elif alzette_mode == 'lookahead-half-parallel':
            extra_registers_start=2
        elif alzette_mode == 'lookahead-sequential':
            extra_registers_start=1

    for trax_nsteps in [1,2]:
        for schedule_mode in ['ripple', 'lookahead-parallel']:
            qc = rCircuit(*x, *y, *key, ancillas)
            ancilla_registers = [AncillaRegister(bits=ancillas[i*len(ancillas)//num_ancilla_registers:(i+1)*len(ancillas)//num_ancilla_registers]) for i in range(num_ancilla_registers)]

            for s in range(trax_nsteps):

                # Add tweak if step counter is odd
                # if ((s % 2) == 1):
                #     qc.xor(x[0], tweak[0])
                #     qc.xor(y[0], tweak[1])
                #     qc.xor(x[1], tweak[2])
                #     qc.xor(y[1], tweak[3])
                #

                if do_alzette:
                    for b in range(4):
                        qc.xor(x[b], key[2*b])
                        qc.xor(y[b], key[2*b+1])
                        if alzette_mode == 'ripple':
                            qc.append(Alzette(x[b], y[b], (RCON[(4*s+b)%8]%2**n), adder_mode='ripple'), list(chain(x[b], y[b])))
                        elif alzette_mode == 'lookahead-sequential':
                            qc.append(Alzette(x[b], y[b], (RCON[(4*s+b)%8]%2**n), ancilla_registers[0], adder_mode='lookahead'), list(chain(x[b], y[b], ancilla_registers[0])))
                        elif alzette_mode == 'lookahead-half-parallel':
                            qc.append(Alzette(x[b], y[b], (RCON[(4*s+b)%8]%2**n), ancilla_registers[b//2], adder_mode='lookahead'), list(chain(x[b], y[b], ancilla_registers[b//2])))
                        elif alzette_mode == 'lookahead-parallel':
                            qc.append(Alzette(x[b], y[b], (RCON[(4*s+b)%8]%2**n), ancilla_registers[b], adder_mode='lookahead'), list(chain(x[b], y[b], ancilla_registers[b])))
                if do_linear:
                    # Linear layer (Sparkle256 permutation)
                    half = n//2
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

                # # Compute key schedule
                if do_schedule:
                    if schedule_mode in ['lookahead-parallel', 'lookahead-sequential']:
                        qc.add(key[0], key[1], ancilla_registers[extra_registers_start], mode='lookahead')
                        qc.add(key[0], (RCON[(2*s)%8]%2**n), ancilla_registers[extra_registers_start])
                    elif schedule_mode == 'ripple':
                        qc.add(key[0], key[1], mode='ripple')
                        qc.add(key[0], (RCON[(2*s)%8]%2**n), ancilla_registers[extra_registers_start])

                    qc.xor(key[2], key[3])
                    qc.xor(key[2], s%2**n)

                    if schedule_mode == 'lookahead-sequential':
                            qc.add(key[4], key[5], ancilla_registers[extra_registers_start], mode='lookahead')
                            qc.add(key[4], (RCON[(2*s+1)%8]%2**n), ancilla_registers[extra_registers_start])
                    if schedule_mode == 'lookahead-parallel':
                            qc.add(key[4], key[5], ancilla_registers[extra_registers_start+1], mode='lookahead')
                            qc.add(key[4], (RCON[(2*s+1)%8]%2**n), ancilla_registers[extra_registers_start+1])
                    elif schedule_mode == 'ripple':
                        qc.add(key[4], key[5], mode='ripple')
                        qc.add(key[4], (RCON[(2*s+1)%8]%2**n), ancilla_registers[extra_registers_start+1])

                    qc.xor(key[6], key[7])
                    qc.xor(key[6], ((s << n//2) % 2**n))

                    key.append(key.pop(0))

            # # Add last round subkeys
            # for b in range(4):
            #     qc.xor(x[b], key[2*b])
            #     qc.xor(y[b], key[2*b+1])
            print(f"TRAX[{trax_nsteps}]: {alzette_mode}/{schedule_mode}", qc.stats['depth'])

def evaluate_constant_adder():
    for n in range(5, 20):
        for c in range(1,n):
            X = QuantumRegister(n, 'X')
            ancillas = AncillaRegister(rConstantDKRSCarryLookaheadAdder.get_num_ancilla_qubits(n))
            qc = rCircuit(X, ancillas)
            qc.add(X, c, ancillas)
            stats = qc.stats
            print(f"n={n}, c={c}\ttoffoli={stats['gate_counts']['ccx']}, cnot={stats['gate_counts']['cx']}, not={stats['gate_counts']['x']}, cost={stats['quantum_cost']}, depth={stats['depth']}")
            print(f"true:\t\ttoffoli={8*n-6*w(n-1)-6*floor(log2(n-1))-10}, cnot={n+2*w(c)-1}, not={2*n-2+w(c)+2*w2(c,n)}, cost={43*n-30*w(n-1)-30*floor(log2(n-1)) + 3*w(c) + 2*w2(c,n) - 53}, depth={2*floor(log2(n-1))+2*floor(log2((n-1)/3))+14}")
            print("---")

   
# compare_adders(32)
#evaluate_constant_adder()
#compare_trax(trax_nsteps=2)
compare_trax()
#compare_alzette()
#compare_sparkle()
#compare_keyschedule()
#linear_layer(32, print_circuit=False)
#linear_layer(2, print_circuit=True)
#test_trax()
