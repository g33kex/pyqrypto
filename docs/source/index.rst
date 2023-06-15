Welcome to Qrypto's documentation!
==================================

**Qrypto** is a library of reversible quantum circuits for basic functions used in classical cryptography. As of now it implements the ARX-box Alzette and its necessary gates.

Because of the underlying architecture, right and left rotations are implemented for free without using any gate. 

This library is based on `Qiskit <https://qiskit.org/>`_.

Implemented Gates
-----------------

These are the quantum gates currently implemented by this projects.

- ``Alzette`` - a 64 bit ARX-box
- ``ADD`` - ripple carry adder
- ``XOR`` - bitwise XOR
- ``XORc`` - bitwise XOR with a constant
- ``ROR`` - right rotation (gate free)
- ``ROL`` - left rotation (gate free)

More gates are planned: shifts, AND, OR, two's complement, ...

Examples
--------

Here is an example on how to use **Qrypto** to generate a simple circuit that operates on qubit vectors.

.. code-block:: python

    from rOperations import rCircuit
    from qiskit import QuantumRegister

    # Create two 4 qubit registers
    X1 = QuantumRegister(4, name='X')
    Y1 = QuantumRegister(4, name='Y')

    # Create a register circuit from registers X and Y. 
    # A register circuit can operate on quantum registers instead of on individual qubits.
    qc = rCircuit(X1, Y1, name='Register Circuit')

    # Rotate left register X by 3 qubits. 
    # A register can be seen as a "view" on logical qubits, 
    # so rotating a register just yields another view on these qubits with swapped indexes.
    X2 = qc.ror(X1, 3)

    # Let's XOR X2 Y1. The result will be stored in X3.
    # Note here that X3 = X2 because xor doesn't modify the view.
    X3 = qc.xor(X2, Y1)

    # If we print the resulting circuit, we can see the XOR was done on a rotated version of X1.
    print(qc.decompose())
    #          ┌───┐          
    #X_0: ─────┤ X ├──────────
    #          └─┬─┘┌───┐     
    #X_1: ───────┼──┤ X ├─────
    #            │  └─┬─┘┌───┐
    #X_2: ───────┼────┼──┤ X ├
    #     ┌───┐  │    │  └─┬─┘
    #X_3: ┤ X ├──┼────┼────┼──
    #     └─┬─┘  │    │    │  
    #Y_0: ──■────┼────┼────┼──
    #            │    │    │  
    #Y_1: ───────■────┼────┼──
    #                 │    │  
    #Y_2: ────────────■────┼──
    #                      │  
    #Y_3: ─────────────────■──

For an example on how to implement a more complex circuit, please read the source code of :py:class:`alzette.Alzette`. 

This library also provides handy tools to add preparation steps and measurements to a circuit operating on registers. The registers can be initialized to an integer initial value, and measurement gates can be automatically added and the values of the output of the classical registers converted to integers. This allows testing the circuit on real data to make sure the implementation is correct.

Let's add a preparation and a measurement step to our previous example and simulate it.

.. code-block:: python

    from rOperations import make_circuit, run_circuit

    # Create a final circuit with measurement and preparation steps
    # Let's initialize X1 to 4 and Y1 to 11
    # Let's also measure X2 and Y1 at the end of the circuit
    # It is possible to measure any QuantumRegistrer that has its qubits in the circuit
    # This means we can measure the final state of any "view" of the qubits
    # However note that the qubits will be in their final state during the measurement
    # If we tried to measure X1, we wouldn't get the initial value of X1
    # but the value of X2 left rotated by 3 bits
    # This is because the value of X1 was overwritten by the XOR operation.
    # This is why it is important to keep track of your registers during operations!
    final_circuit = make_circuit(qc, [4, 11], [X1, Y1], [X2, Y1])

    # We can print the final circuit
    # As you can see the measurements are done on
    print(final_circuit)
    #      ┌─────────────┐ ┌───────┐   ┌─┐                  
    # X_0: ┤0            ├─┤1      ├───┤M├──────────────────
    #      │             │ │       │   └╥┘┌─┐               
    # X_1: ┤1            ├─┤2      ├────╫─┤M├───────────────
    #      │  rPrepare 4 │ │       │    ║ └╥┘┌─┐            
    # X_2: ┤2            ├─┤3      ├────╫──╫─┤M├────────────
    #      │             │ │       │┌─┐ ║  ║ └╥┘            
    # X_3: ┤3            ├─┤0      ├┤M├─╫──╫──╫─────────────
    #      ├─────────────┴┐│  Rxor │└╥┘ ║  ║  ║ ┌─┐         
    # Y_0: ┤0             ├┤4      ├─╫──╫──╫──╫─┤M├─────────
    #      │              ││       │ ║  ║  ║  ║ └╥┘┌─┐      
    # Y_1: ┤1             ├┤5      ├─╫──╫──╫──╫──╫─┤M├──────
    #      │  rPrepare 11 ││       │ ║  ║  ║  ║  ║ └╥┘┌─┐   
    # Y_2: ┤2             ├┤6      ├─╫──╫──╫──╫──╫──╫─┤M├───
    #      │              ││       │ ║  ║  ║  ║  ║  ║ └╥┘┌─┐
    # Y_3: ┤3             ├┤7      ├─╫──╫──╫──╫──╫──╫──╫─┤M├
    #      └──────────────┘└───────┘ ║  ║  ║  ║  ║  ║  ║ └╥┘
    #c0: 4/══════════════════════════╩══╩══╩══╩══╬══╬══╬══╬═
    #                                0  1  2  3  ║  ║  ║  ║ 
    #c1: 4/══════════════════════════════════════╩══╩══╩══╩═
    #                                            0  1  2  3

    # We can generate the QASM of that circuit and save it to a file
    # This can be used to run the circuit on an actual quantum computer
    qc.qasm(filename='circuit.qasm')

    # Let's run this circuit in a simulation and gather the result
    results = run_circuit(qc)

    # We can verify that ror(4, 3)^11 = 3, and that Y1 was unchanged.
    print(results)
    # [3, 11]

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   rOperations
   alzette

