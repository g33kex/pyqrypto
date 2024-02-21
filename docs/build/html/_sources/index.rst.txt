Welcome to pyqrypto's documentation!
====================================

**pyrypto** is a library of reversible quantum circuits for basic functions used in classical cryptography. As of now it implements the ARX-box Alzette and the associated tweakable block cipher TRAX from the `Sparkle-suite <https://sparkle-lwc.github.io/>`_ and their necessary gates. It is easily expandable to more ciphers.

This library provides high level operations on quantum registers by tracking the individual qubits. Because of this, permutations can be implemented for free. Additionally, the quantum cost of a circuit containing these operations can be automatically computed.

This library is based on `Qiskit <https://qiskit.org/>`_.

Implemented Gates
-----------------

These are the quantum gates currently implemented by this project.

- :py:class:`~pyqrypto.sparkle.TraxlEnc` - TRAX-L encryption
- :py:class:`~pyqrypto.sparkle.Alzette` - a 64 bit ARX-box
- :py:class:`~pyqrypto.register_operations.RegisterTTKRippleCarryAdder` - ripple carry adder
- :py:class:`~pyqrypto.register_operations.RegisterDKRSCarryLookaheadAdder` - carry lookahead adder
- :py:class:`~pyqrypto.register_operations.RegisterConstantDKRSCarryLookaheadAdder`` - carry lookahead adder with a constant
- :py:class:`~pyqrypto.register_operations.RegisterXOR` - bitwise XOR
- :py:class:`~pyqrypto.register_operations.RegisterConstantXOR` - bitwise XOR with a constant
- :py:class:`~pyqrypto.register_operations.RegisterNOT` - bitwise NOT
- :py:class:`~pyqrypto.register_operations.RegisterROR` - right rotation (gate free)
- :py:class:`~pyqrypto.register_operations.RegisterROL` - left rotation (gate free)

Examples
--------

Here are some usage examples for **pyqrypto**. Checkout the  `examples folder <https://github.com/g33kex/pyqrypto/tree/main/examples>`_ for more.

Simple Usage
~~~~~~~~~~~~

Here is an example on how to use **pyqrypto** to generate a simple circuit that operates on qubit vectors.

.. literalinclude:: ../../examples/simple_example.py
    :language: python
    :lines: 2-39

The library also provides handy tools to add preparation steps and measurements to a circuit operating on quantum registers. The registers can be initialized to an integer initial value, and measurement gates can be automatically added and the values of the output of the classical registers converted to integers. This allows testing the circuit on real data to make sure the implementation is correct.

Let's add a preparation and a measurement step to our previous example and simulate it. 

.. note ::
        You need to install the `qiskit-aer <https://pypi.org/project/qiskit-aer/>`_ extra dependency to use the simulation feature.

.. literalinclude:: ../../examples/simple_example.py
    :language: python
    :lines: 41-91

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Contents:

   register_operations
   sparkle

Addition
~~~~~~~~

Here is an example on how to do a 8-bit addition using a ripple-carry adder:

.. literalinclude:: ../../examples/adders_example.py
    :language: python
    :dedent: 4
    :lines: 17-36

It is also possible to use a carry-lookahead adder to obtain a shallower circuit, at the expense of ancilla qubits:

.. literalinclude:: ../../examples/adders_example.py
    :language: python
    :dedent: 4
    :lines: 41-66

Alzette
~~~~~~~

Here's how to use the :py:class:`~pyqrypto.sparkle.Alzette` gate:

.. literalinclude:: ../../examples/alzette_example.py
    :language: python
    :lines: 2-38

TRAX-L
~~~~~~

Here's how to use the :py:class:`~pyqrypto.sparkle.TraxlEnc` gate:

.. literalinclude:: ../../examples/traxl_example.py
    :language: python
    :lines: 2-61

