pycrypto
========

**pycrypto** is cryptography library composed of quantum circuits defined with [Qiskit](https://qiskit.org/). It allows to easily implement classical cryptography functions on quantum computers to run quantum attacks on them, such as quantum brute-force with Grover's algorithm.

## Documentation

Read the documentation [here](https://g33kex.github.io/pyqrypto/build/html/index.html).

## Building

Make sure [poetry](https://python-poetry.org/) is installed.

Here are the steps to package this library.
```
poetry build
```

Follow theses steps to build the Sphinx documentation.
```
poetry install
poetry run sphinx-build docs/source/ docs/build/html
```

## Test

Run these commands to test the library:
```
poetry install
poetry run pytest
```

## Contributing

This is an early version of the library, but contributions are always welcome.
