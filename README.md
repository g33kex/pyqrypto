pycrypto
========

**pycrypto** is cryptography library composed of quantum circuits defined with [Qiskit](https://qiskit.org/). It allows to easily implement classical cryptography functions on quantum computers to run quantum attacks on them, such as quantum brute-force with Grover's algorithm.

## Documentation

See the `docs` folder.

TODO: upload documentation on github.io

## Packaging

Here are the steps to package this library.

```
pip install -r reqs.development
python3 -m build
```

Follow theses steps to build the Sphinx documentation.

```
pip install -r reqs.development
cd docs
make html
```


