pycrypto
========

**pycrypto** is cryptography library implemented as quantum circuits using [Qiskit](https://qiskit.org/). It allows to easily implement classical cryptography functions to quantum computers to run quantum attacks on them, such as Grover's algorithm.

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


