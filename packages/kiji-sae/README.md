# kiji-sae

`kiji-sae` provides the core JumpReLU sparse autoencoder implementation used by Kiji Inspector.

## Install

```bash
pip install kiji-sae
```

## What is included

- `JumpReLUSAE`: the sparse autoencoder model
- `JumpReLUFunction`: the activation function used by the model

## Quick start

```python
from kiji_sae import JumpReLUSAE

sae = JumpReLUSAE(
    d_model=512,
    d_sae=4096,
)
```

## Repository

Source code and issue tracker:

- Repository: https://github.com/dataiku/kiji-inspector
- Package directory: https://github.com/dataiku/kiji-inspector/tree/main/packages/kiji-sae
- Issues: https://github.com/dataiku/kiji-inspector/issues

## License

Apache License 2.0
