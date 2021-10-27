# py-daml

Learn how to use NumPy, Pandas, Seaborn , Matplotlib , Plotly , Scikit-Learn , Machine Learning, Tensorflow , and more

## Environment

This repository uses anaconda for virtual environement and vscode as "IDE" the vscode settings are as follows.
"pylint" and "black" libraries are needed for formatting purpose

```json
{
  "python.linting.pylintArgs": ["--load-plugins", "pylint_django"],
  "python.linting.pylintEnabled": true,
  "python.linting.enabled": true
}
```

The steps to create a new environment and libraries are as follows

```sh
conda create -n m1
conda activate m1
export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1
conda env update -f requirements.yaml
```
