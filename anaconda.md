# Install Anaconda and Pip Packages

- Install Anaconda: <https://www.anaconda.com/download>
- Then run the following commands in the terminal:

```bash

conda create -n ai python=3.10 -y && \
conda activate ai && \
grep -v "pypi_0" requirements.txt | sed 's/#.*//' | xargs conda install -y && \
grep "pypi_0" requirements.txt | sed 's/=pypi_0//' | sed 's/=/==/' | xargs pip install && \
conda list -e > requirements.txt

```
