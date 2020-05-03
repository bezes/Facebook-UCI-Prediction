# Facebook-UCI-Prediction

## Files

- `README.md`

Instructions and short description.

- `requirements.txt`

Python library dependencies

- `preparation.py`

Python module with data download and preparation functions 

- `model.py`

Python module with functions for modeling the `Lifetime post consumers` variable.

- EDA.ipynb

Part 1 of the analysis: answering some basic questions on the data set.

- Model.ipynb

Part 2 of the analysis: Modelling `Lifetime Post Consumers`

## How to execute:

Requirements: python 3.7 +

Create and activate the virtual environment:

```
python3 -m venv /path/to/virtual/environment
source /path/to/virtual/environment/bin/activate
```

While on the virtual environment, install the libraries required using:
`
pip install -r requirements.txt
`

You need to add the virtual environment as a jupyter kernel, in order to be able to select it in Jupyter.

`ipython kernel install --name "venv" --user`

Then launch Jupyter:
`jupyter notebook`

You can then open and run the notebooks `EDA.ipynb` and `model.ipynb`.
