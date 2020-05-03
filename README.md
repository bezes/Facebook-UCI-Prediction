# Facebook-UCI-Prediction

This project includes some basic data analysis on the UCI Facebook dataset.

It contains a basic exploratory data analysis with answers to some questions,
and a model for predicting `Lifetime post consumers` for a post,
based on the provided dataset variables.

## Files

- `README.md`: Short description and instructions
- `requirements.txt`: Python library dependencies
- `preparation.py`: Python module with data download and preparation functions 
- `model.py`: Python module with functions for modeling the `Lifetime post consumers` variable.
- `EDA.ipynb`: Part 1 of the analysis: answering some basic questions on the data set.
- `Model.ipynb`: Part 2 of the analysis: Modelling `Lifetime Post Consumers`

## Instructions

Requires`python 3.7` or higher. You need to create a virtual environment:

```
python3 -m venv /path/to/virtual/environment
```

Activate the virtual environment, and install the libraries required using:
```
source /path/to/virtual/environment/bin/activate
pip install -r requirements.txt
```

You need to add the virtual environment as a jupyter kernel, in order to be able to select it in Jupyter:

```
ipython kernel install --name "venv" --user
```

Then launch Jupyter notebook:
```
jupyter notebook
```

You can then open and run the notebooks [EDA.ipynb](https://github.com/bezes/Facebook-UCI-Prediction/blob/master/EDA.ipynb) and [model.ipynb](https://github.com/bezes/Facebook-UCI-Prediction/blob/master/Model.ipynb).


## Source

The data for this analysis are available here:

https://archive.ics.uci.edu/ml/datasets/Facebook+metrics

Moro, S., Rita, P., & Vala, B. (2016). Predicting social media performance metrics and evaluation of the impact on brand building: A data mining approach. Journal of Business Research, 69(9), 3341-3351.


