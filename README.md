# ML4ProM

Please follow the notebooks to reproduce results:
- [./notebooks/1_EDA.ipynb](./notebooks/1_EDA.ipynb) downloads datasets and does **E**xploratory **D**ata **A**nalysis for each dataset to understand datasets better,
- [./notebooks/2_training.ipynb](./notebooks/2_training.ipynb) executes training scripts and presents results,
- [./notebooks/3_post-train.ipynb](./notebooks/3_post-train.ipynb) presents feature importances for each dataset and each ML model.


### How to train models and output results?
Inside the project directory (../ml4prom/) execute following to get to know more about the args:
```python
python -m src.models.train_model -h
```
which returns:
```
  --debug DEBUG         When True, plots ROC-Curve & Confusion Matrix
  --seq_encoding SEQ_ENCODING
                        Possible encodings; 'one-hot' & 'n-gram' where n is an integer
  --unique_traces UNIQUE_TRACES
                        when True, duplicate traces(trace variants) are removed from dataset
  --remove_biased_feats REMOVE_BIASED_FEATS
                        when True, the biased features are removed from dataset, e.g. patient is dead in COVID dataset
```

The following command does multiple things:
- load all datasets
- apply preprocessing, e.g. remove biased features, remove duplicate traces, etc.
- encode traces (sequence of events)
- train ML models with StratifiedKFold cross-validation
- output a `.csv` file to `./reports/` including the accuracy scores
```python
python -m src.models.train_model --seq_encoding one-hot --remove_biased_feats --unique_traces
```



---
#### Future work
- time-series split for CV, see [scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)
- Encode data after train-test split! [see example code](https://stackoverflow.com/questions/55525195/do-i-have-to-do-one-hot-encoding-separately-for-train-and-test-dataset)
- Check out [SHAP values](https://github.com/slundberg/shap)
---