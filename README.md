# ML4ProM
Check out the paper on: [![arXiv](https://img.shields.io/badge/arXiv-2212.00695-b31b1b.svg)](https://arxiv.org/abs/2212.00695)

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

<h2><b> Citation: </b></h2>

```
@inproceedings{velioglu2022explainable,
  title={Explainable Artificial Intelligence for Improved Modeling of Processes},
  author={Velioglu, Riza and G{\"o}pfert, Jan Philip and Artelt, Andr{\'e} and Hammer, Barbara},
  booktitle={International Conference on Intelligent Data Engineering and Automated Learning},
  pages={313--325},
  year={2022},
  organization={Springer}
}
```
