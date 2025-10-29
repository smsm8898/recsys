# Recsys
- Implementation of state-of-the-arts Recommenders
- Compare 7 other Models in the same settings

## Prerequisites
- Python >= 3.9
- numpy >= 2.0.0
- pandas >= 2.3.0
- torch >= 2.8.0

## Usage
Install prerequisites with :
```bash
python -m venv recsys
pip install -r requirements.txt
```

## Dataset
- [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/)
  - [Review](https://github.com/smsm8898/recsys/blob/main/recsys/reviews/movielens100k.md)
```bash
mkdir -p recsys/data/ml-100k/raw
cd recsys/data/ml-100k/raw
wget https://files.grouplens.org/datasets/movielens/ml-100k/u.data
wget https://files.grouplens.org/datasets/movielens/ml-100k/u.item
wget https://files.grouplens.org/datasets/movielens/ml-100k/u.user
```

## Comparison Results
Before run the `main.py`, check the `config.yaml`
```python
python -m src.main.py
```
- 7 Model Train Metrics Comparison
![alt text](<7 Models Train Metrics.png>)

- 7 Model Ranking Metrics Comparison
![alt text](image.png)

## Project Resources
| 논문명 | 리뷰 | 모델 | 학습 |
| :------------- | :------------- | :---------- | :---------- |
| **[MF](https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf)** | [📖](https://velog.io/@smsm8898/Paper-Review-Matrix-Factoriztion)| **[<img src="https://github.githubassets.com/images/icons/emoji/octocat.png" width="20" height="20"> ](https://github.com/smsm8898/recsys)** | [🎥](https://github.com/smsm8898/recsys/blob/main/recsys/notebooks/mf.ipynb)|
| **[NCF](https://arxiv.org/abs/1708.05031)** | [📖](https://velog.io/@smsm8898/Paper-Review-Neural-Collaborative-Filtering) | **[<img src="https://github.githubassets.com/images/icons/emoji/octocat.png" width="20" height="20"> ](https://github.com/smsm8898/recsys/blob/main/recsys/models/ncf.py)** | [🎥](https://github.com/smsm8898/recsys/blob/main/recsys/notebooks/ncf.ipynb) |
| **[FM](https://www.ismll.uni-hildesheim.de/pub/pdfs/Rendle2010FM.pdf)** | [📖](https://velog.io/@smsm8898/Paper-Review-Factorization-Machines) | **[<img src="https://github.githubassets.com/images/icons/emoji/octocat.png" width="20" height="20"> ](https://github.com/smsm8898/recsys/blob/main/recsys/models/fm.py)** | [🎥](https://github.com/smsm8898/recsys/blob/main/recsys/notebooks/fm.ipynb) |
| **[Wide & Deep](https://arxiv.org/abs/1606.07792)** | [📖](https://velog.io/@smsm8898/Paper-Review-Wide-Deep-Learning-for-Recommender-System) | **[<img src="https://github.githubassets.com/images/icons/emoji/octocat.png" width="20" height="20"> ](https://github.com/smsm8898/recsys/blob/main/recsys/models/wd.py)** | [🎥](https://github.com/smsm8898/recsys/blob/main/recsys/notebooks/wd.ipynb) |
| **[DeepFM](https://arxiv.org/abs/1703.04247)** | [📖](https://velog.io/@smsm8898/Paper-Review-DeepFM) | **[<img src="https://github.githubassets.com/images/icons/emoji/octocat.png" width="20" height="20"> ](https://github.com/smsm8898/recsys/blob/main/recsys/models/deepfm.py)** | [🎥](https://github.com/smsm8898/recsys/blob/main/recsys/notebooks/deepfm.ipynb)  |
| **[Deep & Cross Network](https://arxiv.org/abs/1708.05123)** | [📖](https://velog.io/@smsm8898/Paper-Review-DCN) | **[<img src="https://github.githubassets.com/images/icons/emoji/octocat.png" width="20" height="20"> ](https://github.com/smsm8898/recsys/blob/main/recsys/models/dcn.py)** | [🎥](https://github.com/smsm8898/recsys/blob/main/recsys/notebooks/dcn.ipynb)|
| **[DLRM](https://arxiv.org/abs/1906.00091)** | [📖](https://velog.io/@smsm8898/Paper-Review-DLRM) | **[<img src="https://github.githubassets.com/images/icons/emoji/octocat.png" width="20" height="20"> ](https://github.com/smsm8898/recsys/blob/main/recsys/models/dlrm.py)** | [🎥](https://github.com/smsm8898/recsys/blob/main/recsys/notebooks/dlrm.ipynb) |

