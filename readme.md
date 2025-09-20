
# Recommender System 구현

### 1. Download movielens 100k
```
mkdir -p recsys/data/movielens
cd recsys/data/movielnes
wget http://files.grouplens.org/datasets/movielens/ml-100k/u.data
wget http://files.grouplens.org/datasets/movielens/ml-100k/u.item
wget http://files.grouplens.org/datasets/movielens/ml-100k/u.user
```

### 2. Train matrix factorization
```
python -m recsys.models.mf.train
```
