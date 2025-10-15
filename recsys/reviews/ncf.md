> 본 글은 2017년 발표된 Neural Collaborative Filtering을 읽고 요약 및 정리한 글입니다.

___
## 0. 핵심요약
- 딥러닝을 추천 시스템에 성공적으로 도입
- [MF] 전통적 행렬분해의 한계
    - **내적(Inner Product)의 한계**
        - user-item interaction을 단순 내적으로만 모델링
        - 선형적 패턴만 포착
- [NeuMF] 딥러닝 모델을 효과적 도입
    - GMF: user-item의 **저차원 interaction**을 모델링
    - MLP: 다층 딥러닝을 통해 **고차원 interaction**을 모델링
    - NeuMF: GMF와 MLP를 **Joint Training**


## 1. Introduction
- 정보의 홍수 시대에서 추천 시스템은 굉장히 중요
    - 특히 MF로 대표되는 Collaborative filtering
- Netflix Prize를 시작으로 MF는 여러가지 방법으로 발전
    - **Inner product**를 이용해서 user-item interaction을 모델링
    - **complex structure**를 모두 잡기엔 충분하지 않음
- Deep Neural Network를 이용해서 이를 해결하고자 함
    - 다른 연구에서는 DNN을 이용해서 auxiliary information을 모델링하고자 함
    - 여전히 CF 기반 모델은 Inner product에 의존
- 해당 연구에서는 **DNN**을 Collaborative Ciltering 문제 해결에 사용

## 2. Preliminaries
#### 2.1. Learning from Implicit Data
- $y_{ui} = 1$: Observed data(interaction)
    - user가 item을 실제로 좋아하는 지 알 수 없음
    - noisy signal
- $y_{ui} = 0$: Unonserved data
    - 추천은 unobserved entries를 추론하는 문제
    - missing value
- Machine Learning
    - Pointwise learning:  $\min (\hat y_{ui} - y_{ui})$
    - Pairwise learning:  $\max (\hat y_{ui} - \hat y_{uj})$

#### 2.2. Matrix Factorization
- MF는 latent vector의 inner product(Linear model)
$$
\hat y_{ui} \;=\; f(u, i \mid \mathbf{p}_u, \mathbf{q}_i) \;=\; \mathbf{p}_u^\top \mathbf{q}_i \;=\; \sum_{k=1}^K p_{u k}\, q_{i k}
$$
![Figure1](https://velog.velcdn.com/images/smsm8898/post/2f6340ee-27ac-4dcf-909a-aa878fde55eb/image.png)
> **MF(Inner Product)의 한계**
(a)를 통해 $u_4$는 $u_1$, $u_3$$ $u_2$ 순으로 유사
하지만 latent space(b)에서 $p_4$를 $p_1$과 가장 가깝게 놓으면
$p_4$가 $p_3$보다 $p_2$에 더 가까워지고 더 큰 ranking loss 발생

## 3. Neural Collaborative Filtering
#### 3.1. General Framework
![Figure2](https://velog.velcdn.com/images/smsm8898/post/aae75fbc-6a7b-4218-85c5-67e803e006a5/image.png)
- Predictive model
$$
\hat y_{ui} = f(\mathbf{P}^T \mathbf{v}^U_u, \mathbf{Q}^T \mathbf{v}^I_i \;\mid\;\mathbf{P}, \mathbf{Q}, \theta_f)
$$
- NCF with MLP
$$
f(\mathbf{P}^T \mathbf{v}^U_u, \mathbf{Q}^T \mathbf{v}^I_i) 
= \phi_{out}(\phi_{X}(...\phi_{2}(\phi_{1}(\mathbf{P}^T \mathbf{v}^U_u, \mathbf{Q}^T \mathbf{v}^I_i))...))
$$

##### 3.1.1 Learning NCF
- Binary Cross Entropy
implicit feedback의 특성상 $y_{ui}=1$은 u와 i가 연관이 있다(반대는 0)
$$
\mathcal{L} = - \sum_{(u,i) \in \mathcal{Y} \cup \mathcal{Y}^-} y_{ui} \log \hat{y}_{ui} + (1 - y_{ui}) \log (1 - \hat{y}_{ui})
$$

#### 3.2 Generalized Matrix Factorization(GMF)
- MF의 일반화된 형태
$$
\phi_1 ( \mathbf{p}_u, \mathbf{q}_i) = \mathbf{p}_u \, \odot \, \mathbf{q}_i
\quad where \, \odot \, is \; elementwise \; product
$$
$$
\hat y_{ui} = a_{out} ( \mathbf{h}^T ( \mathbf{p}_u \, \odot \, \mathbf{q}_i ))
$$
> 여기서  $a_{out}$을 identity function $h$를 uniform function을 사용하면 MF와 정확히 동일
반대로 $a_{out}$를 non-linear function 사용하면 더 일반화

#### 3.3 Multi-Layer Perceptron(MLP)
- NCF는 user, item에 대한 2-pathway 모델
    - 이는 Multimodal deep learning work에서 흔히 사용
    - CF에서 vector concatenation은 interaction을 모델링하기 불충분
    - 이를 해결하기 위해 MLP를 도입
- MLP Design
    - Activation: ReLU 
    - Tower Pattern: From wide bottom to smaller nueron
    
#### 3.4 Fusion of GMF and MLP
![Figure3](https://velog.velcdn.com/images/smsm8898/post/b1cd7d54-a29c-495c-90d5-98d7d8d64b1c/image.png)
- Embedding Layer 공유
    - NTN(Neural Tensor Network)에서 사용
    - Performance에 한계가 존재(GMF와 MLP의 embedding size가 동일해야 함)
- Embedding Layer 분리
    - NeuMF
    - MF(Linear), DNN(Non-linear)
    - Adam을 이용했을 때 효과적 학습
    
## 4. EXPERIMENT
![Table](https://velog.velcdn.com/images/smsm8898/post/895115b2-cb0c-48aa-ae13-16a6427215d1/image.png)
- **Dataset**
    - _MovieLens_: 최소 20번의 interaction, implicit feedback으로 변환
    - [MovieLens-1m 다운로드](http://grouplens.org/datasets/movielens/1m/)
    - _Pinterest_: 최소 20번의 interaction이 존재하는 데이터만 사용
    - [Pinterest 다운로드](https://sites.google.com/site/xueatalphabeta/academic-projects?pli=1)
- **Evaluation**
    - leave-one-out(latest)
    - Random 100개의 item을 뽑아 평가
    - HR(Hit Ratio)
    - NDCG(Normalized Discounted Cumulative Gain)
    - top 10
    - ItemPop, ItemKNN, BPR, eALS와 비교
- **Hyperparameter**
    - Negative Sampling - 1(Positive) : 4(Negative)
    - Model Parameter - Random Initialize $$N(0,\, 0.01)$$
    - Adam
    - batch size - [128, 256, 512, 1024]
    - learning rate - [0.0001, 0.0005, 0.001, 0.005]
    - $\alpha = 0.5$
    - Predictive Factor(last hidden layer) - [8, 16, 32, 64]
        - 3 hidden layers
        - Predictive Factor: 8 = [32 $\rightarrow$ 16 $\rightarrow$ 8]
       
---
- **RQ1. NCF는 SOTA 모델의 성능을 능가할까? **
![Figure5](https://velog.velcdn.com/images/smsm8898/post/6d7ee8bf-c863-471c-bdca-6a5386d76561/image.png)
> NeuMF가 성능을 모두 이김

![Table2](https://velog.velcdn.com/images/smsm8898/post/55b4efb1-2350-4938-a7cb-c19960c02a6d/image.png)
> pre-training을 했을 때, 더 좋은 성능을 보임

---
- **RQ2. Negative Sampling을 이용한 logloss 학습**
![Figure 6, 7](https://velog.velcdn.com/images/smsm8898/post/0e753737-92aa-4d33-9c74-eed57986f0cb/image.png)

a. 더 많은 interactions가 있을 수록 학습이 잘 됨

b. NeuMF > MLP > GMF

c. 충분한 negative sampling이 필요( > 1)

---
- **RQ3. hidden units을 더 깊이 쌓을수록 좋을까?**
![Table3](https://velog.velcdn.com/images/smsm8898/post/50c8c557-2ab0-4bac-95ec-16846f091e23/image.png)
![Table4](https://velog.velcdn.com/images/smsm8898/post/376609ec-5f0c-4338-90f0-6f76620a7b1d/image.png)
