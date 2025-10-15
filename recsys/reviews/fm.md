> 본 글은 2010년 발표된 Factorization Machines를 읽고 요약 및 정리한 글입니다.

___
## 0. 핵심요약

- **MF**: user–item 관계를 학습
- **SVM**: 데이터 간 경계(Margin)을 학습하는 **분류/회귀 모델**
- **FM**: 모든 feature 간의 관계를 학습하는 **일반화 모델**
- FM: SVM과 Factorization model(ex. MF)의 장점을 결합한 모델
- General Predictor: 다양한 Task에 사용가능
- FM은 huge sparse data 이용 가능

| 구분 | **Matrix Factorization (MF)** | **Factorization Machine (FM)** | **Support Vector Machine (SVM)** |
|------|-------------------------------|--------------------------------|----------------------------------|
| **주요 목적** | user–item interaction (추천 시스템) | 범용 예측 모델 (추천, CTR, 회귀/분류 등) | 분류 또는 회귀 |
| **입력 형태** | user, item | 여러 feature (user, item, context 등) | 여러 feature (벡터 형태 입력) |
| **모델 수식** | $$\hat{y}_{ui} = \mathbf{p}_u^\top \mathbf{q}_i$$ | $$\hat{y} = w_0 + \sum_i w_i x_i + \sum_{i<j} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j$$ | $$\hat{y} = \mathbf{w}^\top \mathbf{x} + b$$ |
| **상호작용 방식** | Inner Product(user, item) | feature 간 **2차 상호작용(term interaction)** | feature 간 **선형 결합(linear combination)** |
| **Feature 처리** | user, item만 사용 | 범주형/연속형 모두 가능 | 범주형/연속형 모두 가능 |
| **비선형 확장** | 불가능 (기본은 선형) | 2차 상호작용까지만 | 가능 (커널 함수 사용 시) |
| **모델 복잡도** | 비교적 단순 | 약간 복잡 (feature 조합 고려) | 커널 종류에 따라 다양 |
| **학습 목표** | 평점/선호도 예측 | 회귀, 분류, CTR 등 다양한 예측 | 클래스 간 마진 최대화 |
| **출력 예시** | 유저가 아이템을 클릭할 확률 | 클릭 확률, 구매 확률, 전환율 | 클래스 라벨 (예: +1 / -1) |
| **대표 사용 예시** | 영화/상품 추천 | 광고 CTR, 추천 랭킹, 피처 상호작용 모델 | 이미지 분류, 스팸 필터, 이진 분류 |

---

## 1. Introduction
- New Predictor: **Factorization Machines(FM)**
    - SVM: very sparse data에서 효과적이지 않음
    - Tensor factorization model: 특수 목적, 보편적 상황에선 효과적이지 않음
    - Specialed factorization models: 특수 목적, 보편적 상황에선 효과적이지 않음
- High Sparsity
- General Predictor
- Linear complexity
## 2. Prediction Under Sparsity
![](https://velog.velcdn.com/images/smsm8898/post/db28f2c2-2344-4d71-8e95-313692b3bc65/image.png)
> 실제 transaction과 같은 상황에 만들어지는 피처 벡터
Factorization Machines에서 사용될 input 데이터 형태
- Common Prediction Task
    - Classification$$\quad T = {+, -}$$
    - Regression $$\quad T = \mathbb{R}$$
$$
y \; : \; \mathbb{R}^n \rightarrow T, \quad x \in \mathbb{R}^n \;
$$
- $$m(x)$$: feature vector x에서 0이 아닌 숫자
- $$\overline{m}_D$$: 모든 $$m(x)$$의 평균


## 3. Factorization Machines(FM)
#### A. Factorization Machine Model
1) Model Equation(degree 2)
$$
\hat{y}(\mathbf{x}) := w_0 + \sum^n_{i=1} w_i x_i 
+ \sum^n_{i=1} \sum^n_{j=i+1} < \mathbf{v_i, v_j}> x_i x_j
$$
$$
where \quad w_0 \in \mathbb{R}, \quad \mathbf{w} \in \mathbb{R}^n, \quad \mathbf{V}^{n \times k}
$$
- 첫번째 항목은 global bias(모델의 기본 예측값)
- 두번째 항목은 i번째 항목의 강도(strength)
- 세번째 항목은 input feature를 embedding하여 내적(interaction)

2) Expressiveness
- 얼마나 다양한 interaction 패턴을 모델링할 수 있는지
    - $$W$$: pair wise interaction을 표현하는 matrix
    - $$k$$: latent factor
    - $$k$$가 충분히 크다면 $$W$$를 근사하거나 똑같이 표현할 수 있다
    - 하지만 sparse 한 상황에서 $$W$$를 제대로 추정할 데이터가 부족하므로 너무 큰 $$k$$는 overfitting을 유발

3) Parameter Estimation Under Sparsity
- interaction parameter를 factorization
    - Sparse data에서 파라미터 학습이 잘 됨
    - 직접적인 데이터가 없어도 간접적인 예측이 가능

4) Computation
- Original$$\quad O(kn^2)$$
- Reform(linear)$$\quad O(kn)$$
- 추천시스템에서는 대부분 x가 0이기 때문에 FM의 계산 비용은 $$O(k\overline{m}_D)$$ 
$$
\sum^n_{i=1}\sum^n_{j=i+1} <\mathbf{v_i, v_j}> x_i x_j \\
= \frac{1}{2} \sum^n_{i=1} \sum^n_{j=1} <\mathbf{v_i,v_j}>x_i x_j - \frac{1}{2} \sum^n_{i=1} <\mathbf{v_i,v_j}>x_i x_i \\
=\frac{1}{2} \left( \sum^n_{i=1}\sum^n_{j=1}\sum^k_{f=1} v_{i,f}v_{j,k}x_ix_j - \sum^n_{i=1}\sum^k_{f=1} v_{i,f}v_{i,f}x_ix_i  \right) \\
=\frac{1}{2}\sum^k_{f=1} \left ( \left ( \sum^n_{i=1}v_{i,f}x_i \right ) \left ( \sum^n_{j=1}v_{j,f}x_j \right ) -\sum^n_{i=1}v^2_{i,f}x^2 \right ) \\
= \frac{1}{2}\sum^k_{f=1}\left(\left(\sum^n_{i=1}v_{i,f}x_i\right)^2 - \sum^n_{i=1}v^2_{i,f}x^2_i\right)
$$

#### B. Factoization Machines as Predictors
- **Regression**: minimal least square error
- **Binary classification**: hinge loss or logit loss
- **Ranking**: pairwise classification loss
대부분 L2 정규화를 통해 overfitting을 완화

#### C. Learning Factozation Machines
- SGD를 통해 model의 parameter를 update($$w_0, \mathbf{w}, \mathbf{V}$$)
- 각각의 gradient는 $$O(1)$$로 계산 가능
$$
\frac{\partial}{\partial \theta} \hat{y}(\mathbf{x}) = 
\begin{cases}
1, & \text{if } \theta = w_0, \\
x_i, & \text{if } \theta = w_i, \\
x_i \cdot \sum^n_{j=1} v_{j,f} x_j \;-\; v+{i,f} x^2_i, & \text{if } \theta = v_{i,f} 
\end{cases}
$$

#### D. $$d - way$$ Factorization Machines
- Original: $$O(k_dn^d)$$ but **Linear!**

$$
\hat{y}(x) := w_0 + \sum^n_{i=1}w_ix_i \\
+ \sum^d_{l=2}\sum^n_{i_1=1}...\sum^n_{i_l=i_{l-1}+1} \left( \prod^l_{j=1}x_{i_j}\right) \left( \sum^{k_l}_{f=1} \prod^l_{j=1} v^{(l)}_{i_j,f} \right)
$$

#### Summary
- full parameter를 사용하는 대신에 모든 factorized interactions을 이용
1) High Sparsity(unobserved interactions를 일반화 가능)
2) Linear Complexity로 학습 가능
3) SGD로 학습 가능


