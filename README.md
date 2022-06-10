# Recommendation-Systems
This project is for our course Big data technology and application in the second semester of junior year at Nankai University, finished by Wei-Feng Yuan.

In this project, I use a mode mixed with baseline and biassvd collaborative filtering.

## Baseline CF

  Some users' scores are generally higher than others, and some users' scores are generally lower than others. For example, some users are naturally willing to give praise to others, soft hearted and easy to talk, while others are more demanding, and always give a score of no more than 80 points (full score of 100 points)

Some items are generally higher than others, and some items are generally lower than others. For example, as soon as some goods are produced, their status will be determined. Some are more popular, while others are disliked.

### Forecast scoring steps

1）Calculate average ratings for all films $\mu$

2）Calculate each user's score and average score $\mu$ 's offset value $b_{u}$

3）Calculate the score and average score of each film $\mu$ 's offset value $b_{i}$

4）Predict user ratings for movies

### Loss function

The average score of all movies can be calculated directly, so the problem is to measure the score bias of each user and the score bias of each movie. For the linear regression problem, we can use the square difference to construct the loss function as follows:


$$
\begin{aligned}
\text { Cost } &=\sum_{u, i \in R}\left(r_{u i}-\hat{r}_{u i}\right)^{2} \\
&=\sum_{u, i \in R}\left(r_{u i}-\mu-b_{u}-b_{i}\right)^{2}
\end{aligned}
$$
Add L2 regularization:
$$
\text { Cost }=\sum_{u, i \in R}\left(r_{u i}-\mu-b_{u}-b_{i}\right)^{2}+\lambda *\left(\sum_{u} b_{u}{ }^{2}+\sum_{i} b_{i}{ }^{2}\right)
$$

### Least square optimization

For the solution of the minimum process, we use the alternating least square method to optimize the implementation. The idea of least square method: find the partial derivative of the loss function, and then make the partial derivative 0

Partial derivative of loss function:


$$
\frac{\partial}{\partial b_{u}} f\left(b_{u}, b_{i}\right)=-2 \sum_{u, i \in R}\left(r_{u i}-\mu-b_{u}-b_{i}\right)+2 \lambda * b_{u}
$$


If the partial derivative is 0, then:


$$
\begin{aligned}
&\sum_{u, i \in R}\left(r_{u i}-\mu-b_{u}-b_{i}\right)=\lambda * b_{u} \\
&\sum_{u, i \in R}\left(r_{u i}-\mu-b_{i}\right)=\sum_{u, i \in R} b_{u}+\lambda * b_{u}
\end{aligned}
$$


Simplify the formula:


$$
b_{u}:=\frac{\sum_{u, i \in R}\left(r_{u i}-\mu-b_{i}\right)}{\lambda_{1}+|R(u)|}
$$

$$
b_{i}:=\frac{\sum_{u, i \in R}\left(r_{u i}-\mu-b_{u}\right)}{\lambda_{2}+|R(i)|}
$$

### Evaluation method

Root Mean Square Error:


$$
\operatorname{RMSE}(\mathrm{X}, \mathrm{h})=\sqrt{\frac{1}{\mathrm{~m}} \sum_{\mathrm{i}=1}^{\mathrm{m}}\left(\mathrm{h}\left(\mathrm{x}^{(\mathrm{i})}\right)-\mathrm{y}^{(\mathrm{i})}\right)^{2}}
$$


Mean Absolute Error:


$$
\operatorname{MAE}(\mathrm{X}, \mathrm{h})=\frac{1}{\mathrm{~m}} \sum_{\mathrm{i}=1}^{\mathrm{m}}\left|\mathrm{h}\left(\mathrm{x}^{(\mathrm{i})}\right)-\mathrm{y}^{(\mathrm{i})}\right|
$$

## BiasSvd CF

### Basic principles

Split the user item scoring matrix into two small matrices, the user hidden factor matrix and the item hidden factor matrix

The user hidden factor matrix represents the user characteristics that will affect the user's rating of items

The item hidden factor matrix represents the item characteristics that will affect the item score

Take out the user vector from the user hidden factor matrix, and take out the dot product of the item vector from the item hidden factor matrix to get the user's score prediction of the item

Using the idea of optimizing the loss function to solve the two matrices, the gradient descent optimization method can be used
$$
\begin{aligned}
\hat{r}_{u i} &=\mu+b_{u}+b_{i}+\overrightarrow{p_{u k}} \cdot \overrightarrow{q_{k i}} \\
&=\mu+b_{u}+b_{i}+\sum_{k=1}^{k} p_{u k} q_{i k}
\end{aligned}
$$

### Loss function

$$
\begin{array}{l}
\text { Cost }=\sum_{u, i \in R}\left(r_{u i}-\mu-b_{u}-b_{i}-\sum_{k=1}{ }^{k} p_{u k} q_{i k}\right)^{2} \\
+\lambda\left(\sum_{U} b_{u}{ }^{2}+\sum_{I} b_{i}{ }^{2}+\sum_{U} p_{u k}{ }^{2}+\sum_{I} q_{i k}{ }^{2}\right)
\end{array}
$$

### Random gradient descent method

Gradient descent update parameters  $p_{u k}$  :

$$
p_{u k}:=p_{u k}+\alpha\left[\sum_{u, i \in R}\left(r_{u i}-\mu-b_{u}-b_{i}-\sum_{k=1}^{k} p_{u k} q_{i k}\right) q_{i k}-\lambda p_{u k}\right]
$$

For the same reason：

$$
\begin{array}{l}
q_{i k}:=q_{i k}+\alpha\left[\sum_{u, i \in R}\left(r_{u i}-\mu-b_{u}-b_{i}-\sum_{k=1}^{k} p_{u k} q_{i k}\right) p_{u k}-\lambda q_{i k}\right] \\
b_{u}:=b_{u}+\alpha\left[\sum_{u, i \in R}\left(r_{u i}-\mu-b_{u}-b_{i}-\sum_{k=1}^{k} p_{u k} q_{i k}\right)-\lambda b_{u}\right] \\
b_{i}:=b_{i}+\alpha\left[\sum_{u, i \in R}\left(r_{u i}-\mu-b_{u}-b_{i}-\sum_{k=1}^{k} p_{u k} q_{i k}\right)-\lambda b_{i}\right]
\end{array}
$$

## RMSE and MAE curves

![7](https://raw.githubusercontent.com/Falcon-Yuan/images/master/202206101849480.svg?token=APJNLFV5Q3E25PSOCKQCB2TCUMRAM)

## Pay attention

We are new to the field of recommended system, maybe something above is wrong, you can contact me by email.
