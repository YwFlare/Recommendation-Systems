# Recommendation-Systems
This project is for our course Big data technology and application in the second semester of junior year at Nankai University, finished by Falcon Yuan.

In this project, I use  **collaborative filtering** model based on **baseline** and **biassvd** to predict user ratings for movies, and achieved a good result.

## Baseline CF

  Some users' scores are generally higher than others, and some users' scores are generally lower than others. For example, some users are naturally willing to give praise to others, soft hearted and easy to talk, while others are more demanding, and always give a score of no more than 80 points (full score of 100 points)

Some items are generally higher than others, and some items are generally lower than others. For example, as soon as some goods are produced, their status will be determined. Some are more popular, while others are disliked.

### Forecast scoring steps

1）Calculate average ratings for all films $\mu$

2）Calculate each user's score and average score $\mu$ 's offset value $b_{u}$

3）Calculate the score and average score of each film $\mu$ 's offset value $b_{i}$

4）Predict user ratings for movies with bias

### Loss function

The average score of all movies can be calculated directly, so the problem is to measure the score bias of each user and the score bias of each movie. For the linear regression problem, we can use the square difference with L2 regularization to construct the loss function：


$$
\text { Cost }=\sum_{u, i \in R}\left(r_{u i}-\mu-b_{u}-b_{i}\right)^{2}+\lambda *\left(\sum_{u} b_{u}{ }^{2}+\sum_{i} b_{i}{ }^{2}\right)
$$



## BiasSvd CF

### Basic principles

1）Split the user item scoring matrix into two small matrices, the user hidden factor matrix and the item hidden factor matrix

2）The user hidden factor matrix represents the user characteristics that will affect the user's rating of items

3）The item hidden factor matrix represents the item characteristics that will affect the item score

4）Take out the user vector from the user hidden factor matrix, and take out the dot product of the item vector from the item hidden factor matrix to get the user's score prediction of the item

5）Using the idea of optimizing the loss function to solve the two matrices, the gradient descent optimization method can be used

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

## Evaluation method

Root Mean Square Error:


$$
\operatorname{RMSE}(\mathrm{X}, \mathrm{h})=\sqrt{\frac{1}{\mathrm{~m}} \sum_{\mathrm{i}=1}^{\mathrm{m}}\left(\mathrm{h}\left(\mathrm{x}^{(\mathrm{i})}\right)-\mathrm{y}^{(\mathrm{i})}\right)^{2}}
$$


Mean Absolute Error:


$$
\operatorname{MAE}(\mathrm{X}, \mathrm{h})=\frac{1}{\mathrm{~m}} \sum_{\mathrm{i}=1}^{\mathrm{m}}\left|\mathrm{h}\left(\mathrm{x}^{(\mathrm{i})}\right)-\mathrm{y}^{(\mathrm{i})}\right|
$$



Our loss: RMSE 22.0979, MAE 16.8197

<img src="https://s2.loli.net/2022/06/11/pTxwgiVtcK6a5I4.png" alt="7" style="zoom:50%;" />

## Pay attention

We are new to the field of recommended system, maybe something above is wrong, you can contact me by email.
