r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**

1. **False.** It depends on the dataset's distribution. For example, if we are learning
to differentiate between dogs and cats, and our dataset is ordered so that the first half
is dogs and the second half is cats, then choosing a training set that contains only dogs
would not allow the model to generalize well to cats.
Shuffling helps avoid this, but a split can still miss rare classes.
There may also be cases where no shuffle will help at all. For example, if we are trying
to classify 10 classes, but our data contains many examples of 8 classes and almost no
examples of the 2 remaining classes, we may not have enough examples to learn or evaluate
those two classes reliably.

2. **False!** It would cause leakage of the test data into the training process.
Validation is a part of the training process, used for finding the right hyperparameters.
This process is not allowed to see anything regarding the test data.

3. **True.** In each fold, the validation set was not used to train the model, so the
model's performance on it is an estimate of how well it will perform on unseen data,
i.e. of its generalization (test) loss. We average the validation performance over all
the folds, and based on it we decide which hyperparameters have the lowest estimated
generalization loss.

4. **True.** Train on noisy labels and on clean labels in otherwise identical runs.
Evaluate both on the same clean held-out data. A small drop in performance indicates
robustness to the level of label noise injected.

"""

part1_q2 = r"""
**Your answer:**

**No!** He used his test set as a validation set.

Although tuning $\lambda$ can help reduce overfitting, this process should be
done on the validation data, not on the test data. If we choose $\lambda$ based on the
performance on the test set, we have leakage of knowledge from the test data into the
training process, which is forbidden. After tuning, evaluate once on a fresh test set.

"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**

If we allow a negative margin $\Delta < 0$, the hinge loss will contain many more zeros:
the term $\max\left(0, \Delta + \boldsymbol{w_j}^\top \boldsymbol{x_i} - \boldsymbol{w_{y_i}}^\top \boldsymbol{x_i}\right)$
is zero even when a wrong class gets a **higher** score than the correct class, as long as it
is higher by at most $|\Delta|$. So the SVM loss will be smaller, and intuitively we allow
bad predictions with no punishment.

In addition, with positive regularization, the unique minimizer is $\boldsymbol{W} = 0$.
If all the weights are zero, all the scores are equal, so each hinge term is $\max(0, \Delta) = 0$,
and the regularization term is also zero. So the total loss is $0$, which is the smallest it can be.
Any nonzero weight matrix incurs positive regularization loss.
In the end, the model gives the same score to every class, so it doesn't learn anything and
can't classify.

"""

part2_q2 = r"""
**Your answer:**

The model tries to learn which digit is in the image.
In the weight images we can see the digits from 0 to 9 represented by the bright (green/yellow)
highlights, so the model learns a kind of template for each digit. Some digits are more
noticeable, like 0 or 3, while others are less clear, like 4 or 6.
As a pixel gets brighter, positive weights increase a class's score and negative weights
decrease it. The highest-scoring template wins.

The 5 classified as 6 has a loop at the bottom that can favor the 6 template. The 7
classified as 9 has an extra left stroke near the top, resembling part of a 9's loop.
The templates compare ink at fixed positions, so unusual strokes can favor another digit.

"""

part2_q3 = r"""
**Your answer:**

1. **Good.** Based on the training set loss graph, we can see that the loss goes down
sharply at first and then decreases more slowly, with a few spikes. So the learning
rate is good overall, though the spikes suggest it is on the high side.
    - If the learning rate were **too high**, we would see bigger oscillations after the big
    drop: the loss would jump up and down from epoch to epoch instead of settling, and it could
    even grow instead of going down.
    - If the learning rate were **too low**, we wouldn't see such a drastic drop in the loss so
    fast. The loss would go down slowly, almost like a straight line, and it would still be
    going down at the end of the training, since the model wouldn't reach the minimum in the
    same number of epochs.

2. **Slightly overfitted.** Based on the accuracy graph, we can say that the model is slightly
overfitted to the training set, because the training accuracy is higher than the validation
accuracy (93.6% vs. 90.6% in my run). The gap is modest, and the separate test
accuracy is 88.9%. The graph shows validation accuracy, while test accuracy is reported
separately.

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**

The ideal residual plot is one where the points are spread randomly around the zero line, with no pattern (no curve or trend), and the spread stays about the same for all values of $\hat{y}$. We can't expect all residuals to be exactly zero, because there can be noise in the data. Also, if all the train residuals were zero, it could indicate overfitting. So what we want is small residuals around zero, and the test residuals should look similar to the train residuals.

In the final plot after CV, most of the dots are concentrated close to the zero line, on both sides. The train and test dots overlap, though the test MSE is higher (14.01 vs. 7.82), and the plot still has some large residuals. Most dots are closer to zero than in the top-5 plot, suggesting a better fit.

In the top-5 plot, the dots are much more spread out and many of them are far from the zero line. We can also see a pattern: at low and high values of $\hat{y}$ many dots fall below zero. The plot uses $\hat{y}-y$ despite its axis label. The systematic pattern suggests that the linear model with only these 5 features underfits. After we added the non-linear features and chose the degree and $\lambda$ with CV, the pattern was weaker and the dots got closer to zero.

"""

part3_q2 = r"""
**Your answer:**

1. Yes, it is still a linear regression model. We can still write the prediction as $\hat{y} = \vec{w}^\top \vec{x} + b$, the only difference is that the features in $\vec{x}$ are now different (for example $x_1^2$, $x_1 x_2$, $\log x_1$). The model is linear in the parameters $\vec{w}$, and that's what makes it linear regression. It is not linear in the original features anymore, but the way we fit $\vec{w}$ (the closed-form solution) stays exactly the same.

2. Only functions in the span of the chosen features and the constant can be represented exactly. As we add non-linear features, the model will be able to capture broader and broader relations than just a simple linear model. To do this, we will need to choose the right features, so that their weighted sum will actually give us the non-linear function we are trying to capture. (Also, too many complicated features can lead to overfitting.)

3. In the new feature space (after adding the non-linear features), the decision boundary is still a hyperplane, because the classifier still computes $\vec{w}^\top \vec{x} + b$, just with the new features. Just like in linear regression, changing the features doesn't change the linear form of the model: the boundary $\vec{w}^\top \vec{x} + b = 0$ is still a linear equation in the new features, so in the new feature space it stays a hyperplane. In multiclass classification, $\vec{w}$ and $b$ are pairwise differences. In the original space the boundary can be curved. For example, the line $z_1+z_2=1$ becomes a circle when $z_i=x_i^2$.

"""

part3_q3 = r"""
**Your answer:**

1. $x$ and $y$ are independent and both $\sim \text{Uniform}(0,1)$, so their joint density is $f(x,y) = 1$ for $x, y \in [0,1]$:

$$
\mathbb{E}_{x,y}[|y-x|] = \int_0^1 \int_0^1 |y-x| \cdot 1 \, dy \, dx
$$

We open the absolute value:

$$
|y-x| =
\begin{cases}
x-y, & y \le x \\
y-x, & y > x
\end{cases}
$$

So we split the inner integral at $y = x$:

$$
\mathbb{E}_{x,y}[|y-x|] = \int_0^1 \int_0^x (x-y) \, dy \, dx + \int_0^1 \int_x^1 (y-x) \, dy \, dx = I_1 + I_2
$$

In $I_2$ we change the order of integration. The region $\{0 \le x \le 1,\ x \le y \le 1\}$ is the same as $\{0 \le y \le 1,\ 0 \le x \le y\}$, so:

$$
I_2 = \int_0^1 \int_x^1 (y-x) \, dy \, dx = \int_0^1 \int_0^y (y-x) \, dx \, dy
$$

$I_2$ written in a different way is:

$$
I_2 = \int_0^1 \int_0^x (x-y) \, dy \, dx = I_1
$$

So:

$$
\mathbb{E}_{x,y}[|y-x|] = I_1 + I_2 = 2 I_1 = 2 \int_0^1 \int_0^x (x-y) \, dy \, dx
$$

Inner integral:

$$
\int_0^x (x-y) \, dy = \left[ xy - \frac{y^2}{2} \right]_0^x = x^2 - \frac{x^2}{2} = \frac{x^2}{2}
$$

Outer integral:

$$
2 \int_0^1 \frac{x^2}{2} \, dx = \int_0^1 x^2 \, dx = \left[ \frac{x^3}{3} \right]_0^1 = \frac{1}{3}
$$

$$
\mathbb{E}_{x,y}[|y-x|] = \frac{1}{3}
$$

2. Here $\hat{x}$ is fixed (we assume $\hat{x} \in [0,1]$) and only $x \sim \text{Uniform}(0,1)$ is random, with density $f(x) = 1$ for $x \in [0,1]$:

$$
\mathbb{E}_{x}[|\hat{x}-x|] = \int_0^1 |\hat{x}-x| \cdot 1 \, dx
$$

We open the absolute value:

$$
|\hat{x}-x| =
\begin{cases}
\hat{x}-x, & x \le \hat{x} \\
x-\hat{x}, & x > \hat{x}
\end{cases}
$$

So we split the integral at $x = \hat{x}$:

$$
\mathbb{E}_{x}[|\hat{x}-x|] = \int_0^{\hat{x}} (\hat{x}-x) \, dx + \int_{\hat{x}}^1 (x-\hat{x}) \, dx
$$

First integral:

$$
\int_0^{\hat{x}} (\hat{x}-x) \, dx = \left[ \hat{x}x - \frac{x^2}{2} \right]_0^{\hat{x}} = \hat{x}^2 - \frac{\hat{x}^2}{2} = \frac{\hat{x}^2}{2}
$$

Second integral:

$$
\int_{\hat{x}}^1 (x-\hat{x}) \, dx = \left[ \frac{x^2}{2} - \hat{x}x \right]_{\hat{x}}^1 = \left( \frac{1}{2} - \hat{x} \right) - \left( \frac{\hat{x}^2}{2} - \hat{x}^2 \right) = \frac{1}{2} - \hat{x} + \frac{\hat{x}^2}{2}
$$

Adding them:

$$
\mathbb{E}_{x}[|\hat{x}-x|] = \frac{\hat{x}^2}{2} + \frac{1}{2} - \hat{x} + \frac{\hat{x}^2}{2} = \hat{x}^2 - \hat{x} + \frac{1}{2}
$$

For $\hat{x}<0$ the expectation is $\frac12-\hat{x}$, and for $\hat{x}>1$ it is $\hat{x}-\frac12$, because the sign of $\hat{x}-x$ is constant on $[0,1]$.

3. Plugging the polynomial into the loss we get:

$$
L_{\text{ED}} = 2\left(\hat{x}^2 - \hat{x} + \frac{1}{2}\right) = 2\hat{x}^2 - 2\hat{x} + 1
$$

The scalar (the constant term) doesn't depend on $\hat{x}$, so it doesn't change the gradient:

$$
\frac{\partial L_{\text{ED}}}{\partial \hat{x}} = 4\hat{x} - 2
$$

which is the same with or without the constant. In training we only use the gradient to update the generator, so the constant has no effect on the training. It also doesn't change where the minimum is: in both cases the minimum is at $\hat{x} = \frac{1}{2}$. The constant only shifts the value of the loss up or down, and we don't care about the value itself, only about which $\hat{x}$ minimizes it. That's why we can drop it.

"""

# ==============

# ==============
