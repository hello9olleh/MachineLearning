# Logistic Regression

Logistic Regression is a supervised learning algorithm.
Logistic Regression is of Multinomial(Multiple Outcomes) and 
Binary(2 Outcomes).

**Basic Idea :**

Logistic Regression is similar to Linear regression such that the Probability is scaled down to (0 to 1).A threshold value is set and the outcome is classified with reference to the threshold.

**Hypothesis :**

The function Z is similar to the Hypothesis of Linear Regression
which is,

Z = $\displaystyle\sum_{i = 0}^{N}(\theta_i)(X_i)$    (Where,  $X_0$ = 1)

$\theta$ — Weight/Parameter/Regression Coefficients
N — Number of Features/Inputs
[**Note :** H has N + 1 terms where $\theta_0$ is Y-Intercept]

**Logistic(Sigmoid) Function :**

Here the probability/Hypothesis (H) varies from $-\infin$ to $\infin$ so sigmoid function (S) is used in order to limit the Values between 0 to 1.

$S(x) = \frac{1}{1 + e^{-x}}$

H (Hypothesis) = $\frac{1}{1 + e^{-Z}}$

**Cost Function :**

The Cost function (J) here is,

$J = \begin{cases}
      -Log(H) & \text{for Y = 1}\\
      -Log(1 - H) & \text{for Y = 0}\\
    \end{cases}$

Which can be brought into single function as,

$J = (\frac{-1}{m})(Y(Log(H)) + (1-Y)(Log(1 - H)))$

**Gradient Descent :**

Gradient Descent is used as an Optimisation method to reduce the value of Cost function (J).

Basic Idea behind the Gradient Descent algorithm is to reach the Global minimum of the Cost Function.

Let Y(Log(H)) = P and Q = (1-Y)(Log(1-H))

Derivative of P,

$\frac{\partial P}{\partial\theta} = (\frac{Y}{H})(\frac{\partial H}{\partial\theta}) = (\frac{Y}{H})(\frac{\partial H}{\partial Z})(\frac{\partial Z}{\partial\theta})$

$\frac{\partial P}{\partial\theta} = (\frac{Y}{H})H(1-H)X$      (Since $\frac{\partial H}{\partial Z} = H(1-H)$ and $\frac{\partial Z}{\partial\theta} = X$)

$\frac{\partial P}{\partial\theta} = Y(1-H)X$

Derivative of Q,

$\frac{\partial P}{\partial\theta} = (\frac{1-Y}{1-H})(-\frac{\partial H}{\partial\theta}) = (\frac{1-Y}{1-H})(-\frac{\partial H}{\partial Z})(\frac{\partial Z}{\partial\theta})$

$\frac{\partial P}{\partial\theta} = -(\frac{1-Y}{1-H})H(1-H)X$

$\frac{\partial P}{\partial\theta} = -H(1-Y)X$

On adding the derivatives of P and Q,

$\frac{\partial J}{\partial\theta} = (\frac{-1}{m})(Y-H)X$

Here the Updation of $\theta$ is same as the Linear Regression.

$\theta_j = \theta_j - \alpha\frac{\partial}{\partial \theta_j}(J)$

**Question :**

**1) Why Mean Squared error can’t be used as Cost function in Logistic Regression?**
   Mean Square Error can’t be used for Logistic Regression as the Function is Non-Convex for Logistic Regression.Which has multiple Local minimum.
**2) How is Learning Rate $\alpha$ related with the accuracy of the model?**
   Lower the Learning Rate the higher the accuracy of the model (This relation holds true till specific value of $**\alpha**$) as having higher value of  **$\alpha$** might skip the Global Minimum. 
**3) Is the Cost Function (J) Convex or Non-Convex?**
   The Cost Function here is Convex such that it has single Global minimum.
**4) How do we use Logistic regression for Multinomial?**
   One of the technique is to split Multi-Class Classification into multiple Binary Classification.
**5) Is there any other algorithm that can be used instead of Gradient Descent to Minimise the Cost Function?**
  Alternatives for Gradient Descent are Evolutionary Algorithm (EA) and 
Particle Swarm Optimisation (PSO).