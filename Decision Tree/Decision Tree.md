# Decision Tree

Decision Tree is Supervised Machine Learning Algorithm.
Decision Tree is of Classification Tree and Regression Tree.
The idea behind Classification Tree is analogous to Logistic Regression (Like Classifying if a Wine Sample is good or bad) and that of Regression Tree is analogous to Linear Regression (Like Predicting the Price of House)

**Decision Tree :**
Each Node represents feature/attribute, each branch represents a decision/rule and each leaf(Which can’t be divided further) represents outcome.

There are some Algorithms to build Decision Tree,
Here CART(Classification and Regression Tree) Algorithm is being discussed,

In CART Gini Index is used as Metric,
Gini Index is the Cost Function here,

**Cost Function :**

$G = 1 -\displaystyle\sum_{i = 0}^{K}P^{2}_i$

where K — Different Values of Target Variable
Gini Index(G) becomes maximum when all the Target values distributed equally,
which is (1-1/K).
Gini Index(G) becomes minimum when it’s distributed to single variable,
which is 0.

Gini Index is calculated for all categorical values for every features and best Gini gain is calculated and Repeated until the desired Tree is obtained.

Entropy (Information Gain) = Entropy Before Splitting - Entropy after Splitting

$E = \displaystyle\sum_{i = 1}^{N}P_i\log(P_i)$

**Question :
1) Mention One of the Algorithm which can be used to build Decision Tree other than CART?**

> ID3 (Iterative Dichotomiser 3) which used Information Gain and Entropy Function.
> 

**2) What is the Disadvantage of CART Algorithm?**

> A small change in the dataset makes the Tree unstable.
> 

**3) Can Decision Tree be used for Multinomial Classification?**

> Yes,Decision Tree can be used for Multinomial Classification.
> 

**4) For what kind of Data, Decision Tree is the most suitable?**

> Non-Linear Data is most suitable for Decision Tree algorithm.
> 

**5) What is the advantage of using Decision Tree?**

> All possible outcomes of a Decision are considered and each path is traced till conclusion.
>