# A3 Technical Report


Foundations of Machine Learning
Project 3: The Mini-ML Library and Autograd Engine
Name: Prahladha
Contents
1 Problem 1: The Mini-ML Library (3 Evaluation Points) 3
1.1 About the Module . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
1.2 Adherence to Scikit-Learn structure . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
1.3 Dataset Loading Module . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
1.4 Linear Regression Models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
1.4.1 Bayesian Regression . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
1.4.2 Ridge Regression . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
1.5 Linear Classification Models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
1.5.1 Linear Discriminant Analysis (LDA) . . . . . . . . . . . . . . . . . . . . . . . . . . 4
1.5.2 Least Squares for Classification . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
1.5.3 The Perceptron Algorithm . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
2 Problem 2: Logistic Regression 5
2.1 About the Module . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
2.2 Iterative Reweighted Least Squares (IRLS): Derivation and Implementation . . . . . . . . 5
2.2.1 Mathematical Formulation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
2.2.2 Implementation Notes . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
2.3 Limits of IRLS and Motivation for SGD . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
2.4 Stochastic Gradient Descent (SGD): Derivation and Implementation . . . . . . . . . . . . 6
2.4.1 Mathematical Equations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
2.4.2 Implementation Notes . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
2.5 Summary: IRLS versus SGD . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
3 Problem 3: Experiment - Spam Classification 7
3.1 About the Module . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
3.2 Dataset Loading and Preprocessing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
3.3 Cross-Validation and Hyperparameter Tuning . . . . . . . . . . . . . . . . . . . . . . . . . 7
3.4 Experimental Procedure . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
3.5 Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
3.6 Observations on Standardization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
4 Problem 4: The Autograd Engine & Visualization 9
4.1 About the Module . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
4.2 The Engine: autograd.py and the Value Class . . . . . . . . . . . . . . . . . . . . . . . . 9
4.3 Neural Network Modules: Layers, Activations, and Sequential . . . . . . . . . . . . . . . . 9
4.4 Toolkit: Loss Functions and Optimizer . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
4.5 Visualization: Traversing and Plotting the Computation Graph . . . . . . . . . . . . . . . 10
4.6 Interpretation of the Visualization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10


5 Problem 5: Finding the Best Generalizing Model 11
5.1 About the Module . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
5.2 Library Usage and Model Saving . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
5.3 Practical Changes and Observations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
5.4 Hyperparameter Tuning and Experimentation . . . . . . . . . . . . . . . . . . . . . . . . . 11
5.5 Training Loss Curves Plot . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
5.6 Performance Comparison Table . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
5.7 Discussion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
2


1 Problem 1: The Mini-ML Library (3 Evaluation Points)
1.1 About the Module
This problem focuses on structuring our machine learning algorithms into a professional, reusable Python
package named my ml lib. The task involves creating a specific directory layout and implementing key
modules for data loading and model implementation, ensuring they adhere to a scikit-learn-like interface.
This modular design is foundational for the subsequent problems in the project.
1.2 Adherence to Scikit-Learn structure
A core principle of the my ml lib package is its adherence to the well-established scikit-learn design.
This ensures consistency and ease of use.
 Models/Estimators: All model classes (e.g., Ridge, Perceptron) implement a fit(X, y) method
to train the model and a predict(X) method to make predictions on new data.
 Transformers: All data preprocessing classes (e.g., StandardScaler) implement fit(X, y=None),
transform(X), and a combined fit transform(X, y=None) method.
This consistent interface allows for the seamless integration of different components, such as using a
transformer to preprocess data before passing it to a model.
1.3 Dataset Loading Module
The my ml lib/datasets/ loaders.py module was created to handle the ingestion of datasets.
 load spambase(): Reads the spambase.data file, which is a comma-separated value (CSV) file,
into a NumPy array for features and a vector for labels.
 load fashion mnist(): Reads the training and testing CSV files for the Fashion-MNIST dataset.
 load synthetic(): Generates a synthetic dataset based on specified parameters, which is useful
for testing and validation.
1.4 Linear Regression Models
The regression models were implemented within the my ml lib/linear models/regression/ .
1.4.1 Bayesian Regression
The bayes.py module implements Bayesian Linear Regression. Instead of finding a single best-fit weight
vector w, this approach models the posterior distribution of the weights given the data. Assuming a
Gaussian likelihood and a zero-mean Gaussian prior on the weights, the posterior is also Gaussian.
The posterior distribution p(w|X, y) is given by:
p(w|X, y) = N (w|m , S )
N N
where the posterior mean m and covariance S are updated sequentially:
N N
S = (S1 + T )1
N 0
m = S (S1m + T y)
N N 0 0
Here, S and m are the prior covariance and mean,  is the precision of the noise, and  is the design
0 0
matrix. The implementation focuses on updating these parameters in the fit() method.
3


1.4.2 Ridge Regression
The ridge.py module implements Ridge Regression, a form of L2-regularized least squares. The ob-
jective is to find the weight vector w that minimizes the sum of squared errors plus a penalty on the
magnitude of the weights.
The cost function for Ridge Regression is:
N
(cid:88)
J(w) = (y  wT x )2 + w2
i i 2
i=1
This has a closed-form solution found by setting the gradient to zero. The implementation in the fit()
method calculates this optimal w directly using the pseudo-inverse solution:
w = (XT X + I)1XT y
where  is the regularization strength and I is the identity matrix.
1.5 Linear Classification Models
The classification models were implemented within the my ml lib/linear models/classification/ .
1.5.1 Linear Discriminant Analysis (LDA)
The lda.py module implements LDA, a generative classifier that models the class-conditional densities
p(x|C ) as Gaussian distributions with a shared covariance matrix. The decision boundary it learns is
k
linear.
LDA finds the projection vector w that maximizes the ratio of between-class variance to within-class
variance (Fishers criterion):
wT S w
J(w) = B
wT S w
W
where S is the between-class scatter matrix and S is the within-class scatter matrix. The optimal w
B W
is the eigenvector corresponding to the largest eigenvalue of S1S . The fit() method computes the
W B
class means, shared covariance, and this projection vector.
1.5.2 Least Squares for Classification
The least squares.py module implements a simple classifier using the same formulation as least
squares regression. It maps target labels (e.g., {0, 1}) to numerical values and finds the weight vector w
that minimizes the squared error.
The solution is identical to Ridge Regression with  = 0:
w = (XT X)1XT y = Xy
where X is the Moore-Penrose pseudo-inverse of X. While simple and guaranteed to find a solution,
this method is sensitive to outliers and does not produce probabilistic outputs, making it less ideal for
classification than other methods.
1.5.3 The Perceptron Algorithm
The perceptron.py module implements the Perceptron algorithm, one of the earliest algorithms for
supervised learning of a binary classifier. The model makes a prediction using a sign (step) function(sgn
function):
y = sign(wT x + b)
The fit() method implements the Perceptron learning rule. For each training sample (x , y ), if the
i i
prediction is incorrect (i.e., y (wT x + b)  0), the weights and bias are updated:
i i
w  w + y x
i i
b  b + y
i
where  is the learning rate. This process is repeated for a fixed number of epochs or until convergence.
The algorithm is guaranteed to find a separating hyperplane if one exists .
4


2 Problem 2: Logistic Regression
2.1 About the Module
This problem required designing and implementing logistic regression, one of the fundamental classifi-
cation algorithms in machine learning, within our custom package (my ml lib). The requirement was
to support L2 regularization and provide both an efficient closed-form solver (IRLS) and a scalable,
deep-learning-compatible solver (SGD), all within a scikit-learn-inspired (fit, predict, predict proba,
etc.).
2.2 Iterative Reweighted Least Squares (IRLS): Derivation and Implemen-
tation
2.2.1 Mathematical Formulation
Logistic regression models the log-odds of the probability for the positive class (y = 1) as a linear function
of the features:
(cid:18) (cid:19)
p
log = wT x + b
1  p
where p = P (y = 1|x). The probability can be equivalently expressed using the sigmoid function:
1
p = (wT x + b) =
1 + ewT xb
For a dataset {(x , y )}N , the negative log-likelihood (logistic loss) for all samples, regularized by a L2
i i i=1
penalty (), is:
N
(cid:88)
J(w, b) =  [y log p + (1  y ) log(1  p )] + w2
i i i i 2
i=1
The IRLS algorithm is an application of Newton-Raphson optimization to this convex objective.
Each iteration proceeds as:
1. Compute predicted probabilities for all samples: p = (wT x + b).
i i
2. Build weight matrix W , diagonal with entries w = p (1  p ).
i i i
3. Compute gradient and Hessian:
J(w) = XT (p  y) + w
H = XT W X + I
4. Update weights using Newton step:
w = w  H1J(w)
new old
This process is repeated until convergence.
2.2.2 Implementation Notes
The IRLS version in logistic.py implements this algorithm with fit() for training, predict proba()
for predicting probabilities, and predict() for generating discrete class predictions. The bias term b is
explicitly modeled and is not regularized.
2.3 Limits of IRLS and Motivation for SGD
IRLS is exceptionally efficient for small and medium datasets. However, it requires:
 Computing and inverting a d  d Hessian every iteration (where d is number of features).
 Storing this matrix in memory, which becomes infeasible for large feature counts (e.g., images,
text).
As the size of the problem scales, standard practice in modern machine learning is to use Stochastic
Gradient Descent (SGD), which requires only first-order gradients and can process data in mini-batches.
This change makes logistic regression practical even for very large datasets, is numerically stable, and
aligns with the code structure of deep learning frameworks.
5


2.4 Stochastic Gradient Descent (SGD): Derivation and Implementation
2.4.1 Mathematical Equations
SGD performs parameter updates using noisy, batch-wise estimates of the gradient. The regularized
logistic cost for a mini-batch of size m is:
m
1 (cid:88)
J (w, b) =  [y log p + (1  y ) log(1  p )] + w2
b m j j j j 2
j=1
The gradient for each batch is:
m
1 (cid:88)
J (w, b) = (p  y )x + w
b m j j j
j=1
Parameter updates follow:
m
1 (cid:88)
w  w  J (w) b  b   (p  y )
b m j j
j=1
where  is the learning rate.
2.4.2 Implementation Notes
The SGD-based implementation include lr, epochs, and batch size in init . The fit() method
shuffles the data, processes mini-batches, and applies SGD updates vectorized over each batch. All
regularization is applied to weights only, not bias. Probabilities and predictions are computed as before.
2.5 Summary: IRLS versus SGD
 Efficiency: IRLS converges quickly for small problems but is not scalable. SGD scales to arbi-
trarily large datasets and is deep-learning compatible.
 Numerical Stability: SGD avoids matrix inversion and is robust, while IRLS can be unstable
for ill-conditioned data.
 Practicality: For most real-world and competition-size datasets, SGD is the preferred optimizer.
 Interface: Both versions use object-oriented with scikit-learn pipeline design.
In summary, our implementation began with a mathematically rigorous IRLS optimizer but transi-
tioned to a practical, scalable SGD version to ensure our logistic regression model is suitable for real-world
applications and fully compatible with modern experimentation workflows.
6


3 Problem 3: Experiment - Spam Classification
3.1 About the Module
In this section, we perform a comprehensive experiment on the Spambase dataset, employing our custom
logistic regression implementation from my ml lib.
The objective is to investigate the impact of feature standardization and hyperparameter tun-
ing using K-Fold cross-validation. All data loading and preprocessing are performed using our own
loaders and transformers, preserving a modular and reproducible workflow.
3.2 Dataset Loading and Preprocessing
The Spambase dataset characterizes emails using real-valued features and a binary target denoting
whether the email is spam (y = 1) or non-spam (y = 0). It is loaded using the load spambase()
function in datasets/ loaders.py.
To prepare features, we implemented StandardScaler in preprocessing/data.py, which:
 Computes mean and variance over the training set,
 Transforms features to zero mean and unit variance.
This transformation follows the scikit-learn design with fit(), transform(), and fit transform()
methods. Standardization makes feature distributions more uniform, helping scale-dependent algorithms
(especially regularized linear models) perform better and converge faster.
3.3 Cross-Validation and Hyperparameter Tuning
Grid search hyperparameter tuning was conducted using the KFold utility implemented in my ml lib/model selection/kfold.py.
With k = 5, the training data is split into five folds. For each combination of (lr, epochs, ):
1. The logistic regression model is trained on k  1 folds,
2. Evaluated on the remaining fold,
3. Validation accuracy is averaged across all folds.
The best-performing hyperparameters are selected based on mean validation accuracy. This process
increases reliability of performance estimates and guards against overfitting.
3.4 Experimental Procedure
We conducted two independent experiments:
1. Raw Features: Logistic regression applied directly to the unscaled Spambase features. Hyper-
parameter grid search via 5-fold cross-validation was used to optimize learning rate, epochs, and
regularization strength ().
2. Standardized Features: Logistic regression applied after StandardScaler transformation. The
same grid search procedure was repeated as above.
For each case, the model with the best validation performance was retrained on the full training set
and evaluated on the held-out test set.
3.5 Results
Preprocessing Best Hyperparameters Train Error Test Error
Raw {lr = 0.001, epochs = 500,  = 0.01} 0.1432 0.1391
Standardized {lr = 0.01, epochs = 500,  = 0.01} 0.0799 0.0902
Table 1: SGD-based Logistic Regression on Spambase: Best settings and corresponding train/test errors.
Best :
7


 Raw Features:  = 0.01
 Standardized Features:  = 0.01
3.6 Observations on Standardization
Feature standardization led to a substantial improvement in both cross-validation accuracy and
generalization performance. While the best  for both cases remained 0.01, the optimal learning rate
increased for standardized data (0.01 vs 0.001).
 Train error dropped from 14.3% to 8.0%
 Test error dropped from 13.9% to 9.0%
Feature scaling is crucial in regularized linear models because regularization penalizes large coef-
ficients. Unscaled features with large magnitudes can dominate the optimization process, leading to
suboptimal weights and poor generalization. Standardization brings all features to a common scale,
ensuring fair contribution and a well-behaved regularization penalty.
This improves:
 Convergence rate,
 Numerical stability,
 Hyperparameter tuning reliability.
In conclusion, our custom logistic regression implementation significantly benefited from feature
standardization, reinforcing best practices in machine learning workflows.
8


4 Problem 4: The Autograd Engine & Visualization
4.1 About the Module
This problem is the core of the project. We built a dynamic computation graph and an automatic
differentiation engine from scratch, replicating the core principles behind PyTorchs autograd.
The objective was to implement a Value class capable of tracking numerical computations and sup-
porting backward propagation, along with essential neural network layers, loss functions, and an SGD
optimizer.
Finally, we visualized the computation graph for a simple neural network operation to validate the
correctness of our engine.
4.2 The Engine: autograd.py and the Value Class
The Value class is the foundation of our autograd engine. Each object of this class:
 Wraps a NumPy array (or scalar),
 Tracks its gradient,
 Stores references to its parent nodes ( prev),
 Records the operation ( op) that produced it.
Whenever a mathematical operation such as add, mul, sub, pow, truediv, matmul, exp, log, relu,
sum, or mean is performed, a new Value node is created and connected to its parent nodes, thereby
forming a dynamic computation graph (DAG).
Each node stores a custom backward function (closure) specific to its operation. The backward()
method, when called on the root node (usually the scalar loss), traverses the graph in reverse topological
order using a depth-first search (DFS). At each step, the stored backward function is invoked to propagate
gradients backward according to the chain rule.
Broadcasting and shape alignment are carefully managed to ensure correct gradient accumulation,
especially for matrix multiplications and reductions.
This design, inspired by the micrograd library, enables on-demand, expressive, and efficient differ-
entiation for arbitrary computation graphs.
4.3 Neural Network Modules: Layers, Activations, and Sequential
In my ml lib/nn/modules/, we implemented modular building blocks for neural networks. These include:
 Linear: Implements a fully connected (affine) layer. The forward computation ( call ) performs
XW + b. The weights and biases are stored as Value parameters, ensuring automatic gradient
tracking.
 ReLU and Sigmoid: Non-linear activation layers. ReLU performs max(0, x) and Sigmoid applies
(x). Both propagate gradients through their respective backward rules defined in the Value class.
 Sequential: Provides a simple way to stack modules into a feed-forward model. This structure
mimics PyTorchs nn.Sequential and enables flexible composition of layers.
This modular approach allows us to construct arbitrary network architectures for both forward and
backward passes with clean, object-oriented design.
4.4 Toolkit: Loss Functions and Optimizer
We implemented two key components to complete the training workflow:
 CrossEntropyLoss: Computes the softmax cross-entropy between predicted logits and target
labels. The result is a scalar loss, on which backward() is called to propagate gradients through
the network.
9


 SGD (Stochastic Gradient Descent): Updates parameters using the rule


for all learnable Value parameters. After each optimization step, gradients are reset to zero to
prepare for the next batch.
Together, these components allow us to train models end-to-end within our custom deep learning
mini-framework.
4.5 Visualization: Traversing and Plotting the Computation Graph
To verify the correctness of our autograd system, we visualized the computation graph of a small multi-
layer perceptron (MLP).
We implemented a recursive graph traversal function, get all nodes and edges(), which performs
a depth-first search starting from the root Value (usually the output or loss). This function returns all
unique nodes and their directed edges (parent  child) that represent the computation graph formed
during the forward pass.
The visualization was generated using the provided visualize.py script, which utilizes Graphviz to
render the node-edge structure.
For testing, we built and visualized a tiny network:
Sequential(Linear(2, 3), ReLU())
operating on a single 2D input. Running a forward and backward pass produced the graph shown
below, where each node displays its data and gradient, and each edge shows the computational and
gradient flow.
Figure 1: Computation graph visualization.
4.6 Interpretation of the Visualization
The visualization confirms that every parameter, activation, and operation is correctly represented in
the computation DAG.
Gradients propagate in reverse  from the scalar loss through intermediate operations to the learnable
parameters (weights and biases). All array shapes, broadcasting behaviors, and activation boundaries
are correctly maintained.
 Each operation node (e.g., matmul, add, relu) connects clearly to its operands.
 Gradient arrows confirm correct reverse-mode differentiation.
 Manual tracing validates that all forward computations are correctly captured.
Overall, the visualization demonstrates that our autograd engine performs accurate, end-to-end au-
tomatic differentiation  faithfully reproducing the behavior of modern deep learning frameworks.
10


5 Problem 5: Finding the Best Generalizing Model
5.1 About the Module
This capstone challenge required using the full my ml lib library  including our preprocessing modules,
classical models, and new autograd engine  to train and tune five different model architectures on the
Fashion-MNIST dataset.
Our objective was to optimize hyperparameters for each architecture using the validation set as a
proxy for generalization, then report test set accuracy as a final metric. The problem followed stan-
dard machine learning methodology: data splitting, tuning, retraining, and submitting only the top-
performing model for hidden evaluation.
5.2 Library Usage and Model Saving
All data loading, preprocessing, and modeling were performed using our implemented modules under
my ml lib.
Data was split via our train test val split() function in model selection/split.py, which takes
features X, labels y, and train/val/test proportions, returning the respective arrays.
Model saving and loading followed scikit-learn and PyTorch-like Structures: OvR Logistic models
used pickle, while autograd models used save state dict to export weights to .npz files.
After saving all five model checkpoints to the saved_models/ directory, we kept only the best-
performing model using a clean-up script.
5.3 Practical Changes and Observations
Logistic Regression: IRLS to SGD
Initially, our LogisticRegression implementation used IRLS for optimization. However, Fashion-
MNISTs high dimensionality (d = 784 features) made Hessian computation and inversion prohibitively
expensive in both time and memory.
IRLS required O(d3) time per iteration and the inversion of a 784  784 matrix (4 MB per matrix
per iteration). For larger polynomial or RBF feature sets, this quickly led to memory errors.
We therefore replaced IRLS with a fully vectorized SGD algorithm: each epoch performs simple
gradient updates in batches, scaling well to large datasets and remaining compatible with our autograd
and neural net systems.
Polynomial Feature Expansion: Memory Limitations
Our PolynomialFeatures transformer allowed crafting higher-order interactions between pixels.
However, even for degree = 2, the feature space exploded to  307,944 features (C(784 + 2, 2)).
For degree = 3, this expansion would reach  60 million features  over 400 MB per image and
more than 19 GB for the training set.
As a result, i encountered MemoryError: Unable to allocate X GB for an array. Thus, we
limited polynomial expansion to degree 2, balancing computational feasibility and generalization.
Gaussian Basis Features: Handling Memory Errors
Similarly, when generating Gaussian basis transformations, the naive vectorized approach attempted
to construct a 48,000  n  784 tensor, consuming several GBs of RAM.
centers
To avoid overflow, we implemented a fast iterative version that processed batches and computed
basis features one subset at a time. This approach scaled efficiently, though n was still capped to
centers
prevent excessive computation and memory usage.
5.4 Hyperparameter Tuning and Experimentation
For each model type, hyperparameter tuning was performed as follows:
 OvR Logistic Regression: Tuned L2 regularization , epochs, and learning rate using the
validation set to pick the best values.
 Softmax Regression (Raw): Trained Sequential(Linear) with a softmax output; tuned lr,
batch size, and epochs.
11


 Softmax + Polynomial Features: Preprocessed inputs using PolynomialFeatures (degree =
2); tuned lr, epochs, and . Grid search was restricted to avoid RAM crashes.
 Softmax + Gaussian Basis Features: Input transformed via GaussianBasisFeatures (itera-
tive). Tuned n , , learning rate, and epochs.
centers
 MLP: Built using Sequential(Linear, ReLU, Linear) from our autograd engine. Tuned hidden
units (64, 128), lr, batch size, and epochs. Validation performance guided the final best params.
5.5 Training Loss Curves Plot
Figure 2: Training loss curve comparison across best autograd-based model configurations.
5.6 Performance Comparison Table
Table 2: Best test accuracy (%) for all five model types on Fashion-MNIST.
Model Best Hyperparameters Test Accuracy (%)
OvR Logistic Regression  = 0.1, lr = 0.01, epochs = 40, bs = 256 91.69
Softmax (Raw) lr = 0.05, epochs = 40, bs = 256 92.36
Softmax + Polynomial degree = 2, lr = 0.01, epochs = 30, bs = 64 90.43
Softmax + RBF n = 40,  = 35.0, lr = 0.01, epochs = 30, bs = 64 84.51
centers
MLP hidden = 128, lr = 0.1, epochs = 40, bs = 256 97.30
5.7 Discussion
(a) OvR Logistic vs. Softmax (Raw)
Softmax Regression slightly outperformed OvR Logistic Regression (92.36% vs 91.69%). The improve-
ment arises because softmax jointly optimizes for all classes, enforcing mutual exclusivity (probabilities
sum to 1) and yielding better-calibrated predictions.
OvR trains independent binary experts, which can overlap in ambiguous regions, leading to less
consistent decision boundaries.
12


(b) Polynomial and Gaussian Basis Features
Polynomial features gave comparable results (90.43%) but suffered from computational constraints;
higher degrees caused memory errors.
Gaussian basis expansion performed worse (84.51%) and demanded custom iterative computation to
avoid memory overflow.
Both highlight the limits of hand-crafted feature engineering in high-dimensional image data  it
causes feature explosion, overfitting, and rarely surpasses learned representations.
(c) Linear Models vs. MLP
The MLP achieved the best performance (97.30%), outperforming all linear models by over 5%.
This aligns with the Universal Approximation Theorem, which states that networks with at
least one hidden layer can approximate any continuous function  allowing nonlinear and hierarchical
feature learning unavailable to linear classifiers.
(d) Tuning Process and Key Insights
Critical hyperparameters included learning rate, batch size, epochs, and hidden layer size (for MLP).
Regularization () mattered mainly for linear models.
Attempts to increase polynomial degree or Gaussian centers frequently crashed, demonstrating the
risk of high-dimensional feature spaces. Loss curves confirmed faster convergence and lower minima for
MLPs compared to all other methods.
Summary: Through systematic experimentation, we verified that neural networks with learned
features (MLP) generalize better than linear or manually engineered models. Careful validation-driven
tuning and robust handling of memory constraints were crucial to achieving these results.
13
