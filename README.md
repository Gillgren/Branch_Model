# Branch_Model
The Branch Model is an interpretable machine learning model. This means that it can transparently reveal what it has learned about the relationship between the input parameters and the output parameter. The example in this repository demonstrates how the methodology works.

# Scenario
Assume that we have a variable of interest y, and we want to investigate how it depends on three input parameters A, x and B. Here, the true underlying function is:

y = Asin(10x) + B

which we pretend we do not know.

# Step 1 

In the first script: 01_benchmark_nn.py, we build a regular dense neural network that predicts y from A, x and B. Since a regular neural network is a black box, we cannot determine how y depends on the three inputs yet. This step is just about obtaining a benchmark prediction error that we can compare with the Branch Model later.

# Step 2 

In the second script: 02_finding_branch_model_architecture.py, we begin playing around with Branch Models. For three input parameters, the Branch Model architecture looks like:

The idea here is that the network is split into branches with a maximum of two inputs each. This enables interpretability as the output of each branch can be plotted versus its two inputs, which is demonstrated in step 3. However, before that, we need to figure out which of the inputs that should go into the different branches. This is the main task of step 2, and is done by training Branch Models and evaluating the error of each possible setup. As can be seen in the image below, parsing A and x through Branch 1 results in the lowest error, meaning that it is the appropriate choice of parameters to be parsed through branch 1. This is expected since the first term in true underlying equation needs to be calculated before we can add B.

![input split test](images/input_split_test_from_script_02.pdf)

Script 03 plots the mapping of the two branches in the final Branch Model to transparently reveal the relationship between the output y and the three inputs A, x and B. As expected, the visualizations mirror the true underlying function. See the images branch_1_from_script_03.pdf and branch_2_from_script_03.pdf in "images".
