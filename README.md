# Branch_Model
The Branch Model is an interpretable machine learning model. This means that it can transparently reveal what it has learned about the relationship between the input parameters and the output parameter. The example in this repository demonstrates how the methodology works.

# Scenario
Assume that we have a variable of interest y, and we want to investigate how it depends on three input parameters A, x and B. Here, the true underlying function is:

y = Asin(10x) + B

which we pretend we do not know.

# Step 1 

In the first script: 01_benchmark_nn.py, we build a regular dense neural network that predicts y from A, x and B. This is just to obtain a benchmark error that we can compare with the Branch Model later.

# Step 2 

In the second script: 02_finding_branch_model_architecture.py, we begin playing around with Branch Models. For three input parameters, the architecture should look like:

The main task of step 2 is to investigate which input parameters that should go into the different branches by evaluating the error of each possible setup. As we can see in the image below, parsing A and x through Branch 1 results in the lowest error, meaning that it is the appropriate choice of parameters to be parsed through branch 1. This is expected since the first term in true underlying equation needs to be calculated before we can add B.

![input split test](images/input_split_test_from_script_02.pdf)

Script 03 plots the mapping of the two branches in the final Branch Model to transparently reveal the relationship between the output y and the three inputs A, x and B. As expected, the visualizations mirror the true underlying function. See the images branch_1_from_script_03.pdf and branch_2_from_script_03.pdf in "images".
