# Branch_Model
The Branch Model is an interpretable machine learning model. This means that it can transparently reveal what it has learned about the relationship between the input parameters and the output parameter. 

This repository includes an example where the Branch Model is used to predict an output y from three input parameters: A, x, and B, where the true underlying function is:

y = Asin(10x) + B

Script 01 first employs a regular dense neural network to calculate a benchmark error

Script 02 investigates which input parameters that should go into the different branches by evaluating the error of each possible setup. As expected, parsing A and x through Branch 1 results in the lowest error, meaning that it is the appropriate choice of parameters to be parsed through branch 1 since the first term in true underlying equation needs to be calculated before we can add B. See the image input_split_test_from_script_02.pdf in "images"

Script 03 plots the mapping of the two branches in the final Branch Model to transparently reveal the relationship between the output y and the three inputs A, x and B. As expected, the visualizations mirror the true underlying function. See the images branch_1_from_script_03.pdf and branch_2_from_script_03.pdf in "images".
