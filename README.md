Binary Bayesian Classifiers
===========================
Python scripts for performing both independent and dependent Bayesian classification on binary datasts. Performance on randomly generated dependent datasets averages to ~90% correctness when used on datasets of 4 classes. Confusion matrices describing performance in more detail may be found in classifier directories.

Both dependent and independent classifier algorithms are trained and tested using a default 5-fold cross validation scheme. Number of folds can easily be modified in the k_fold_validator() parameters.

Usage
-----
* Datasets based on randomly generated dependence trees can be created by running `generate_data.py` in the `dependent_data_generator` directory. Although probabilities are entirely random upon each class generation, the structure of the trees across all generated datasets may be modified in the `dep_tree.py` script.

* The classifier can be invoked by running the `cross_validator.py` script within classifier directories. This script should be general enough to run classification on any number or size of binary datasets.

* The wine dataset can be converted to binary form by running `process_wine_info.py` from within the `wine_dataset_interpreter` directory. Output is ready to be fed into `cross_validator.py` for classification. A seperate directory is already set up for the wine classifier, containing all datasets and minor invocation differences already implemented.

Note
----
Running the classifier scripts requires both NetworkX and Matplotlib