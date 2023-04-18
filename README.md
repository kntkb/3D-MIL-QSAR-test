# QSAR modeling based on conformation ensembles using a multi-instance learning approach
**Cloned and modified from https://github.com/cimm-kzn/3D-MIL-QSAR.git**  

This repository containes the Python source code from the paper ["QSAR modeling based on conformation ensembles using a
multi-instance learning approach"](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00692) with additional features.

- Support other ML methods to compare with Multi-Instance Learning approach.
- Add configuration file to run experiments easier.
- Add statistical report (mean signed error, root mean squared error, mean unsigned error, kendall tau, pearson r)


## Overview
Our research focuses on the application of Multi-Instance Learning (MIL) in QSAR modeling.
In Multi-Instance Learning, each training object is represented by several feature
vectors (bag) and a label. In our implementation, an example (i.e., a molecule) is presented
by a bag of instances (i.e., a set of conformations), and a label (a bioactivity value) is available
only for a bag (a molecule), but not for individual instances (conformations).
Both traditional MI algorithms and MI deep neural networks were used for model building.

