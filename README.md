# **Parkinson's Disease Regression Analysis**

Welcome to the Parkinson's Disease Regression Analysis repository. This project seeks to automate the process of measuring the Unified Parkinson's Disease Rating Scale (UPDRS) for Parkinson's disease patients using patient voice recordings. By applying linear regression on voice recordings, we hope to create a simple, efficient, and easily manageable method to adapt treatments based on specific patient conditions.

## **Project Description**

This project focuses on patients affected by Parkinson's disease who may have difficulties in controlling muscles, starting movements, and even speaking. Since the amount of treatment, particularly Levodopa, should be increased as the illness progresses and provided at the right time during the day, an automated way to measure total UPDRS becomes critical.

In this project, we use voice recordings from patients that are easily obtained via a smartphone. These vocal features are then used in a linear regression model to estimate total UPDRS. We use three different linear regression algorithms and apply them to a publicly available dataset.

## **Data Analysis**

The available dataset consists of 22 features, out of which the 'subject ID' and 'test time' are removed, and 'total UPDRS' is considered as the regressand. The remaining 19 features are used as regressors, which include many voice parameters and motor UPDRS.

The dataset is shuffled and split into a 70% training set, 15% validation set (for unbiased evaluation of the model fit on the training dataset), and 15% test set. Each regressor and the regressand are normalized using their mean and standard deviation.

The analysis also highlights the covariance matrix for the entire normalized dataset, showcasing the correlation between total and motor UPDRS, among shimmer parameters, among jitter parameters, and between total UPDRS and voice parameters.