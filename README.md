# Predict Student Performance from Game Play
 
## Introduction
This repository contains the source code for [Predict Student Performance from Game Play](https://www.kaggle.com/competitions/predict-student-performance-from-game-play) kaggle competition. Objective of the competition is to predict how well a student will score in quizes in an EdTech game based on their interactions during the gameplay. 

This solution is ranked 143 out of 2103 (top 7%) in the private leaderboard and is placed in the Bronze medal category. 

## Set up
Ensure you have setup Kaggle API on your host, you may refer to this [documentation](https://github.com/Kaggle/kaggle-api) on how to set it up.

Run the following code to download the competition data and divide the training data into individual level_groups.

```
python prepare_data.py
```

## Train
To train the models, run the following command
```
python train.py [--model] [--seed] [--folds] [--level_groups] [--threshold] [--n_trials]

Arguments:
--model           Type of model to be trained. Valid options are 'lightgbm' and 'catboost'. Default is 'lightgbm'
--seed            Seed value for reproducibility. Default is 42
--folds           Number of folds for cross-validation. Default is 5
--level_groups    Level group of the training data. Takes in the input as a list and accept either one of these values '0-4', '5-12', '13-22'
--threshold       Threshold value used for binary classification. Default is 0.6
--n_trials        Number of trials used in optuna hyperparameter optimization. Default is 1 
```

## Inference
Note that this competition uses the Kaggle Time-Series API during inference to ensure data is delivered in groupings that do not allow access to future data. 

To perform prediction, run the following command

```
python infer.py [--models] [--lightgbm_wt] [--catboost_wt] [--folds] [--threshold]

Arguments:
--models          Model for inference. Takes in the input as a list and valid options are 'lightgbm' and 'catboost'. Default is ['lightgbm', 'catboost']     
--lightgbm_wt     Weightage assigned to ensemble predictions for LightGBM. Default is 0.56
--catboost_wt     Weightage assigned to ensemble predictions for CatBoost. Default is 0.30
--folds           Number of folds used during training. Default is 5
--threshold       Threshold value used for binary classification. Default is 0.53
```
