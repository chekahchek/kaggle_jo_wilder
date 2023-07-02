import os
import sys
import random
import numpy as np 
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier, early_stopping
from catboost import CatBoostClassifier
import optuna
import pickle
import gc
import math
import polars as pl
import argparse
from utils import seed_everything, feature_engineer, get_answer_time_2, prepare_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='lightgbm', help='Type of model to be trained')
    parser.add_argument('--seed', type=int, default=42, help='Seed value for reproducibility')
    parser.add_argument('--folds', type=int, default=5, help='Number of folds for cross validation')
    parser.add_argument('--level_groups', type=list, default=['0-4'], help='Denotes the level groups of the data. Accept either of these values: 0-4, 5-12, 13-22')
    parser.add_argument('--threshold', type=int, default=0.6, help='Threshold value to convert raw probabilities to label. Anything above threshold will be 1 while anything below threshold is 0')
    parser.add_argument('--n_trials', type=int, default=1, help='Number of trials for optuna optimization')
    args = parser.parse_args()

    seed_everything(args.seed)

    
    def run_single_fold(fold, train_idx, val_idx, question, threshold):
        def objective(trial, X_train=train.loc[train_idx, features], X_val=train.loc[val_idx, features],\
                    y_train=train.loc[train_idx, question], y_val=train.loc[val_idx, question]):
            
            if args.model == 'lightgbm':
                lightgbm_params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'num_leaves': trial.suggest_int('num_leaves', 8, 5000),
                    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 30, 400),
                    'n_estimators': trial.suggest_int('n_estimators', 30, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 10),
                    'reg_alpha' : trial.suggest_float('reg_alpha', 0.0, 10),
                    'reg_lambda' : trial.suggest_float('reg_lambda', 0.0, 10),
                }
                model = LGBMClassifier(**lightgbm_params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[early_stopping(50, verbose=False)])
                
            elif args.model == 'catboost':
                catboost_params = {
                    'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.1),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 3, 10),
                    'depth': trial.suggest_int('depth', 6, 10),
                    'rsm' : trial.suggest_float('rsm', 0.5, 1.0),
                    'verbose' : 0,
                }
                model = CatBoostClassifier(**catboost_params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=None, early_stopping_rounds=50)
                
            
            preds = model.predict_proba(X_val.values)
            preds = (preds[:, 1] >= threshold).astype(int)
            metric = f1_score(y_val, preds, average='macro')
            return metric

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=args.n_trials)
        print('Best trial:', study.best_trial.params)
        
        if args.model == 'lightgbm':
            best_model = LGBMClassifier(**study.best_trial.params)
            best_model.fit(train.loc[train_idx, features], train.loc[train_idx, question], \
                    eval_set=[(train.loc[val_idx, features], train.loc[val_idx, question])], \
                    callbacks=[early_stopping(50, verbose=False)])
        elif args.model == 'catboost':
            best_model = CatBoostClassifier(**study.best_trial.params)
            best_model.fit(train.loc[train_idx, features], train.loc[train_idx, question], \
                        eval_set=[(train.loc[val_idx, features], train.loc[val_idx, question])],\
                        early_stopping_rounds=50, verbose=None)
            
                            
        preds = best_model.predict_proba(train.loc[val_idx, features].values)
        preds = preds[:, 1]
        
        valid_sessions = train.loc[val_idx, 'session_id'].values
        _oof = pd.DataFrame({'session_id': valid_sessions, question: preds})
        
        preds = (preds >= threshold).astype(int)
        y_val = train.loc[val_idx, question]
        score = f1_score(y_val, preds, average='macro')
        
        if args.model == 'lightgbm':
            cols = best_model.feature_name_  + ['f1', 'fold', 'question']
            pickle.dump(best_model, open(f"{args.model}_fold{fold}_q{question}", "wb"))
        elif args.model == 'catboost':
            cols = best_model.feature_names_  + ['f1', 'fold', 'question']
            pickle.dump(best_model, open(f"{args.model}_fold{fold}_q{question}", "wb"))
            
            
        values = np.concatenate((best_model.feature_importances_ , [score, fold, question])).reshape(1, -1) 
        feat_impt = pd.DataFrame(values, columns=cols)
        print(f"Best {args.model} at fold {fold} question {question} has F1 score = {score:.4f}")
        
        del best_model
        gc.collect()
    
    return feat_impt, _oof


    for level_groups in args.level_groups:
        data = pd.read_parquet(f"/kaggle/input/jo-wilder-data/level_groups_{level_groups}.parquet")
        train = prepare_data(data, level_groups, train=True)

        if level_groups == "13-22":
            out = get_answer_time_2(data, stage2_path=f'/kaggle/input/jo-wilder-data/level_groups_{level_groups}.parquet')
            train = pd.merge(train, out, left_on='session_id', right_on='session_id', how='left')
        
        if level_groups == "0-4":
            targets = [i for i in range(1,4)]
        elif level_groups == "5-12":
            targets = [i for i in range(4, 14)]
        elif level_groups == "13-22":
            targets = [i for i in range(14, 19)]
        features = [cols for cols in train.columns if cols not in targets]
        features.remove('session_id')
            
        feat_impt_df= []
        oof_out = None
        gkf = GroupKFold(n_splits=args.folds)
        

        for ii, question in enumerate(targets):
            oof = []
            print(f"{'-' * 30} Question : {question} {'-' * 30}")

            for fold, (train_idx, val_idx) in enumerate(gkf.split(X=train, groups=train['session_id'])):
                print(f" {'-' * 30} Fold : {fold} {'-' * 30} ")
                feat_impt, _oof = run_single_fold(fold, train_idx, val_idx, question, args.threshold)
                feat_impt_df.append(feat_impt)
                oof.append(_oof)


            oof = pd.concat(oof, axis=0)
            if oof_out is None:
                oof_out = oof
            else:
                oof_out[question] = oof[question]

        feat_impt_out = pd.concat(feat_impt_df, axis=0)
        feat_impt_out.to_csv(f'feature_importance_stage{level_groups}.csv', index=False)
        oof_out.to_csv(f'oof_level_groups_{level_groups}.csv', index=False)
        
        del data
        del train
        gc.collect()