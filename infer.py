import os
import sys
import random
import numpy as np 
import pandas as pd
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import pickle
import gc
from tqdm import tqdm
import polars
import argparse
from utils import feature_engineer, get_answer_time_2, prepare_data
import jo_wilder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=list, default=['lightgbm', 'catboost'], help='Models to be used')
    parser.add_argument('--lightgbm_wt', type=int, default=0.56, help='Weightage of LightGBM for ensemble prediction')
    parser.add_argument('--catboost_wt', type=int, default=0.30, help='Weightage of CatBoost for ensemble prediction')
    parser.add_argument('--folds', type=int, default=5, help='Number of folds for cross validation')
    parser.add_argument('--threshold', type=int, default=0.53, help='Threshold value to convert raw probabilities to label. Anything above threshold will be 1 while anything below threshold is 0')
    args = parser.parse_args()
    
    # Placeholder values to retain features when looping through jo_wilder API
    features_to_retain_stage1 = ['word_that']
    new_features_name_stage1 =  ['word_that_dummy']
    
    # Load models
    models = {}
    ALL_MODELS = []
    for model_name in args.models:
        ALL_MODELS.extend(os.path.join('./models', i) for i in os.listdir('./models') if model_name in i)

    for m in ALL_MODELS:
        model_name = m.split('/')[-1]
        models[model_name] = pickle.load(open(m,'rb'))
    
    
    # Predictions
    jo_wilder.make_env.__called__ = False
    env = jo_wilder.make_env()
    iter_test = env.iter_test()
    retained_features = {}

    for (raw_test, sample_submission) in iter_test:
        if raw_test['level_group'].iloc[0] == "0-4":
            questions = [1, 2, 3]
            level_grp = "0-4"
        elif raw_test['level_group'].iloc[0] == "5-12":
            questions = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
            level_grp = "5-12"
        elif raw_test['level_group'].iloc[0] == "13-22":
            questions = [14, 15, 16, 17, 18]
            level_grp = "13-22"


        test = prepare_data(data, level_grp, train=False)
    
        sess_to_retain = {}
        sess = test['session_id'].iloc[0]
        if level_grp == "0-4":
            for feat in features_to_retain_stage1:
                sess_to_retain.update({feat: test[feat].iloc[0]})
            retained_features[sess] = sess_to_retain
        elif level_grp == "13-22":
            out = get_answer_time_2(raw_test, retained_features=retained_features, train=False)
            test = pd.concat([test, out], axis=1)

        test = test.iloc[:, 1:]
        all_probs = {}
        for question in questions:
            for model in args.models:
                probs = 0 
                for fold in range(args.fold):
                    clf = models[f"{model}_fold{fold}_q{question}"]
                    probs = probs + clf.predict_proba(test)[0, 1]
                probs = probs / args.fold
                all_probs[model] = probs


            avg_predictions = None
            for model, wt in zip(args.models, [args.lightgbm_wt, args.catboost_wt]):
                if avg_predictions is None:
                    avg_predictions = all_probs[model] * wt
                else:
                    avg_predictions = avg_predictions + all_probs[model] * wt

            pred = int(avg_predictions >= args.threshold)
            mask = sample_submission['session_id'].str.contains(f"q{question}")
            sample_submission.loc[mask, 'correct'] = pred
        env.predict(sample_submission)