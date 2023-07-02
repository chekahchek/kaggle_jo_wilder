import os
import zipfile
import numpy as np
import pandas as pd

#Check Kaggle API file
KAGGLE_CONFIG_DIR = os.path.join(os.path.expanduser('~'), '.kaggle')
assert 'kaggle.json' in os.listdir(KAGGLE_CONFIG_DIR), 'Ensure you have a kaggle.json file stored in the .kaggle folder of your home directory'

# Log in to Kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

# Download competition dataset
import kaggle
os.makedirs("data")
kaggle.api.competition_download_files("predict-student-performance-from-game-play", path="data")
with zipfile.ZipFile('data/predict-student-performance-from-game-play.zip', 'r') as zipref:
    zipref.extractall('data')

# Divide train.csv into different level groups and save them as parquet
df = pd.read_csv('data/train.csv')
for level_groups in ['0-4', '5-12', '13-22']:
    df.loc[df['level_group'] == level_groups, :].to_parquet(f"level_groups_{level_groups}.parquet", index=False)

