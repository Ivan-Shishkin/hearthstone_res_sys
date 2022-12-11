import pandas as pd
from lib.CustomKNN import CustomKNN
from lib.CustomGridSearchCV import CustomGridSearchCV
import pickle

from tqdm import tqdm
tqdm.pandas()


if __name__ == '__main__':
    # get mage data
    df_train = pd.read_csv('../../data/df_train_aug.csv')
    df_mage = df_train.loc[df_train['hero'] == 'mage'].copy()

    # drop old data
    df_mage['update_date'] = pd.to_datetime(df_mage['update_date'], format='%Y-%m-%d')
    df_mage['year'] = df_mage['update_date'].dt.year
    df_mage = df_mage.loc[df_mage['year'].isin(['2018'])]
    # drop useless columns
    df_mage = df_mage[['deckid', 'cards']].copy()

    # transform str to list
    df_mage['cards'] = df_mage['cards'].progress_apply(lambda x: list(map(int, x[1:-1].split(', '))))

    gs = CustomGridSearchCV(estimator=CustomKNN,
                            param_grid={'n_neighbors': [250, 500, 1000], 'leaf_size': [10, 30, 50, 100]},
                            cv=10,
                            scoring='Precision',
                            direction='maximize')

    gs.fit(df_mage, 'logs.txt')

    print(gs.results_dataset)
