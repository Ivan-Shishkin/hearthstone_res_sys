import os.path

import pandas as pd
from lib.CustomKNN import CustomKNN
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm
import random
import datetime


class CustomGridSearchCV:

    def __init__(self, *, estimator=CustomKNN,
                 param_grid={'n_neighbors': [1000], 'leaf_size': [30]},
                 scoring='Precision', direction='minimize',
                 cv=5):

        self.model = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.results_dataset = None
        self.direction = direction

        if scoring != 'Precision':
            raise AttributeError('Implemented only Precision score')

        if direction not in ['minimize', 'maximize']:
            raise AttributeError('Invalid direction value; support values -- [minimize, maximize]')

    def fit(self, dataset, output_file_path=None):

        if output_file_path is None:

            if not os.path.exists('logs/'):
                os.mkdir('logs/')

            output_file_path = f'logs/custom_gs_{datetime.datetime.now()}'

        with open(output_file_path, 'w') as f:
            f.write('n_neighbors,leaf_size,score\n')

            # lists to store results
            n_arr = []
            l_arr = []
            s_arr = []

            # get params for iteration
            neighbors_arr = np.array(self.param_grid['n_neighbors'])
            leafs_arr = np.array(self.param_grid['leaf_size'])

            for (n, l) in np.array(np.meshgrid(neighbors_arr, leafs_arr)).T.reshape(-1, 2):

                # split data
                kf = KFold(n_splits=self.cv)

                splits_scores_list = []
                for train_index, test_index in kf.split(dataset):

                    train_dataset = dataset.iloc[train_index].copy()
                    test_dataset = dataset.iloc[test_index].copy()

                    model = CustomKNN(n_neighbors=n, leaf_size=l, p=2, num_of_preds=2)
                    model.fit(train_dataset)
                    print('Model_fit')

                    scores_list = []

                    # get predictions & metrics
                    for i, row in tqdm(test_dataset.iterrows(), total=test_dataset.shape[0]):
                        # hide 1 random card
                        deck = row['cards']
                        rand_idx = random.randrange(len(deck))
                        incomplete_deck = deck[:rand_idx] + deck[rand_idx + 1:]
                        target_card = deck[rand_idx]

                        # get predictions
                        predicted_cards_df = model.predict(incomplete_deck)
                        first = predicted_cards_df.loc[0, 'card']
                        second = predicted_cards_df.loc[1, 'card']

                        # get metric
                        score = int(target_card == first or target_card == second)
                        scores_list.append(score)

                    score_of_split = np.mean(np.array(scores_list), axis=0)
                    splits_scores_list.append(score_of_split)

                kf_mean_score = np.mean(np.array(splits_scores_list), axis=0)

                n_arr.append(n)
                l_arr.append(l)
                s_arr.append(kf_mean_score)

                if output_file_path is not None:
                    f.write(f'{n},{l},{kf_mean_score}\n')

            self.results_dataset = pd.DataFrame(data={'neighbors_num': n_arr,
                                                      'leaf_size': l_arr,
                                                      'score': s_arr})
            if self.direction == 'minimize':
                self.results_dataset.sort_values(by='score', ascending=True, inplace=True)
            else:
                self.results_dataset.sort_values(by='score', ascending=False, inplace=True)

            self.results_dataset.reset_index(drop=True, inplace=True)

        return self

    def get_best_score(self):
        print(f'Best {self.scoring} -- {self.results_dataset.loc[0, "score"]}')

    def get_best_params(self):
        print(f'Best params -- {self.results_dataset.loc[0, "score"]}')
