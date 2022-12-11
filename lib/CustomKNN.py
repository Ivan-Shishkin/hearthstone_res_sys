import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from scipy.spatial.distance import minkowski

from tqdm import tqdm
tqdm.pandas()


class CustomKNN:

    def __init__(self, *,
                 n_neighbors=1000,
                 leaf_size=30,
                 p=2,
                 n_jobs=None,
                 num_of_preds=2):

        self.n_neighbors = n_neighbors
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs

        self.model = None

        self.legendary_cards_ids = np.array([90145, 31110, 474, 33177, 22325, 90764, 378800,
                                             90190, 90792, 774144, 89877, 1108024, 90609, 33134,
                                             89370, 49628, 904874, 22342, 33168, 33156, 474989,
                                             22324, 27254, 89335, 22338, 475085, 49744, 14456,
                                             49756, 378803, 90624, 55520, 378802, 378801, 303,
                                             76960, 151349, 61828, 1024949, 55501, 12294, 241,
                                             76983, 12196, 14448, 285, 432, 90171, 127287,
                                             127264, 90220, 90621, 127293, 12217, 1004141, 474986,
                                             55468, 89437, 33173, 495, 463939, 475058, 35230,
                                             90234, 90614, 474979, 14451, 90240, 90602, 76929,
                                             90150, 90595, 76873, 7742, 73318, 475159, 90826,
                                             90209, 503, 1024989, 22343, 475129, 18, 90159,
                                             55473, 904864, 76895, 90566, 42036, 487632, 1024950,
                                             62898, 89405, 33138, 62922, 475033, 614784, 22393,
                                             220, 89334, 89374, 61822, 12187, 329913, 151357,
                                             62856, 90546, 474992, 89848, 90545, 33170, 22368,
                                             89415, 90720, 12182, 22313, 12290, 90237, 90176,
                                             90169, 22353, 1004133, 49683, 12232, 151377, 31117,
                                             1025325, 49693, 329918, 210732, 89359, 76920, 35207,
                                             228, 14438, 368689, 22260, 90554, 42022, 12293,
                                             738192, 562746, 12292, 151320, 90789, 329931, 55454,
                                             90562, 76893, 1024964, 1004192, 12272, 35201, 151344,
                                             90718, 49706, 90593, 614806, 22346, 474995, 329942,
                                             251, 89336, 90569, 33127, 474994, 39, 12295,
                                             89890, 14454, 151342, 12287, 89352, 76891, 55508,
                                             210656, 1024993, 89375, 7747, 62845, 151361, 22276,
                                             562747, 615055, 49684, 12282, 90250, 33149, 90557,
                                             90719, 73326, 388956, 33, 396, 210727, 22314,
                                             388974, 1024971, 55462, 179, 22349, 89406, 12225,
                                             12291, 151332, 49731, 203, 89863, 90741, 487636,
                                             22288, 210713, 90622, 42021, 12183, 61814, 12244,
                                             267, 61831, 89402, 1024963, 682, 7745, 89385,
                                             55451, 442035, 49632, 90285, 602, 329917, 90189,
                                             388955, 33131, 12268, 76947, 674, 388950, 1025002,
                                             127294, 12190, 442044, 49682, 7746, 89865, 463942,
                                             42031, 35188, 1004187, 388940, 22264, 89443, 49702,
                                             463925, 7744, 89889, 474996, 614779, 329909, 127292,
                                             12296, 49726, 89803, 500129, 90615, 474997, 463927,
                                             1004132, 62923, 984492, 487645, 1024976, 127271, 1024986,
                                             89812, 49737, 22323, 151364, 90213, 329941, 1024990,
                                             1024966, 89909, 90760, 210686, 62872, 49622, 329907,
                                             210742, 774178, 217, 62844, 61819, 487696, 90646,
                                             55457, 184662, 55522, 89411, 76986, 378844, 487661,
                                             55556, 90835, 90193, 487676, 329891, 210728, 210677,
                                             49659, 89813, 33139, 1024975, 474999, 151321, 90606,
                                             3, 22295, 89345, 22296, 22389, 89804, 388960,
                                             90199, 474998, 329943, 210683, 90560, 151368, 463924,
                                             42030, 329935, 61816, 49657, 774160, 49728, 329873,
                                             90680, 329939, 89805, 210667, 76907, 245, 90223,
                                             49633, 210730, 89424, 210717, 90174, 1024980, 475000,
                                             55464, 89860, 487662, 210811, 90568, 49729, 90585,
                                             339, 210779, 19, 456, 35248, 58723, 52582,
                                             487698, 210715, 90722, 388954, 463930, 90825, 89377,
                                             774138, 49624, 210805, 983954, 35190, 1024974, 89856,
                                             89840, 89818, 89918, 89888, 210804, 89892, 89802,
                                             90796, 52588, 90701, 90798, 89898, 210741, 55523,
                                             90724, 55538, 55447, 55551, 90705, 90697, 55481,
                                             55497, 89881, 90813, 90698, 55512, 55470, 76869,
                                             73321, 76949, 388953, 76975, 76976, 76930, 151352,
                                             76878, 73322, 76967, 329906])
        self.num_of_preds = num_of_preds

        self.df_train = None

        self.cards_features = pd.read_csv('top_card_rates/df_mage_cards_all.csv')
        self.cards_features.set_index('id', inplace=True)

        df_cards_vectors = self.cards_features[['is_battlecry_card',
                                                'is_divineshield_card',
                                                'is_rush_card',
                                                'is_deathrattle_card',
                                                'is_taunt_card',
                                                'is_combo_card',
                                                'is_inspire_card',
                                                'is_stealth_card',
                                                'is_charge_card',
                                                'is_overload_card',
                                                'is_lifesteal_card',
                                                'is_freeze_card',
                                                'is_discover_card',
                                                'is_windfury_card',
                                                'is_echo_card',
                                                'is_secret_card',
                                                'rarity',
                                                'type',
                                                'set',
                                                'race']].copy()

        # get dummies
        self.cards_vectors = pd.concat([df_cards_vectors.drop(['rarity', 'type', 'race', 'set'], axis=1),
                                        pd.get_dummies(df_cards_vectors[['rarity', 'type', 'race', 'set']],
                                                       drop_first=True)], axis=1)

        self.cards_vectors = self.cards_vectors.astype('int')

        # scaling

        scaler = StandardScaler()
        self.cards_vectors_norm = pd.DataFrame(scaler.fit_transform(self.cards_vectors),
                                               columns=self.cards_vectors.columns,
                                               index=self.cards_vectors.index)

        self.cards_vec_dict = self.cards_vectors_norm.T.to_dict('list')

    def fit(self, df_train):
        # add cards vectors to every deck
        df_train.loc[:, 'cards_vectors'] = df_train.loc[:, 'cards'].progress_apply(lambda x: list(map(lambda x: self.cards_vec_dict[x], x)))

        # add decks vectors
        df_train.loc[:, 'deck_vector'] = df_train.loc[:, 'cards_vectors'].progress_apply(lambda x: np.array(x).mean(axis=0))

        # fit model
        self.model = NearestNeighbors(n_neighbors=self.n_neighbors,
                                      algorithm='ball_tree',
                                      leaf_size=self.leaf_size,
                                      p=self.p,
                                      n_jobs=self.n_jobs).fit(df_train['deck_vector'].to_list())

        # save train dataset
        self.df_train = df_train

        return self

    def predict(self, deck, verbose=False):

        if len(deck) >= 30:
            raise AttributeError('Deck is already full')

        # get banned cards
        values, counts = np.unique(deck, return_counts=True)
        banned_cards = []
        for card_id, card_count in zip(values, counts):

            # max copies of plain cards
            if card_count == 2:
                banned_cards.append(card_id)

            # legendary cards must not be duplicated
            if card_id in self.legendary_cards_ids:
                banned_cards.append(card_id)

        # get cards vectors
        cards_vector = np.array([self.cards_vec_dict[x] for x in deck])

        # get deck vectors
        incomplete_deck_vector = cards_vector.mean(axis=0)
        test_vector = cards_vector.sum(axis=0)

        # get neighbors
        distances, neighbors = self.model.kneighbors(incomplete_deck_vector.reshape(1, -1), n_neighbors=2)

        if verbose:
            print(f'neighbors -- {neighbors}')
            print(f'distances -- {distances}')

        # get potential cards list

        first_neighbors_cards = self.df_train.iloc[neighbors[0][0]]['cards']
        first_neighbors_vector = self.df_train.iloc[neighbors[0][0]]['deck_vector']

        second_neighbors_cards = self.df_train.iloc[neighbors[0][1]]['cards']
        second_neighbors_vector = self.df_train.iloc[neighbors[0][1]]['deck_vector']

        if verbose:
            print(f'first_neighbors_cards -- {first_neighbors_cards}')
            print(f'second_neighbors_cards -- {second_neighbors_cards}')

        predicted_cards = np.unique(np.concatenate((first_neighbors_cards, second_neighbors_cards), axis=0))
        predicted_cards = np.sort(predicted_cards)

        df_preds = pd.DataFrame(index=predicted_cards, columns=['distance'])

        for card in predicted_cards:

            if card not in banned_cards:

                # print(card)

                complete_deck = deck + [card]
                complete_deck_vector = np.array([self.cards_vec_dict[x] for x in complete_deck]).mean(axis=0)
                # print(complete_deck)

                distance_1 = minkowski(complete_deck_vector.flatten(), first_neighbors_vector.flatten(),
                                       p=self.p)
                distance_2 = minkowski(complete_deck_vector.flatten(), second_neighbors_vector.flatten(),
                                       p=self.p)

                # print('\t 1: ',distance_1)
                # print('\t 2: ',distamce_2)

                df_preds.loc[card] = min(distance_1, distance_2)
            else:
                pass

        df_preds.sort_values(by='distance', inplace=True)
        df_preds.reset_index(inplace=True)
        df_preds.columns = ['card', 'distance']

        return df_preds[:self.num_of_preds]