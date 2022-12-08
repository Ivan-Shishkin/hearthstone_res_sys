import pandas as pd
from lib.CustomKNN import CustomKNN
from lib.CustomSVD import CustomSVD
import pickle

from tqdm import tqdm
tqdm.pandas()


if __name__ == '__main__':

    # get mage data
    df_train = pd.read_csv('../data/df_train_aug.csv')

    df_train['cards'] = df_train['cards'].progress_apply(lambda x: list(map(int, x[1:-1].split(', '))))
    df_train['classes'] = df_train['classes'].progress_apply(lambda x: x[1:-1].split(', '))
    df_train['costs'] = df_train['costs'].progress_apply(lambda x: list(map(int, x[1:-1].split(', '))))
    df_train['attacks'] = df_train['attacks'].progress_apply(lambda x: list(map(int, x[1:-1].split(', '))))
    df_train['healths'] = df_train['healths'].progress_apply(lambda x: list(map(int, x[1:-1].split(', '))))

    df_mage = df_train.loc[df_train['hero'] == 'mage'].copy()

    # KNN

    ngbrs = CustomKNN()
    ngbrs.fit(df_mage)

    pickle.dump(ngbrs, open('knn.sav', 'wb'))

    loaded_model = pickle.load(open('knn.sav', 'rb'))
    preds = loaded_model.predict([24, 24, 30, 67, 67, 160, 177, 177, 188, 241, 309, 495, 589, 589, 595, 595, 622, 622, 475127, 475128, 475133,
         475133, 1024934, 1024934, 1024956, 1024956, 1024957, 1024982, 1024982])

    print(preds)

    # SVD

    df_surprise_mage = pd.read_csv('../data/mage_top_50.csv')

    model = CustomSVD()
    model.fit(df_surprise_mage)

    pickle.dump(model, open('svd.sav', 'wb'))

    loaded_model = pickle.load(open('svd.sav', 'rb'))
    preds = loaded_model.predict(
        [24, 24, 30, 67, 67, 160, 177, 177, 188, 241, 309, 495, 589, 589, 595, 595, 622, 622, 475127, 475128, 475133,
         475133, 1024934, 1024934, 1024956, 1024956, 1024957, 1024982, 1024982])

    print(preds)

