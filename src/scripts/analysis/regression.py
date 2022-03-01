import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pandas import read_csv, concat, DataFrame
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, RidgeCV, MultiTaskLasso
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from sklearn.tree import DecisionTreeRegressor

from pytorch_tabnet.tab_model import TabNetRegressor

from src.config import CLEANED_PATH, DataType, UNCLEANED_PATH, RAW_FILE, INFERENCE_FILE
from src.logger import log


def clip_score(val):
    if val > 4.0:
        return 4.0
    elif val < 1.0:
        return 1.0
    else:
        return val


def add_missing_entries(source_path, target_path):
    s_df = read_csv(source_path, sep=',', usecols=['pair_id'])
    t_df = read_csv(target_path, sep=',', usecols=['pair_id', 'Overall'])
    print(s_df['pair_id'].unique().shape)
    print(t_df['pair_id'].unique().shape)
    # print(t_df.shape)
    # n_df =
    # print(n_df.shape)
    final_rows = []
    for row in s_df.itertuples(index=False):
        p_id = row[0]
        t_row = t_df.loc[t_df['pair_id'] == p_id]
        score = t_row.iloc[0, 1] if t_row.shape[0] >= 1 else random.choice([2, 3])
        final_rows.append([p_id, score])

    df = DataFrame(final_rows, columns=['pair_id', 'Overall'])
    df['Overall'] = df['Overall'].apply(clip_score)
    df.to_csv(target_path, mode='w', index=False)


if __name__ == "__main__":
    train_path = CLEANED_PATH.format(data_type=DataType.train.name) + INFERENCE_FILE
    test_path = CLEANED_PATH.format(data_type=DataType.test.name) + INFERENCE_FILE
    log.info(f"Using {train_path} for training linear regression and {test_path} as test data.")

    # train_df = read_csv(train_path, sep=',', usecols=['sentences_mean', 'sentences_min', 'sentences_max',
    #                                                   'sentences_med', 'title', 'n1_title_n2_text',
    #                                                   'n2_title_n1_text', 'n1_title_n1_text',
    #                                                   'n2_title_n2_text', 'start_para', 'end_para',
    #                                                   'ner', 'tf_idf', 'wmd_dist', 'overall'])

    # train_df = read_csv(train_path, sep=',', usecols=['sentences_mean', 'sentences_min', 'sentences_max',
    #                                                   'sentences_med', 'title', 'n1_title_n2_text',
    #                                                   'n2_title_n1_text', 'n1_title_n1_text',
    #                                                   'n2_title_n2_text', 'start_para', 'end_para', 'overall'])

    train_df = read_csv(train_path, sep=',', usecols=['tf_idf', 'wmd_dist', 'overall'])

    # train_df = read_csv(train_path, sep=',', usecols=['sentences_mean', 'sentences_min', 'sentences_max',
    #                                                   'sentences_med', 'title', 'n1_title_n2_text',
    #                                                   'n2_title_n1_text', 'n1_title_n1_text',
    #                                                   'n2_title_n2_text', 'start_para', 'end_para',
    #                                                   'wmd_dist', 'overall'])
    # train_df = read_csv(train_path, sep=',')
    # train_df.drop(columns=['pair_id'], inplace=True)

    print(train_df.shape)
    train_df = train_df.drop_duplicates()
    print(train_df.shape)
    # test_df = read_csv(test_path, sep=',', usecols=['sentences_mean', 'sentences_min', 'sentences_max',
    #                                                 'sentences_med', 'title', 'n1_title_n2_text',
    #                                                 'n2_title_n1_text', 'n1_title_n1_text',
    #                                                 'n2_title_n2_text', 'start_para', 'end_para',
    #                                                 'ner', 'tf_idf', 'wmd_dist'])

    # test_df = read_csv(test_path, sep=',', usecols=['sentences_mean', 'sentences_min', 'sentences_max',
    #                                                 'sentences_med', 'title', 'n1_title_n2_text',
    #                                                 'n2_title_n1_text', 'n1_title_n1_text',
    #                                                 'n2_title_n2_text', 'start_para', 'end_para'])

    test_df = read_csv(test_path, sep=',', usecols=['tf_idf', 'wmd_dist'])
    #
    # test_df = read_csv(test_path, sep=',', usecols=['sentences_mean', 'sentences_min', 'sentences_max',
    #                                                 'sentences_med', 'title', 'n1_title_n2_text',
    #                                                 'n2_title_n1_text', 'n1_title_n1_text',
    #                                                 'n2_title_n2_text', 'start_para', 'end_para',
    #                                                 'wmd_dist'])

    test_df_full = read_csv(test_path, sep=',')
    # train_df = read_csv(train_path, sep=',', usecols=['title', 'sentences', 'overall'])
    # train_df = read_csv(train_path, sep=',', usecols=['title', 'sentences', 'start_para', 'overall'])

    y = train_df.pop('overall')

    # l_cols = ['entities', 'narrative', 'time', 'geography', 'overall']
    # y = train_df[l_cols]
    # train_df.drop(columns=l_cols, inplace=True)

    x = train_df

    # pca = PCA(n_components=4)
    # x = pca.fit_transform(train_df, y)
    # print(pca.explained_variance_ratio_)

    # x, y = df['computed_score'].to_numpy().reshape(-1, 1), df['original_score'].to_numpy()
    train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=0.05, random_state=5)
    # r = LinearRegression()
    # r = RidgeCV(alphas=[5e-3, 5e-2, 5e-1, 1], alpha_per_target=True, gcv_mode='eigen')
    r = KernelRidge(alpha=1.0, kernel='poly', degree=3)
    # r = MLPRegressor(hidden_layer_sizes=(10, 5), random_state=5, max_iter=500,
    #                  alpha=0.1, early_stopping=True, learning_rate='adaptive')
    # r = MultiTaskLasso(alpha=0.1)
    # r = TabNetRegressor()

    # estimators = [('lr', LinearRegression()), ('dtr', DecisionTreeRegressor(max_depth=3))]

    # r = StackingRegressor(estimators=estimators,
    #                       final_estimator=RandomForestRegressor(n_estimators=5,
    #                                                             random_state=42))
    # r = DecisionTreeRegressor(max_depth=5)
    r.fit(train_x, train_y)
    # print(r.score(train_x, train_y))
    # print(r.coef_)
    # print(r.best_score_)
    print(r.dual_coef_)
    # print("".join([str(x) for x in r.dual_coef_]))
    y_pred = r.predict(val_x)
    y_pred = np.vectorize(clip_score)(y_pred)
    # print(type(y_pred))
    print(pearsonr(y_pred, val_y))
    #
    # log.info("Mean squared error: %.2f" % mean_squared_error(test_y, y_pred))
    # # The coefficient of determination: 1 is perfect prediction
    # log.info("Coefficient of determination: %.2f" % r2_score(test_y, y_pred))
    #
    # plt.scatter(test_x, test_y, color="black")
    # plt.plot(test_x, y_pred, color="blue", linewidth=3)
    # plt.show()

    # test_df = pca.transform(test_df)
    y_pred = r.predict(test_df)
    test_df_full['Overall'] = y_pred
    out_path = 'results/ablations/kr_tfwmd_poly3_full.csv'
    test_df_full.to_csv(out_path, mode='w', columns=["pair_id", "Overall"], index=False)

    UNCLEANED_PATH = UNCLEANED_PATH.format(data_type=DataType.test.name)
    RAW_FILE = RAW_FILE.format(data_type=DataType.test.name)

    add_missing_entries(UNCLEANED_PATH + RAW_FILE, out_path)
