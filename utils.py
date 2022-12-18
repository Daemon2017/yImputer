import datetime

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeRegressor

strs_order = [
    'DYS472', 'DYS436', 'DYS575', 'DYS435', 'DYS590', 'DYS494', 'DYS632', 'DYS490', 'DYS593', 'DYS425', 'DYS641',
    'DYS454', 'DYS434', 'DYS450', 'DYS455', 'DYF395S1b', 'DYS726', 'DYS578', 'DYS426', 'YCAIIa', 'DYS531', 'DYS438',
    'DYS459a', 'DYS594', 'DYS640', 'DYF395S1a', 'DYS636', 'DYS568', 'DYS392', 'Y-GGAAT-1B07', 'DYS565', 'DYS459b',
    'DYS638', 'DYS716', 'DYS492', 'YCAIIb', 'DYS617', 'DYS587', 'DYS388', 'DYS462', 'DYS717', 'DYS589', 'DYS413b',
    'DYS445', 'DYS393', 'DYS572', 'DYS487', 'DYS437', 'DYS556', 'DYS463', 'DYS495', 'DYS511', 'DYS497', 'DYS561',
    'DYS464a', 'DYS540', 'DYS485', 'DYS464b', 'DYS525', 'DYS448', 'DYS389I', 'DYS537', 'DYS461', 'DYS19', 'DYS385a',
    'DYS643', 'DYS441', 'DYF406S1', 'DYS413a', 'DYS464d', 'DYS505', 'DYS452', 'DYS464c', 'DYS520', 'Y-GATA-H4',
    'DYS391', 'DYS522', 'DYS385b', 'DYS513', 'DYS533', 'DYS510', 'DYS390', 'DYS635', 'DYS552', 'DYS607', 'DYS389II',
    'DYS460', 'DYS442', 'DYS446', 'DYS447', 'DYS557', 'DYS715', 'DYS504', 'Y-GATA-A10', 'DYS444', 'DYS532', 'DYS549',
    'DYS456', 'DYS439', 'DYS481', 'DYS650', 'DYS458', 'DYS534', 'DYS570', 'DYS714', 'DYS576', 'DYS712', 'DYS449',
    'CDYa', 'CDYb', 'DYS710']


def get_prepared_df(df):
    df.columns = map(str.upper, df.columns)
    df = df.drop(columns=['SHORT HAND', 'LNG', 'LAT', 'NGS'], errors='ignore')
    df = df.dropna()
    df = df.replace(r'\.0$', '', regex=True)
    df = df.loc[~(df == '0').any(1)]
    df = df.loc[~(df == '0-0').any(1)]
    df = df.loc[~(df == '0-0-0-0').any(1)]

    for palindrome_column in ['CDY', 'DYF395S1', 'DYS385', 'DYS413', 'DYS459', 'YCAII']:
        str_splitted = df[palindrome_column].astype(str).str.split('-')
        a_df = pd.DataFrame(str_splitted.str[0].rename(palindrome_column + 'a'))
        b_df = pd.DataFrame(str_splitted.str[-1].rename(palindrome_column + 'b'))
        df = pd.concat([df, a_df, b_df], axis=1)
        del df[palindrome_column]

    for palindrome_column in ['DYS464']:
        str_splitted = df[palindrome_column].astype(str).str.split('-')
        a_df = pd.DataFrame(str_splitted.str[0].rename(palindrome_column + 'a'))
        b_df = pd.DataFrame(str_splitted.str[1].rename(palindrome_column + 'b'))
        c_df = pd.DataFrame(str_splitted.str[-2].rename(palindrome_column + 'c'))
        d_df = pd.DataFrame(str_splitted.str[-1].rename(palindrome_column + 'd'))
        df = pd.concat([df, a_df, b_df, c_df, d_df], axis=1)
        del df[palindrome_column]

    for column in df:
        if column not in ['KIT NUMBER']:
            try:
                df = df.drop(df[df[column].astype(str).str.contains('-', na=False)].index)
            except KeyError as KE:
                print(KE)

    df = df.drop(['KIT NUMBER'], axis=1, errors='ignore')
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df = df.astype(float, errors='ignore')
    df = df.astype(int, errors='ignore')
    df = df[strs_order]

    return df


def get_sparse_df(df):
    np.random.seed(0)
    sparse_df = df.mask(np.random.random(df.shape) < .25)
    return sparse_df


def get_fitted_imputer(df):
    iterative_imputer = IterativeImputer(
        estimator=DecisionTreeRegressor(
            criterion="squared_error",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=0),
        sample_posterior=False,
        max_iter=10,
        initial_strategy='mean',
        imputation_order='roman',
        verbose=2,
        random_state=0)
    start_time = datetime.datetime.now()
    iterative_imputer.fit(df)
    print('Fit time: ' + str((datetime.datetime.now() - start_time).total_seconds() / 60) + ' min')
    return iterative_imputer


def get_imputed_df(imputer, sparse_df):
    imputed = imputer.transform(sparse_df)
    imputed_df = pd.DataFrame(imputed, columns=sparse_df.columns)
    imputed_df = imputed_df.astype(int)
    return imputed_df


def get_imputation_score(df, imputed_df):
    mean_score = df.eq(imputed_df.values).mean().mean()
    print('Mean score: ' + str(mean_score))
