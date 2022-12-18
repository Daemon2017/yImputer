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
ftdna_strs_order = [
    'DYS393', 'DYS390', 'DYS19', 'DYS391', 'DYS385a', 'DYS385b', 'DYS426', 'DYS388', 'DYS439', 'DYS389I', 'DYS392',
    'DYS389II', 'DYS458', 'DYS459a', 'DYS459b', 'DYS455', 'DYS454', 'DYS447', 'DYS437', 'DYS448', 'DYS449', 'DYS464a',
    'DYS464b', 'DYS464c', 'DYS464d', 'DYS460', 'Y-GATA-H4', 'YCAIIa', 'YCAIIb', 'DYS456', 'DYS607', 'DYS576', 'DYS570',
    'CDYa', 'CDYb', 'DYS442', 'DYS438', 'DYS531', 'DYS578', 'DYF395S1a', 'DYF395S1b', 'DYS590', 'DYS537', 'DYS641',
    'DYS472', 'DYF406S1', 'DYS511', 'DYS425', 'DYS413a', 'DYS413b', 'DYS557', 'DYS594', 'DYS436', 'DYS490', 'DYS534',
    'DYS450', 'DYS444', 'DYS481', 'DYS520', 'DYS446', 'DYS617', 'DYS568', 'DYS487', 'DYS572', 'DYS640', 'DYS492',
    'DYS565', 'DYS710', 'DYS485', 'DYS632', 'DYS495', 'DYS540', 'DYS714', 'DYS716', 'DYS717', 'DYS505', 'DYS556',
    'DYS549', 'DYS589', 'DYS522', 'DYS494', 'DYS533', 'DYS636', 'DYS575', 'DYS638', 'DYS462', 'DYS452', 'DYS445',
    'Y-GATA-A10', 'DYS463', 'DYS441', 'Y-GGAAT-1B07', 'DYS525', 'DYS712', 'DYS593', 'DYS650', 'DYS532', 'DYS715',
    'DYS504', 'DYS513', 'DYS561', 'DYS552', 'DYS726', 'DYS635', 'DYS587', 'DYS643', 'DYS497', 'DYS510', 'DYS434',
    'DYS461', 'DYS435']


def get_prepared_df(df):
    df.columns = map(str.upper, df.columns)
    df = df.drop(columns=['KIT NUMBER', 'SHORT HAND', 'LNG', 'LAT', 'NGS'], errors='ignore')
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

    df = df[~df.stack().astype(str).str.contains('-', na=False).any(level=0)]
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
