import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeRegressor

utils_columns = ['KIT NUMBER']
non_STR_columns = ['SHORT HAND', 'LNG', 'LAT', 'NGS']

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

start_time = datetime.datetime.now()
combined_df = pd.read_csv('combined_snp_str_map.csv', dtype=str)
combined_df.columns = map(str.upper, combined_df.columns)
combined_df = combined_df.drop(columns=non_STR_columns)
combined_df = combined_df.dropna()

for palindrome_column in ['CDY', 'DYF395S1', 'DYS385', 'DYS413', 'DYS459', 'YCAII']:
    a_df = pd.DataFrame(combined_df[palindrome_column].str.split('-').str[0].rename(palindrome_column + 'a'))
    b_df = pd.DataFrame(combined_df[palindrome_column].str.split('-').str[-1].rename(palindrome_column + 'b'))
    combined_df = pd.concat([combined_df, a_df, b_df], axis=1)
    del combined_df[palindrome_column]

for palindrome_column in ['DYS464']:
    a_df = pd.DataFrame(combined_df[palindrome_column].str.split('-').str[0].rename(palindrome_column + 'a'))
    b_df = pd.DataFrame(combined_df[palindrome_column].str.split('-').str[1].rename(palindrome_column + 'b'))
    c_df = pd.DataFrame(combined_df[palindrome_column].str.split('-').str[-2].rename(palindrome_column + 'c'))
    d_df = pd.DataFrame(combined_df[palindrome_column].str.split('-').str[-1].rename(palindrome_column + 'd'))
    combined_df = pd.concat([combined_df, a_df, b_df, c_df, d_df], axis=1)
    del combined_df[palindrome_column]

# for palindrome_column in ['DYS19', 'DYS425']:
#     splited = combined_df[palindrome_column].str.split('-')

for column in combined_df:
    if column not in utils_columns:
        try:
            combined_df = combined_df \
                .drop(combined_df[combined_df[column].str.contains('-', na=False)].index)
        except KeyError as KE:
            print(KE)

kit_number_df = pd.DataFrame(data=combined_df['KIT NUMBER'])

no_kit_number_df = combined_df.drop(['KIT NUMBER'], axis=1)
no_kit_number_df = no_kit_number_df.astype(float)
no_kit_number_df = no_kit_number_df.astype(int, errors='ignore')
no_kit_number_df = no_kit_number_df[strs_order]

del combined_df

np.random.seed(0)
sparse_no_kit_number_df = no_kit_number_df.mask(np.random.random(no_kit_number_df.shape) < .25)

iterative_imputer = IterativeImputer(
    # estimator=KNeighborsRegressor(n_neighbors=5),
    # estimator=AdaBoostRegressor(
    #     n_estimators=50,
    #     learning_rate=1.0,
    #     random_state=0),
    # estimator=BayesianRidge( # Mean: 0.8692483205630953
    #     n_iter=300,
    #     verbose=True),
    # estimator=xgboost.XGBRegressor(  # Mean: 0.8883833875007838
    #     n_estimators=100,
    #     random_state=0,
    #     n_jobs=-1,
    #     verbosity=2,
    #     learning_rate=0.3,
    #     max_depth=6,
    #     tree_method='auto',
    #     predictor='auto'),
    # estimator=GradientBoostingRegressor(  # Mean: 0.8961456876929831
    #     loss="squared_error",
    #     learning_rate=0.1,
    #     n_estimators=100,
    #     criterion="friedman_mse",
    #     min_samples_split=2,
    #     min_samples_leaf=1,
    #     max_depth=3,
    #     random_state=0,
    #     verbose=2),
    # estimator=ExtraTreesRegressor(  # Mean: 0.9297287077650671
    #     n_estimators=10,
    #     criterion='squared_error',
    #     max_depth=None,
    #     min_samples_split=2,
    #     min_samples_leaf=1,
    #     max_features='auto',
    #     bootstrap=False,
    #     n_jobs=-1,
    #     random_state=0,
    #     verbose=2,
    #     max_samples=None),
    # estimator=RandomForestRegressor(  # Mean: 0.954639525146363
    #     n_estimators=10,
    #     criterion='squared_error',
    #     max_depth=None,
    #     min_samples_split=2,
    #     min_samples_leaf=1,
    #     max_features='auto',
    #     bootstrap=False,
    #     n_jobs=-1,
    #     random_state=0,
    #     verbose=2,
    #     max_samples=None),
    estimator=DecisionTreeRegressor(  # Mean: 0.9591956309621924
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
iterative_imputer.fit(no_kit_number_df)

imputed = iterative_imputer.transform(sparse_no_kit_number_df)
imputed_no_kit_number_df = pd.DataFrame(imputed, columns=sparse_no_kit_number_df.columns)
imputed_no_kit_number_df = imputed_no_kit_number_df.astype(int, errors='ignore')

mean_score = no_kit_number_df.eq(imputed_no_kit_number_df.values).mean().mean()
print('Mean: ' + str(mean_score))
print('Time: ' + str((datetime.datetime.now() - start_time).total_seconds() / 60) + ' min')

# no_kit_number_df.eq(imputed_no_kit_number_df.values).mean().sort_values(ascending=False).index.values.tolist()
