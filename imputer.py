import datetime

import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeRegressor

import utils

combined_df = pd.read_csv('combined_snp_str_map.csv', dtype=str)
combined_df = utils.get_prepared_df(combined_df)

kit_number_df, no_kit_number_df = utils.get_splitted_df(combined_df)

del combined_df

sparse_no_kit_number_df = utils.get_sparse_df(no_kit_number_df)

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
    #     predictor='auto',
    #     subsample=1),
    # estimator=GradientBoostingRegressor(  # Mean: 0.8961456876929831
    #     loss="squared_error",
    #     learning_rate=0.1,
    #     n_estimators=100,
    #     criterion="friedman_mse",
    #     min_samples_split=2,
    #     min_samples_leaf=1,
    #     max_depth=3,
    #     random_state=0,
    #     verbose=2,
    #     validation_fraction=0),
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
    estimator=DecisionTreeRegressor(  # Mean: 0.9606124449056386
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
iterative_imputer.fit(no_kit_number_df)
print('Fit time: ' + str((datetime.datetime.now() - start_time).total_seconds() / 60) + ' min')

utils.get_imputation_score(iterative_imputer, no_kit_number_df, sparse_no_kit_number_df)
