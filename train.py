import joblib
import pandas as pd

import utils

combined_df = pd.read_csv('combined_snp_str_map.csv', dtype=str)
combined_df = utils.get_prepared_df(combined_df, True)

iterative_imputer = utils.get_fitted_imputer(combined_df)
joblib.dump(iterative_imputer, 'iterative_imputer.pkl', compress=True)

for sparse_percent in [0.89, 0.775, 0.67, 0.59, 0.5, 0.4, 0.34, 0.25, 0.2, 0.15, 0.1, 0.05]:
    sparse_combined_df = utils.get_sparse_df(combined_df, sparse_percent)
    imputed_df = utils.get_imputed_df(iterative_imputer, sparse_combined_df)
    utils.get_imputation_score(combined_df, imputed_df)
    print("=====+++++=====")
