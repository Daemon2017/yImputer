import joblib
import pandas as pd

import utils

combined_df = pd.read_csv('combined_snp_str_map.csv', dtype=str)
combined_df = utils.get_prepared_df(combined_df, True)

iterative_imputer = utils.get_fitted_imputer(combined_df)
joblib.dump(iterative_imputer, 'iterative_imputer.pkl', compress=True)

sparse_combined_df = utils.get_sparse_df(combined_df)
imputed_df = utils.get_imputed_df(iterative_imputer, sparse_combined_df)
utils.get_imputation_score(combined_df, imputed_df)
