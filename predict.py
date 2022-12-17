import joblib
import pandas as pd

import utils

combined_df = pd.read_csv('combined_snp_str_map.csv', dtype=str)
combined_df = utils.get_prepared_df(combined_df)

iterative_imputer = joblib.load('iterative_imputer.pkl')

sparse_combined_df = utils.get_sparse_df(combined_df)
utils.get_imputation_score(iterative_imputer, combined_df, sparse_combined_df)
