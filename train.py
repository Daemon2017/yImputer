import pandas as pd

import utils

combined_df = pd.read_csv('combined_snp_str_map.csv', dtype=str)
combined_df = utils.get_prepared_df(combined_df)

kit_number_df, no_kit_number_df = utils.get_splitted_df(combined_df)

iterative_imputer = utils.get_fitted_imputer(no_kit_number_df)

sparse_no_kit_number_df = utils.get_sparse_df(no_kit_number_df)
utils.get_imputation_score(iterative_imputer, no_kit_number_df, sparse_no_kit_number_df)
