import pandas as pd
from sklearn.model_selection import train_test_split

import utils

combined_df = pd.read_csv('combined_snp_str_map.csv', dtype=str)
combined_df = utils.get_prepared_df(combined_df)

kit_number_df, no_kit_number_df = utils.get_splitted_df(combined_df)

X_train, X_test = train_test_split(no_kit_number_df, test_size=0.50, random_state=0, shuffle=True)

iterative_imputer = utils.get_fitted_imputer(X_train)

for subset in [X_train, X_test]:
    sparse_subset = utils.get_sparse_df(subset)
    utils.get_imputation_score(iterative_imputer, subset, sparse_subset)