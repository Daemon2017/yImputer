import pandas as pd
from sklearn.model_selection import train_test_split

import utils

combined_df = pd.read_csv('combined_snp_str_map.csv', dtype=str)
combined_df = utils.get_prepared_df(combined_df, True)

for test_size in [0.25, 0.33, 0.50, 0.66, 0.75]:
    print('Test part %: ' + str(test_size * 100))
    X_train, X_test = train_test_split(combined_df, test_size=test_size, random_state=0, shuffle=True)

    iterative_imputer = utils.get_fitted_imputer(X_train)

    for df in [X_train, X_test]:
        sparse_subset = utils.get_sparse_df(df, 0.25)
        imputed_df = utils.get_imputed_df(iterative_imputer, sparse_subset)
        utils.get_imputation_score(df, imputed_df)
    print("=====+++++=====")
