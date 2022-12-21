import os
import pathlib

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, Response
from waitress import serve

import utils

app = Flask(__name__)

iterative_imputer = joblib.load(os.path.join(pathlib.Path().resolve(), 'iterative_imputer.pkl'))


@app.route('/predict_json', methods=['POST'])
def predict_json():
    sparse_df = pd.concat([pd.DataFrame(columns=utils.train_strs_order),
                           pd.json_normalize(request.json)])
    sparse_df = sparse_df.replace(r'^\s*$', np.nan, regex=True)
    imputed_df = utils.get_imputed_df(iterative_imputer, sparse_df)
    imputed_df = imputed_df[utils.full_ftdna_strs_order]
    return Response(imputed_df.to_json(orient="records"), mimetype='application/json')


@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    sep = request.args.get('sep')
    sparse_df = pd.concat([pd.DataFrame(columns=utils.full_ftdna_strs_order),
                           pd.read_csv(request.files['file'], sep=sep)])
    sparse_df = utils.get_prepared_predict_df(sparse_df, False)
    imputed_df = utils.get_imputed_df(iterative_imputer, sparse_df)
    imputed_df = imputed_df[utils.full_ftdna_strs_order]
    return Response(imputed_df.to_csv(sep=sep, index=False), mimetype='text/csv')


@app.route('/predict_ftdna', methods=['POST'])
def predict_ftdna():
    sep = request.args.get('sep')
    sparse_df = pd.concat([pd.DataFrame(columns=utils.ftdna_strs_order),
                           pd.read_csv(request.files['file'], sep=sep)])
    sparse_df = sparse_df[utils.ftdna_strs_order]
    sparse_df = utils.get_prepared_predict_df(sparse_df, True)
    imputed_df = utils.get_imputed_df(iterative_imputer, sparse_df)
    for palindromes in [['DYS385a', 'DYS385b'], ['DYS459a', 'DYS459b'], ['DYS464a', 'DYS464b', 'DYS464c', 'DYS464d'],
                        ['CDYa', 'CDYb'], ['DYF395S1a', 'DYF395S1b'], ['DYS413a', 'DYS413b'], ['YCAIIa', 'YCAIIb']]:
        imputed_df[palindromes[0][:-1]] = imputed_df[palindromes].astype(str).apply("-".join, axis=1)
        imputed_df = imputed_df.drop(columns=palindromes)
    imputed_df = imputed_df[utils.ftdna_strs_order]
    return Response(imputed_df.to_csv(sep=sep, index=False), mimetype='text/csv')


@app.route('/predict_yfull', methods=['POST'])
def predict_yfull():
    sep = request.args.get('sep')
    sparse_df = pd.concat([pd.DataFrame(columns=utils.full_yfull_strs_order),
                           pd.read_csv(request.files['file'], sep=sep, index_col=0, header=None).T])
    sparse_df = sparse_df.drop(sparse_df.index.to_list()[1:], axis=0)
    sparse_df = sparse_df[sparse_df.columns.drop(list(sparse_df.filter(regex='_FT|_YS')))]
    sparse_df = sparse_df[utils.full_yfull_strs_order]
    for column in sparse_df.columns:
        sparse_df[column] = sparse_df[column].astype(str).str.split('.').str[0]
    sparse_df = sparse_df.rename(columns=utils.yfull_to_ftdna_dict)
    sparse_df = utils.get_prepared_predict_df(sparse_df, False)
    imputed_df = utils.get_imputed_df(iterative_imputer, sparse_df)
    ftdna_to_yfull_dict = dict((v, k) for k, v in utils.yfull_to_ftdna_dict.items())
    imputed_df = imputed_df.rename(columns=ftdna_to_yfull_dict)
    imputed_df = imputed_df[utils.full_yfull_strs_order]
    return Response(imputed_df.to_csv(sep=sep, index=False), mimetype='text/csv')


if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=8080)
