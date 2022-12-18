import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, Response

import utils

app = Flask(__name__)

iterative_imputer = joblib.load('iterative_imputer.pkl')


@app.route('/predict_json', methods=['POST'])
def predict_json():
    str_dict = request.json
    sparse_df = pd.json_normalize(str_dict)
    sparse_str_order = sparse_df.columns
    sparse_df = sparse_df[utils.strs_order]
    sparse_df = sparse_df.replace(r'^\s*$', np.nan, regex=True)
    imputed_df = utils.get_imputed_df(iterative_imputer, sparse_df)
    imputed_df = imputed_df[sparse_str_order]
    return Response(imputed_df.to_json(orient="records"), mimetype='application/json')


@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    sep = request.args.get('sep')
    sparse_df = pd.read_csv(request.files['file'], sep=sep)
    sparse_df = utils.get_prepared_df(sparse_df)
    imputed_df = utils.get_imputed_df(iterative_imputer, sparse_df)
    return Response(imputed_df.to_csv(sep=sep, index=False), mimetype='text/csv')


@app.route('/predict_text', methods=['POST'])
def predict_text():
    strs = request.data.decode("utf-8").split()
    sparse_df = pd.DataFrame([strs], columns=utils.strs_in)
    sparse_df = utils.get_prepared_df(sparse_df)
    imputed_df = utils.get_imputed_df(iterative_imputer, sparse_df)
    imputed_df = imputed_df[utils.strs_out]
    return Response(np.array2string(imputed_df.values[0], separator='	').replace('\n', ''), mimetype='text/plain')


if __name__ == '__main__':
    app.run()
