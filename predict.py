import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, Response
from waitress import serve

import utils

app = Flask(__name__)

iterative_imputer = joblib.load('iterative_imputer.pkl')


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
    sparse_df = pd.concat([pd.DataFrame(columns=utils.ftdna_strs_order),
                           pd.read_csv(request.files['file'], sep=sep)])
    sparse_df = utils.get_prepared_df(sparse_df, False)
    imputed_df = utils.get_imputed_df(iterative_imputer, sparse_df)
    imputed_df = imputed_df[utils.full_ftdna_strs_order]
    return Response(imputed_df.to_csv(sep=sep, index=False), mimetype='text/csv')


if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=8080)
