import os
import sys

import joblib
from flask import Flask, request, Response
from flask_cors import CORS
from waitress import serve

import utils

TEXT_PLAIN = 'text/plain'

app = Flask(__name__)
cors = CORS(app)

iterative_imputer = joblib.load(os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), 'iterative_imputer.pkl'))


@app.route('/predict_yfull', methods=['POST'])
def predict_yfull():
    sparse_df = utils.get_prepared_predict_df(request)
    imputed_df = utils.get_imputed_df(iterative_imputer, sparse_df[utils.train_strs_order])
    return Response(imputed_df[utils.full_ftdna_strs_order].to_csv(header=False, sep='\t', index=False),
                    mimetype=TEXT_PLAIN)


if __name__ == '__main__':
    print('yImputer ready!')
    serve(app, host="0.0.0.0", port=8080)
