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
    for palindromes in [['DYS385a', 'DYS385b'], ['DYS459a', 'DYS459b'], ['DYS464a', 'DYS464b', 'DYS464c', 'DYS464d'],
                        ['CDYa', 'CDYb'], ['DYF395S1a', 'DYF395S1b'], ['DYS413a', 'DYS413b'], ['YCAIIa', 'YCAIIb']]:
        imputed_df[palindromes[0][:-1]] = imputed_df[palindromes].astype(str).apply("-".join, axis=1)
        imputed_df = imputed_df.drop(columns=palindromes)
    return Response(imputed_df[utils.ftdna_strs_order].to_csv(header=False, sep='\t', index=False),
                    mimetype=TEXT_PLAIN)


if __name__ == '__main__':
    print('yImputer ready!')
    serve(app, host="0.0.0.0", port=8080)
