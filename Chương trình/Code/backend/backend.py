from model import predict
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)

@app.route('/')
def home():
    return "Backend"

@app.route('/get-data', methods=['GET'])
@cross_origin()
def getData():
    try:
        # token = request.args.get('token', type=str)
        xlsx = pd.ExcelFile("data/data_final_retrieve.xlsx")
        data = {}
        for sheet in xlsx.sheet_names:
            df = pd.read_excel(xlsx,sheet)
            df = df.fillna("")
            data[sheet] = df.to_dict('records')
        return jsonify({"status":"done", "data":data})
    except Exception as e:
        return jsonify({"status":"error"}), 400

@app.route('/search', methods=['GET'])
@cross_origin()
def search():
    # try:
        keywords = request.args.get('keywords', type=str)
        data = predict(keywords)
        return jsonify({"status":"done", "data":data})
    # except Exception as e:
        # return jsonify({"status":"error"}), 400

if __name__ == '__main__':
    app.debug = True
    app.run(host="0.0.0.0",port=9101)