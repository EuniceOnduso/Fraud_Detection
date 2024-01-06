from flask import Flask, jsonify, request
import service
import pandas as pd
from io import StringIO
import joblib

app = Flask(__name__)

model=  joblib.load('./mymodel.pkl')

@app.route("/")
def home():
    return "hello flask developer"
@app.route("/jsonUpload", methods = ["POST"])
def jsonEndpoint():
    data= request.get_json()
    data_frame = service.changeJsonToDataFrame(data)
    return service.myReturn(data_frame, model)

@app.route("/csvUpload", methods=["POST"])
def csvEndpoint():
    try:
        if 'file' not in request.files:
            return jsonify({"response":"no file uploaded"})
        else:
            file = request.files['file']
            if not file.filename.endswith('.csv'):
                return jsonify({"response":"bad file format"})
            else:
                data_frame = pd.read_csv(StringIO(file.stream.read().decode('utf-8')))
                data_frame = data_frame.drop(columns='Class', axis=1)
                return service.myReturn(data_frame, model)
    except Exception as e:
        return str(e), 400

if __name__ == "__main__":
    app.run(debug=True)