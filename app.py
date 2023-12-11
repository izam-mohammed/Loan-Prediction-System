from flask import Flask, request, jsonify
from loanPrediction.config.configuration import ConfigurationManager
from loanPrediction.pipeline.prediction import PredictionPipeline
from loanPrediction.utils.common import load_json
import pandas as pd
from pathlib import Path
from loanPrediction import logger
import os

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    return "Use API call to /predict_data"


@app.route("/train", methods=["GET", "POST"])
def training():
    os.system("python main.py")
    return "Training Successful!"


# API call
@app.route("/predict_data", methods=["POST", "GET"])
def pred():
    try:
        gender = request.args.get("gender")
        married = request.args.get("married")
        education = request.args.get("education")
        self_employed = request.args.get("self_employed")
        applicant_income = request.args.get("applicant_income")
        coapplicant_income = request.args.get("coapplicant_income")
        loan_amount = request.args.get("loan_amount")
        loan_amount_term = request.args.get("loan_amount_term")
        credit_history = request.args.get("credit_history")
        property_area = request.args.get("property_area")

        logger.info("\n\nX" + "==" * 16 + "Predicting" + "==" * 16 + "X")

        config = ConfigurationManager()
        config = config.get_prediction_config()
        data_path = config.data_path

        data = pd.DataFrame(
            {
                "Gender": [gender],
                "Married": [married],
                "Education": [education],
                "Self_Employed": [self_employed],
                "ApplicantIncome": [applicant_income],
                "CoapplicantIncome": [coapplicant_income],
                "LoanAmount": [loan_amount],
                "Loan_Amount_Term": [loan_amount_term],
                "Credit_History": [credit_history],
                "Property_Area": [property_area],
            }
        )
        data.to_csv(data_path, index=False)

        obj = PredictionPipeline()
        obj.main()

        file = load_json(path=Path(config.prediction_file))
        predict = file["prediction"]
        logger.info("\nX" + "==" * 38 + "X\n")

        return jsonify({"result": bool(predict)})

    except Exception as e:
        print("The Exception message is: ", e)
        return jsonify({"result": "error"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
