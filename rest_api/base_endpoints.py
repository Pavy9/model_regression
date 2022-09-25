from flask import jsonify,request
from werkzeug.utils import secure_filename
import os
import settings
from dao.base_dao import models

class BaseBS():
    def allowed_file(self,filename):
        allowed_extensions = ['xls','csv','xlsx']
        extension = filename.split(".")
        if extension[1] in allowed_extensions:
            return True
        else:
            return False

    def file_upload(self):
        if "file" not in request.files.keys():
            resp  = jsonify({"messaqe":"No file"})
            resp.status_code =400
            return resp
        file = request.files['file']
        filename = file.filename
        if file.filename=='':
            resp = jsonify({"messaqe": "No file selected for upload"})
            resp.status_code = 400
            return resp
        if file and self.allowed_file(filename):
            filename = secure_filename(file.filename)
            curdir = settings.curDir + "/data"
            file.save(os.path.join(curdir,filename))
            resp = jsonify({"message" : "File uploaded successfully"})
            resp.status_code =201
            return resp
        else:
            resp = jsonify({"message": "not allowed format"})
            resp.status_code = 201
            return resp

    def model_prediction(self):
        linear = models.linear()
        dt=models.decision_tree()
        rforest= models.random_forest()
        adaboost= models.adaboost()
        xgboost= models.xgboost()
        svr = models.svm()
        result = {"Linear Regression": linear , "Decision Tree": dt,
                  "Random Forest" : rforest , "Adaboost": adaboost,
                  "XGBoost" : xgboost, "Support Vector Machine" : svr}
        return result

class_instance= BaseBS()

