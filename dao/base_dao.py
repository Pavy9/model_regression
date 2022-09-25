import os
import pandas as pd
import numpy as np
import settings
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR

class BaseDao():
    def load_file(self):
        path= settings.curDir+"/data/"
        os.chdir(path)
        for file in os.listdir():
            if file.endswith(".csv"):
                file_path = f"{path}/{file}"
            elif file.endswith(".xls"):
                file_path = f"{path}/{file}"
            elif file.endswith(".xlsx"):
                file_path = f"{path}/{file}"
        file = pd.read_csv(file_path)
        return file

    def trn_tst_split(self,file):
        X=file.iloc[:,:-1]
        y=file.iloc[:,-1]
        X_train , X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)
        return  X_train , X_test, y_train, y_test

    def model_performance(self,y_test,y_pred):
        MAE = metrics.mean_absolute_error(y_test,y_pred)
        MSE = metrics.mean_squared_error(y_test,y_pred)
        RMSE = np.sqrt(metrics.mean_squared_error(y_test,y_pred))

        return MAE,MSE,RMSE

    def linear (self):
        try:
            file = self.load_file()
            file = pd.get_dummies(file,dummy_na=True)
            X_train, X_test, y_train, y_test = self.trn_tst_split(file)
            regressor = LinearRegression()
            regressor.fit(X_train,y_train)
            y_pred= regressor.predict(X_test)
            MAE,MSE,RMSE = self.model_performance(y_test,y_pred)

            result ={"model":"Linear regression" , "Mean Absolute Error" : MAE ,"Mean Squared Error" : MSE, "Root Mean Squared Error": RMSE}
            return result
        except Exception as e:
            print ("Exception handled in Linear Regression",e)
            result ={"Exception":e}
            return result

    def decision_tree(self):
        try:
            file = self.load_file()
            X_train, X_test, y_train, y_test = self.trn_tst_split(file)
            regressor = DecisionTreeRegressor(random_state =0 )
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_test)
            MAE, MSE, RMSE = self.model_performance(y_test, y_pred)

            result = {"model": "Linear regression", "Mean Absolute Error": MAE, "Mean Squared Error": MSE,
                      "Root Mean Squared Error": RMSE}
            return result
        except Exception as e:
            print("Exception handled in Decision Tree Regression", e)
            result = {"Exception": e}
            return result


    def random_forest(self):
        try:
            file = self.load_file()
            X_train, X_test, y_train, y_test = self.trn_tst_split(file)
            regressor = RandomForestRegressor(n_estimators=100,random_state =0 )
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_test)
            MAE, MSE, RMSE = self.model_performance(y_test, y_pred)

            result = {"model": "Random Forest regression", "Mean Absolute Error": MAE, "Mean Squared Error": MSE,
                      "Root Mean Squared Error": RMSE}
            return result
        except Exception as e:
            print("Exception handled in Random Forest regression", e)
            result = {"Exception": e}
            return result


    def adaboost(self):
        try:
            file = self.load_file()
            X_train, X_test, y_train, y_test = self.trn_tst_split(file)
            regressor = AdaBoostRegressor()
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_test)
            MAE, MSE, RMSE = self.model_performance(y_test, y_pred)

            result = {"model": "Adaboost regression", "Mean Absolute Error": MAE, "Mean Squared Error": MSE,
                      "Root Mean Squared Error": RMSE}
            return result
        except Exception as e:
            print("Exception handled in Adaboost regression", e)
            result = {"Exception": e}
            return result

    def xgboost(self):
        try:
            file = self.load_file()
            file = pd.get_dummies(file, dummy_na=True)
            X_train, X_test, y_train, y_test = self.trn_tst_split(file)
            regressor = XGBRegressor()
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_test)
            MAE, MSE, RMSE = self.model_performance(y_test, y_pred)

            result = {"model": "XGBoost regression", "Mean Absolute Error": MAE, "Mean Squared Error": MSE,
                      "Root Mean Squared Error": RMSE}
            return result
        except Exception as e:
            print("Exception handled in XGBoost regression", e)
            result = {"Exception": e}
            return result

    def svm(self):
        try:
            file = self.load_file()
            file = pd.get_dummies(file, dummy_na=True)
            X_train, X_test, y_train, y_test = self.trn_tst_split(file)
            regressor = SVR(kernel='rbf')
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_test)
            MAE, MSE, RMSE = self.model_performance(y_test, y_pred)

            result = {"model": "Support vector regression", "Mean Absolute Error": MAE, "Mean Squared Error": MSE,
                      "Root Mean Squared Error": RMSE}
            return result
        except Exception as e:
            print("Exception handled in Support vector regression", e)
            result = {"Exception": e}
            return result



models=BaseDao()