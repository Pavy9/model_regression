from flask import Flask
import settings
import connexion
import sys

sys.path.insert(0,settings.curDir)

flask_app = Flask (__name__)
app= connexion.FlaskApp (__name__)

@app.route("/")
def home():
    return "Prediction"

if __name__== '__main__':
    app.add_api('swagger/sw_config.yaml',arguments ={'tittle': 'Prediction'})
    app.run(debug=True)
