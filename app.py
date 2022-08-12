from flask import Flask, jsonify, request, render_template, redirect, make_response
import pickle
import pandas as pd

app =Flask(__name__)


@app.route("/")
def index():
  return render_template("index.html")

@app.route("/predict", methods = ["GET", "POST"])
def pred():
  df = pd.DataFrame(columns = ['age','sex','bmi', 'children', 'smoker', 'region'])
  if request.method == 'POST':    
    name = request.form["name"]
    age = request.form["age"]
    sex = request.form["sex"]
    bmi = request.form["bmi"]
    children = request.form["children"]
    smoker = request.form["smoker"]
    region = request.form["region"]
    df = df.append({'age': age,'sex': sex,'bmi':bmi, 'children': children, 'smoker':smoker, 'region':region}, ignore_index= True)
  #return df.to_dict()
  with open('model/Lin_reg_model.pkl', 'rb') as file:
    model = pickle.load(file)
    prediction = model.predict(df)
  print(df)
  print(prediction)

  with open('data_collection.txt', 'a') as file:
    file.write("%s\n" % df)

  return render_template("result.html", pred = str("%.2f" %prediction[0]) + "  " + "USD", name = name)
  
  
if __name__ == "__main__":
  app.run()