from flask import Flask,render_template,request
from Spam_Detection_Model import prediction

app=Flask(__name__)
@app.route("/",methods=["POST","GET"])
def home():
    if request.method=="POST":
        text_data=str(request.form['emailContent'])
        result=prediction(text_data)
        return render_template("index.html",result=result)
    return render_template("index.html")

if __name__=="__main__":
    app.run(debug=True)


