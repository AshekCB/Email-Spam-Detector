import pandas as pd

data=pd.read_csv("D:\Flask Demo's\Email Spam Detector\email spam detection.csv")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Split the data
X = data['Message']  # Text data
y = data['Category']    # Target labels
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert text data to numerical data using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
x_train_tfidf = vectorizer.fit_transform(x_train)
x_test_tfidf = vectorizer.transform(x_test)  # Use `transform` on test data, not `fit_transform`

# Train the model
model = RandomForestClassifier()
model.fit(x_train_tfidf, y_train)

# Predict on the test data
y_pred = model.predict(x_test_tfidf)
# Evaluate the accuracy
#print("Accuracy:", round(accuracy_score(y_test, y_pred)*100,2))


# Sample text prediction
'''
text = ["Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat..."]

# Transform the text into numerical format
text_transformed = vectorizer.transform(text)

# Predict
res = model.predict(text_transformed)

if res[0]=='ham':
    print("It Is Not A Spam Message")
'''

# defining the function for prediction of user inputs

def prediction(text):
    if len(text)<=10:
        return "The Email Content Cannot be less then 10 characters and Please provide the proper content 🤝....!!!"
    
    text_vector=vectorizer.transform([text])
    res=model.predict(text_vector)
    
    if res[0] == 'ham':
        return f"😀 The content of email you provided is **NOT spam**. \n It seems harmless and safe!"
    elif res[0]=='spam':
        return f"⚠️ The content of email you provided is identified as **SPAM**. \nPlease be cautious before trusting this email."
    else:
        return f"😔 Sorry Something Goes Wrong....!!!"
    

#sample check
'''
text="Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"
t1="Abhishek"
res=prediction(t1)#--> o/p --> spam
print("The Result from Model is : \n",res)
'''
    