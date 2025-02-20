import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
data = {
    'Review':['The Product is Great!','Very Poor Quality','Will buy again','Not worth the price','Excellent for the money','very bad Experience','Loved it','Would not Recommend','Amazing Service','Terrible Product'],
    'Sentiment': [1,0,1,0,1,0,1,0,1,0]
}
df = pd.DataFrame(data)
df['Review'] = df['Review'].str.lower()
vectorizer = TfidfVectorizer(stop_words='english')
x = vectorizer.fit_transform(df['Review'])
y = df['Sentiment']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42)
model = LogisticRegression(random_state=42)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test,y_pred))
print("\nconfusion Matrix")
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot=True, fmt='d',cmap='Blues',xticklabels=['Negative','Positive'],yticklabels=['Negative','Positive'])
plt.title('Confusion Matrix')
plt.show()