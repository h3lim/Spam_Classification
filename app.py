import streamlit as st
import nltk
import pickle
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def transform_txt(txt):
  txt = txt.lower()
  txt = nltk.word_tokenize(txt)

  y = []
  for i in txt:
    if i.isalnum():
      y.append(i)
  txt = y[:]
  y.clear()

  for i in txt:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)

  txt = y[:]
  y.clear()

  for i in txt:
    y.append(ps.stem(i))

  return " ".join(y)
nltk.download('punkt')
nltk.download('stopwords')
tf = pickle.load(open('tk.pkl','rb'))
model = pickle.load(open('voting.pkl','rb'))


st.title('Spam Classifier')

input_sms = st.text_input("Please Enter the message")

if st.button('Predict'):
  transform_sms = transform_txt(input_sms)

  vector_input = tf.transform([transform_sms])

  result = model.predict(vector_input)[0]

  if result == 1:
    st.header("Spam")
  else:
    st.header("Ham")