import streamlit as st
import pickle
import nltk
import string
from nltk.stem.porter import PorterStemmer
from PIL import Image

from nltk.corpus import stopwords
nltk.download("stopwords")
stpwrd = stopwords.words("english")


ps = PorterStemmer()


img = Image.open(r"img.jpg")
new_img = img.resize((1100,400))
st.image(
          new_img ,
          channels="RGB")

#function
def transform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)

  lst = []

  for i in text:
    if i.isalnum():
      lst.append(i)

  text = lst[:]
  lst.clear()

  for i in text:
    if i not in string.punctuation and i not in stopwords.words("english"):
      lst.append(i)

  text = lst[:]
  lst.clear()

  for i in text:
    lst.append(ps.stem(i))

  return " ".join(lst)


#importing files
tf = pickle.load(open("Vectorizer.pkl","rb"))
model = pickle.load(open("Model.pkl","rb"))


#Setting Title
st.markdown("<h1 style='text-align: center; color: grey;'>Email Spam Classifier</h1>", unsafe_allow_html=True)
st.markdown("---")

email = st.text_area(
    label = " ",
    max_chars= 1000,
    placeholder="Enter Text"
)

def endpoint(email):
  transformed_text = transform_text(email)
  vector_input = tf.transform([transformed_text])
  result = model.predict(vector_input)

  return result


style = "<style>.row-widget.stButton {text-align: center;}</style>"
st.markdown(style, unsafe_allow_html=True)

if st.button("Predict"):

  result = endpoint(email)

  if result == 1:
    st.warning("Spam")
  else:
    st.success("Not Spam")