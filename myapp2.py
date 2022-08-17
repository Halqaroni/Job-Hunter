import numpy as np 
import pandas as pd
import streamlit as st
import re
import pdfplumber
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
from streamlit_lottie import st_lottie



st.markdown("# Job Hunter 🔍 ")
st.sidebar.markdown("# 🔍🔍 Get The Best Talent 🔍🔍")
st.sidebar.markdown("It can be very difficult to search for a job, specially with the amount of job postings out there. You may be wasting your time applying to specific jobs that your resume is not tailored for.")


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_hello = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_cjkryupy.json")

st_lottie(
    lottie_hello,
    speed=1,
    reverse=False,
    loop=True,
    quality="", # medium ; high
    height=None,
    width=None,
    key=None,
)

primaryColor="#3872fb"
backgroundColor="#0e1117"
secondaryBackgroundColor="#cece1b"
textColor="#fafafa"
font="sans serif"

def Footer():
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            footer:after {
            content:'Thank You Erfan & Giti';
            visibility: visible;
            display: block;
            position: relative;
            text-align:center;
            color:#4A6AD0;
            #background-color: #CBD1C3; 
            padding: 5px;
            top: 2px; }
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

Footer()
     

st.markdown('This tool will predict what position you may get based on your resume.')

# uploads
jobposts_upload_file = st.file_uploader('STEP 1: Upload job postings with columns: ID, Description,Category',type='csv', accept_multiple_files=False)
resume_upload_file = st.file_uploader('STEP 2: Upload your resume as a PDF', type='pdf', accept_multiple_files=False)

def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URl
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText

def FixResCol(df1):
    df1['Description']=df1['Description'].str.lower()
    df1['Description']=df1['Description'].replace(np.nan, '')
    df1['Description Cleaned'] = ''
    df1['Description Cleaned'] = df1.Description.apply(lambda x: cleanResume(x))
    df1=df1.dropna()
    df1 = df1[['ID', 'Category', "Description"]]
    
def extract_text(feed):
    text = ''
    with pdfplumber.open(feed) as pdf:
        pages = pdf.pages
        for p in pages:
            text += p.extract_text()
    return text

def convert_df(df):
    """
    Dataframe to CSV
    """
    return df.to_csv().encode('utf-8')

if jobposts_upload_file is not None:
    df = pd.read_csv(jobposts_upload_file, encoding='utf8')
    FixResCol(df)

if resume_upload_file is not None:
    pdf=extract_text(resume_upload_file)
    jd = cleanResume(pdf)
    
def magic():
    tfidf = TfidfVectorizer(sublinear_tf= True, #use a logarithmic form for frequency
                       min_df = 5, #The least amount of numbers of documents a word must be present in to be kept
                       norm= 'l2', #This ensure all our feature vectors have a euclidian norm of 1
                       ngram_range= (1,2), #This to indicate that we want to consider both unigrams and bigrams.
                       stop_words ='english',#Removes stop words
                        max_features = 10000)#More than 10k would break the function and notebook
    features = tfidf.fit_transform(df['Description']).toarray()
    train_set=[jd]
    y=tfidf.transform(train_set)
    cosine_sim = cosine_similarity(features,y)
    df["Fitness For Position"]=cosine_sim
    df.sort_values(by=['Fitness For Position'],ascending=False,inplace=True)
    n = 50
    df.drop(df.index[n:], inplace=True)
    df.set_index('ID',inplace=True)
    df.drop(["Description", "Unnamed: 0"], axis=1, inplace=True)
    

    
if st.button('Search'):
    if (jobposts_upload_file is not None) & (resume_upload_file is not None):
        magic()
        csv_candidates = convert_df(df)
        st.download_button("Download", 
                    csv_candidates, 
                    "Top 50 Potential Jobs.csv",
                    "csv",
                    key ='Download-CSV')
        
        
        
