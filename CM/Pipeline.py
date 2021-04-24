import pandas as pd

from nltk.corpus import stopwords
stop = stopwords.words('english')
import string 

def stopwords_removal(textList):
    processed= textList.apply(lambda x: " ".join([word for word in x.split() if word not in (stop)]))
    return processed


def punctuation_removal(textList):
    processed=[]
    for i in textList:
        processed.append(''.join([char.lower() for char in i if char not in string.punctuation]))
    return pd.Series(processed)

    
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def lemWords(textList):
    processed=[]
    for j in textList:
        temp=[]
        for i in j.split():
             temp.append(lemmatizer.lemmatize(i))
        processed.append(" ".join([word for word in temp]))

    return(pd.Series(processed))

def pipeline(textList):
    textList=punctuation_removal(textList)
    textList=stopwords_removal(textList)
    textList=lemWords(textList)
    return textList

##### Sample Code
#import Pipeline
#Pipeline.pipeline(df['text'])