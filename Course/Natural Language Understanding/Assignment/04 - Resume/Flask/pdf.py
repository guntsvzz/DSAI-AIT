import numpy as np
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load('en_core_web_md')
skill_path = "static/skills.jsonl"

ruler = nlp.add_pipe("entity_ruler")
ruler.from_disk(skill_path)
nlp.pipe_names

#before that, let's clean our resume.csv dataframe
def preprocessing(sentence):

    stopwords = list(STOP_WORDS)

    doc = nlp(sentence)

    cleaned_tokens = []

    for token in doc:
        if token.text not in stopwords and token.pos_ != 'PUNCT' and \
            token.pos_ != 'SPACE' and token.pos_ != 'SYM':
                cleaned_tokens.append(token.lemma_.lower().strip())
    
    return " ".join(cleaned_tokens)

from PyPDF2 import PdfReader

def readPDF(pdf,num_page = 0):
    reader = PdfReader(pdf)
    page = reader.pages[num_page] #first page just for demo
    text = page.extract_text() 
    text = preprocessing(text)
    doc = nlp(text)

    educations = []
    skills = []

    for ent in doc.ents:
        if ent.label_ == 'SKILL':
            skills.append(ent.text)
        if ent.label_ == 'EDUCATION':
            educations.append(ent.text)

    return set(skills) #set(educations)