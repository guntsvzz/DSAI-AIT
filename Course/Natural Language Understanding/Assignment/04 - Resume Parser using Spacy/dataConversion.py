import spacy
from PyPDF2 import PdfReader
from spacy.lang.en.stop_words import STOP_WORDS


nlp = spacy.load('en_core_web_md')
skill_path = "static\skills_educations.jsonl"

ruler = nlp.add_pipe("entity_ruler")
ruler.from_disk(skill_path)
nlp.pipe_names

# before that, let's clean our resume.csv dataframe
def preprocessing(sentence):

    stopwords = list(STOP_WORDS)
    doc = nlp(sentence)
    cleaned_tokens = []

    for token in doc:
        if token.text not in stopwords and token.pos_ != 'PUNCT' and token.pos_ != 'SPACE' and \
                token.pos_ != 'SYM':
            cleaned_tokens.append(token.lemma_.lower().strip())

    return " ".join(cleaned_tokens)

def readPDF(cv_path, page=5):
    reader = PdfReader(cv_path)
    page = reader.pages[page]
    text = page.extract_text()
    text = preprocessing(text)
    doc = nlp(text)

    skills = []
    educations = []

    for ent in doc.ents:
        if ent.label_ == 'SKILL':
            skills.append(ent.text)
        if ent.label_ == 'EDUCATION':
            educations.append(ent.text)
    return set(skills), set(educations)