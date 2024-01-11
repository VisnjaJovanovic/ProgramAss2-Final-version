import gensim
import string
import nltk
import contractions
import unicodedata 

import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.probability import FreqDist

nltk.download('punkt')
nltk.download('stopwords')

import wikipediaapi 

user = 'VisnjaCodingProject/1.0 (visnja.jovanovic@nulondon.ac.uk)'


wiki_wiki = wikipediaapi.Wikipedia(user, 'en')

model = gensim.models.Word2Vec(vector_size = 200, window=10, min_count=5, sg=0)
model.save ("./word2vec.model")

def preprocess_text(text):
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stopwords.words('english')]
    return words

def train_save_model(text):
    processed_text = preprocess_text(text)
    sentences = nltk.sent_tokenize(text)
    processed_sentences = [preprocess_text(sentence) for sentence in sentences]
    
    model.build_vocab (processed_sentences, update=False)
    model.train(processed_sentences, total_examples=model.corpus_count, epochs=model.epochs)
    model.save("./word2vec.model")
    print("finished training on"+ text)

def get_wikipages_content(page_titles):
    combined_text = ""

    for page_title in page_titles:
        page = wiki_wiki.page(page_title)
        if page.exists():
            page_content = page.text
            combined_text += page_content 
    return combined_text

page_titles = ['Polar_bear', 'Artificial_intelligence',
    'Leonardo_da_Vinci', 'Quantum_mechanics',
    'Great_Wall_of_China', 'Neuroscience',
    'The_Mona_Lisa', 'Machine_learning',
    'Amazon_Rainforest', 'Cryptocurrency',
    'Rome', 'Renewable_energy', 'Hawaii', 'Synthetic_biology',
    'Cleopatra', 'Artificial_neural_network',
    'Industrial_Revolution', 'Genome_editing',
    'Michelangelo', 'Dark_matter',
    'The_Three_Musketeers', 'Neuromorphic_computing',
    'Galileo_Galilei', 'Biofuel',
    'Abraham_Lincoln',]

combined_content = get_wikipages_content(page_titles)

train_save_model(combined_content)
