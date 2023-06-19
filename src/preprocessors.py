import os
import re
import ast
import csv
import math
import nltk
import unicodedata
import pandas as pd
from tqdm import tqdm
from src.helpers import save_as_csv
from src.aggregators import count_rows
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
nltk.download('stopwords')
nltk.download("wordnet")
nltk.download('omw-1.4')


CLEANER = re.compile('<.*?>')
def init_preprocess(input_text):
    """
    Performs initial preprocessing of input_text string.
    The idea is to reduce the number of unique texts in some document by applying this function.
    Preprocessing parts:
        1) removing control characters
        2) removing tags
        3) removing extra spacing

    Parameters
    ----------
    input_text : str
        Text to be preprocessed

    Returns
    -------
    output_text : str
        Preprocessed text

    """
    ### removing control characters
    output_text = "".join(ch if unicodedata.category(ch)[0]!="C" else ' ' for ch in input_text)
    output_text = unicodedata.normalize("NFKD", output_text)
    ### removing tags
    output_text = re.sub(CLEANER, ' ', output_text)
    ### removing extra spacing
    output_text = re.sub(' +', ' ', output_text).lstrip()
    return output_text


STOP_WORDS = stopwords.words("russian")
LEMMATIZER = MorphAnalyzer()
def preprocess_type(input_type):
    """
    Preprocessing of type input_type string.
    Preprocessing parts:
        - removing control characters
        - removing tags
        - removing extra spacing
        - converting characters to lower case
        - keeping only alphabetic symbols
        - removing stop words
        - lemmatizing

    Parameters
    ----------
    input_type : str
        Type to be preprocessed

    Returns
    -------
    output_type : str
        Preprocessed type

    """
    output_type = str(input_type)
    ### removing control characters
    output_type = "".join(ch if unicodedata.category(ch)[0]!="C" else ' ' for ch in output_type)
    output_type = unicodedata.normalize("NFKD", output_type)
    ### removing tags
    output_type = re.sub(CLEANER, ' ', output_type)
    ### removing extra spacing
    output_type = re.sub(' +', ' ', output_type).lstrip()
    ### to lower case
    output_type = output_type.lower()
    ### keeping only alphabetic and numeric symbols
    output_type = re.sub("[^ a-zа-яё]", " ", output_type)
    ### removing extra spacing
    output_type = re.sub(' +', ' ', output_type).lstrip()
    ### remove stop words
    output_type = [word for word in output_type.split(' ') if (word not in STOP_WORDS) and (word != '')]
    try:
        output_type = ' '.join([LEMMATIZER.normal_forms(word)[0] for word in output_type])
    except:
        pass
    return output_type

def preprocess_date(input_date):
    """
    Performs preprocessing of date input_date as string
    It is supposed, that input_date is in the following formats:
        - YYYY:mm:dd[:HH:mm:ss]
        - dd:mm:YYYY[:HH:mm:ss]
    where
        - ":" can be any character except for digit
        - [:HH:mm:ss] possible extension of input date
    The resulting output format is:
        - YYYY/mm/dd[/HH/mm/ss]

    Parameters
    ----------
    input_date : str
        Date to be preprocessed

    Returns
    -------
    output_date : str
        Preprocessed date

    """
    try:
        output_date = re.sub(r'\D', '/', input_date).split('/')
        if len(output_date[2]) == 4:
            output_date[0], output_date[2] = output_date[2], output_date[0]
        output_date = '/'.join(output_date)
    except:
        output_date = None
    return output_date



LEMMATIZER = MorphAnalyzer()
def lemmatize_word(input_word):
    """
    Performs lemmatizing of word input_word

    Parameters
    ----------
    input_word : str
        Word to be lemmatized

    Returns
    -------
    output_word : str
        Lemmatized word

    """
    try:
        output_word = LEMMATIZER.normal_forms(input_word)[0]
    except:
        output_word = input_word
    return output_word


def lemmatize_sentence(input_text, converter):
    """
    Performs lemmatizing of sentence input_text

    Parameters
    ----------
    input_text : "list"
        Sentence to be lemmatized

    converter : dict
        Dictionary, where keys = not lemmatizied words, values = corresponding lemma

    Returns
    -------
    output_text : str
        Lemmatized sentence

    """
    try:
        output_text = [converter[word] for word in ast.literal_eval(input_text)]
    except:
        output_text = input_text
    return output_text


STOP_WORDS = stopwords.words("russian")
def preprocess_text(input_text, remove_stop_words=True):
    """
    Performs preprocessing of text input_text with the following steps:
        - converting characters to lower case
        - replacing some specific characters with appropriate ones
        - keeping only alphabetic and numeric symbols
        - removing extra spacing
        - removing stop words

    Parameters
    ----------
    input_text : str
        Text to be preprocessed

    Returns
    -------
    output_text : str
        Preprocessed text

    """
    try:
        ### to lower case
        output_text = input_text.lower()
        ### dealing with specific characters
        output_text = output_text.replace('%', ' процент ')
        output_text = output_text.replace('&quot', ' ')
        output_text = re.sub('й', 'й', output_text)
        ### keeping only alphabetic and numeric symbols
        output_text = re.sub("[^ 0-9a-zа-яё]", " ", output_text)
        ### removing extra spacing
        output_text = re.sub(' +', ' ', output_text).lstrip()
        ### remove stop words
        if remove_stop_words:
            output_text = [word for word in output_text.split(' ') if (word not in STOP_WORDS) and (word != '')]
    except:
        output_text = None
    return output_text


def preprocess_data_and_get_vocabulary(path_to_read, path_to_write, path_to_vocabulary, filename, chunksize):
    """
    Performs preprocessing of Identifier DataFrame using preprocess_text function and retrieves a vocabulary
    Preprocessing is performed on fragmented data (chunks)

    Parameters
    ----------
    path_to_read : str
        Path to Identifier DataFrame to be preprocessed

    path_to_write : str
        Path to directory where preprocessed Identifier DataFrame would be saved

    path_to_vocabulary : str
        Path to directory where vocabulary would be saved

    filename : str
        Filename of csv file

    chunksize : int
        Size of a single chunk
    """
    ### writing header to new csv file
    with open(os.path.join(path_to_read, filename), 'r', encoding='utf8', newline='') as infile, \
         open(os.path.join(path_to_write, filename), 'w', encoding='utf8', newline='') as outfile:
        csv.writer(outfile).writerow(csv.DictReader(infile).fieldnames)

    ### counting the number of rows
    n_rows = count_rows(path=path_to_read,
                        filename=filename,
                        leave=True)

    ### creating tqdm iterator over chunks
    iterator = tqdm(pd.read_csv(os.path.join(path_to_read, filename),
                                chunksize=chunksize),
                    total=math.ceil((n_rows-1)/chunksize), desc='CHUNKS')
    iterator.set_postfix({'voc_length': 0, 'new_voc_length': 0})

    ### empty list to collect unique words
    vocabulary = []

    ### looping over chunks
    for it, df in enumerate(iterator):
        ### preprocessing chunk
        df['text'] = list(map(lambda x: preprocess_text(x), df['text'].values))
        save_as_csv(path=path_to_write,
                    filename=filename,
                    data=df,
                    mode='a',
                    index=False,
                    header=False)

        ### extraction current vocabulary
        new_vocabulary = df['text'].dropna().values
        del df
        new_vocabulary = list(set([word for list_ in new_vocabulary for word in list_]))
        new_vocabulary_length = len(new_vocabulary)

        ### updating main vocabulary
        vocabulary += new_vocabulary
        del new_vocabulary
        vocabulary = list(set(vocabulary))
        iterator.set_postfix({'voc_length': len(vocabulary), 'new_voc_length': new_vocabulary_length})

    ### sorting and saving vocabulary
    vocabulary = list(sorted(vocabulary))
    vocabulary = pd.DataFrame(data=vocabulary, columns=['word'])
    save_as_csv(path=path_to_vocabulary,
                filename=f'voc_{filename}',
                data=vocabulary,
                mode='w',
                index=False,
                header=True)
    del vocabulary


def preprocess_date_in_data(path_to_read, path_to_write, filename, chunksize):
    """
    Performs preprocessing (for date) of Basis DataFrame using preprocess_date function
    Preprocessing is performed on fragmented data (chunks)

    Parameters
    ----------
    path_to_read : str
        Path to Basis DataFrame to be preprocessed

    path_to_write : str
        Path to directory where preprocessed Basis DataFrame would be saved

    filename : str
        Filename of csv file

    chunksize : int
        Size of a single chunk
    """
    with open(os.path.join(path_to_read, filename), 'r', encoding='utf8', newline='') as infile, \
         open(os.path.join(path_to_write, filename), 'w', encoding='utf8', newline='') as outfile:
        csv.writer(outfile).writerow(csv.DictReader(infile).fieldnames)

    ### counting the number of rows
    n_rows = count_rows(path=path_to_read,
                        filename=filename,
                        leave=True)

    ### creating tqdm iterator over chunks
    iterator = tqdm(pd.read_csv(os.path.join(path_to_read, filename),
                                chunksize=chunksize),
                    total=math.ceil((n_rows-1)/chunksize), desc='CHUNKS')

    ### looping over chunks
    for it, df in enumerate(iterator):
        ### preprocessing chunk
        df['date_parsed'] = list(map(lambda x: preprocess_date(x), df['date_parsed'].values))
        save_as_csv(path=path_to_write,
                    filename=filename,
                    data=df,
                    mode='a',
                    index=False,
                    header=False)
        del df


def lemmatize_vocabulary(path_to_vocabulary, filename):
    """
    Lemmatizes vocabulary and creates lemma_vocabulary with unique lemmas and IDs

    Structure of vocabulary:
        - word
        - lemma_id
    Structure of lemma_vocabulary:
        - lemma
        - id

    Parameters
    ----------
    path_to_vocabulary : str
        Path to vocabulary

    filename : str
        Filename of csv file
    """

    ### reading vocabulary
    vocabulary = pd.read_csv(os.path.join(path_to_vocabulary, filename))

    ### lemmatizing
    vocabulary['lemma'] = list(tqdm(map(lambda x: lemmatize_word(x), vocabulary['word'].values),
                                    total=len(vocabulary)))


    ### creating dataframe dedicated to lemmas
    lemma_vocabulary = pd.DataFrame(vocabulary['lemma'].unique(), columns=['lemma'])
    lemma_vocabulary['id'] = list(range(len(lemma_vocabulary)))

    ### creating new file dedicated to lemmas
    save_as_csv(path=path_to_vocabulary,
                filename='lemma_'+filename,
                data=lemma_vocabulary,
                mode='w',
                index=False,
                header=True)

    ### joining lemma_vocabulary and vocabulary
    lemma_vocabulary = lemma_vocabulary.set_index('lemma')
    vocabulary = vocabulary.join(other=lemma_vocabulary,
                                 on='lemma',
                                 how='left') \
                           .rename(columns={'id': 'lemma_id'}) \
                           .drop('lemma', axis=1)
    del lemma_vocabulary

    ### rewriting file
    save_as_csv(path=path_to_vocabulary,
                filename=filename,
                data=vocabulary,
                mode='w',
                index=False,
                header=True)
    del vocabulary

def lemmatize_text(path_to_read, path_to_write, path_to_vocabulary, filename, chunksize):
    """
    Lemmatizes 'text' column of Identifier DataFrame using vocabulary and
    saves lemmatized data to new column 'lemmatized'
    Lemmatazing is performed on fragmented data (chunks)
    Saves new Identifier DataFrame to a new file 'p_<filename>'
    Deletes original DataFrame

    Parameters
    ----------
    path_to_read : str
        Path to Identifier DataFrame to be lemmatized

    path_to_write : str
        Path to directory where lemmatized Identifier DataFrame would be saved

    path_to_vocabulary : str
        Path to vocabulary

    filename : str
        Filename of csv file

    chunksize : int
        Size of a single chunk
    """
    ### reading vocabulary and transforming it to dict
    vocabulary = pd.read_csv(os.path.join(path_to_vocabulary, 'voc_'+filename))
    vocabulary = dict(zip(vocabulary['word'], vocabulary['lemma_id']))

    ### writing header to new csv file
    with open(os.path.join(path_to_read, filename), 'r', encoding='utf8', newline='') as infile, \
         open(os.path.join(path_to_write, 'p_'+filename), 'w', encoding='utf8', newline='') as outfile:
        csv.writer(outfile).writerow(csv.DictReader(infile).fieldnames + ['lemmatized'])

    ### counting the number of rows
    n_rows = count_rows(path=path_to_read,
                        filename=filename,
                        leave=True)

    ### creating tqdm iterator over chunks
    iterator = tqdm(pd.read_csv(os.path.join(path_to_read, filename),
                                chunksize=chunksize),
                    total=math.ceil((n_rows-1)/chunksize), desc='CHUNKS')

    ### looping over chunks
    for it, df in enumerate(iterator):
        ### lemmatazing chunk
        df['lemmatized'] = list(map(lambda x: lemmatize_sentence(x, vocabulary), df['text'].values))
        save_as_csv(path=path_to_write,
                    filename='p_'+filename,
                    data=df,
                    mode='a',
                    index=False,
                    header=False)

    ### deleting previous file
    #os.remove(os.path.join(path_to_read, filename))





