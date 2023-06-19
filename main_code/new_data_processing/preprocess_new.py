import os
from src.preprocessors import preprocess_data, preprocess_date, preprocess_text

import warnings
warnings.filterwarnings("ignore")


PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))
PATH_TO_READ = os.path.join(PROJECT_PATH, '../../data/raw/separated/')
PATH_TO_WRITE = os.path.join(PROJECT_PATH, '../../data/preprocessed/new_data/')
PATH_TO_APPEND = os.path.join(PROJECT_PATH, '../../data/preprocessed/')
kwargs = {'path_to_read': PATH_TO_READ,
          'path_to_write': PATH_TO_WRITE,
          'path_to_append': PATH_TO_APPEND,
          'prep_func': preprocess_text,
          'columns': ['text'],
          'write_only': False}

print('+'+'='*100)
print('| Starting preprocessing step for NEW data')
##############################################################################################
###                              Preprocessing titles                                      ###
print('+'+'-'*60)
print('| Preprocessing NEW titles ...')
preprocess_data(filename='title.csv', **kwargs)
print('| Done')
##############################################################################################
###                               Preprocessing summaries                                  ###
print('+'+'-'*60)
print('| Preprocessing NEW summaries ...')
preprocess_data(filename='summary.csv', **kwargs)
print('| Done')
##############################################################################################
###                             Preprocessing basis dataframe                              ###
kwargs['prep_func'] = preprocess_date
kwargs['columns'] = ['date', 'date_parsed']
print('+'+'-'*60)
print('| Preprocessing NEW basis data ...')
preprocess_data(filename='main.csv', **kwargs)
print('| Done')
print('+'+'='*100)


##############################################################################################
###                             Update vocabulary for titles                               ###
"""
TODO
"""
##############################################################################################
###                              Update vocabulary for summaries                           ###
"""
TODO
"""
##############################################################################################
###                             Lemmatizing new titles                                     ###
"""
TODO
"""
##############################################################################################
###                             Lemmatizing new summaries                                  ###
"""
TODO
"""


