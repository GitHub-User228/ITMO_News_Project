import os
from src.helpers import save_as_csv
from src.reformers import main_splitter, NewDataMergerer

import warnings
warnings.filterwarnings("ignore")

PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))
STORAGE_PATH = os.path.join(PROJECT_PATH, '../../data/raw/unseparated/')
FILENAME = 'raw_data_new.csv'


# Part I - split new data
dfs, sets = main_splitter(chunksize=None,
                          path=os.path.join(STORAGE_PATH, FILENAME),
                          cols=['title', 'summary', 'link'],
                          all_cols=['title', 'summary', 'type',
                                    'date', 'link', 'source',
                                    'date_parsed'],
                          do_preprocessing=[True, True, False],
                          indexer1='')

# Part II - create indexation for new dataset based on indexation of main dataset
dfs, sets = NewDataMergerer().merge(path=os.path.join(PROJECT_PATH, '../../data/raw/separated/'),
                                    filenames=[key+'.csv' for key in sets.keys()],
                                    data_new=dfs,
                                    fields_new=sets)

# Part III - append new dataset to main dataset
STORAGE_PATH = os.path.join(PROJECT_PATH, '../../data/raw/separated/')
save_as_csv(path=STORAGE_PATH, filename='main.csv', data=dfs, mode='a', index=False)
save_as_csv(path=STORAGE_PATH, filename='title.csv', data=sets['title'], mode='a', index=True)
save_as_csv(path=STORAGE_PATH, filename='summary.csv', data=sets['summary'], mode='a', index=True)
save_as_csv(path=STORAGE_PATH, filename='link.csv', data=sets['link'], mode='a', index=True)

# Part IV - save new data separately
STORAGE_PATH = os.path.join(PROJECT_PATH, '../../data/raw/separated/new_data')
save_as_csv(path=STORAGE_PATH, filename='main.csv', data=dfs, mode='w', index=False)
save_as_csv(path=STORAGE_PATH, filename='title.csv', data=sets['title'], mode='w', index=True)
save_as_csv(path=STORAGE_PATH, filename='summary.csv', data=sets['summary'], mode='w', index=True)
save_as_csv(path=STORAGE_PATH, filename='link.csv', data=sets['link'], mode='w', index=True)