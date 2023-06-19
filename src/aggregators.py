import os
import csv
import ast
import math
from tqdm import tqdm
import numpy as np
from tqdm.notebook import tqdm as tqdm2
import pandas as pd
from collections import Counter
from src.helpers import save_as_npz, get_size, csr_vappend
from scipy.sparse import lil_matrix
from sklearn.preprocessing import minmax_scale



def counter_unique_with_grouping(path, filename, chunksize, n_rows,
                                groups, column_to_group, columns_to_process,
                                renamed_columns, max_rows=None):
    """
    TODO
    """
    ### creating iterator
    n_chunks = math.ceil((n_rows - 1) / chunksize)
    if max_rows is not None:
        n_chunks = math.ceil((min(max_rows, n_rows) - 1) / chunksize)
    iterator = tqdm(pd.read_csv(os.path.join(path, filename), chunksize=chunksize, nrows=max_rows),
                    total=n_chunks, desc='CHUNKS')

    ### creating empty dict
    items = pd.DataFrame(data=[[[] for _ in renamed_columns] for _ in groups], index=sorted(groups), columns=renamed_columns)

    ### looping through chunks
    for df in iterator:
        df = df[[column_to_group]+columns_to_process].rename(columns=dict(zip(columns_to_process,
                                                                              renamed_columns)))
        for col in renamed_columns:
            vals = df[[column_to_group, col]].dropna() \
                                             .groupby(column_to_group)[col] \
                                             .apply(set) \
                                             .apply(list)
            del df[col]
            items.loc[vals.index, col] += vals
            items.loc[vals.index, col] = items.loc[vals.index, col] \
                                              .apply(set) \
                                              .apply(list)
            del vals

    counts = pd.DataFrame(index=items.index, columns=items.columns)
    for col in counts.columns:
        counts[col] = items[col].apply(len)

    return counts, items



def counter_with_grouping(path, filename, chunksize, n_rows,
                          groups, column_to_group, columns_to_process,
                          renamed_columns):
    """
    TODO
    """
    ### creating iterator
    n_chunks = math.ceil((n_rows - 1) / chunksize)
    iterator = tqdm(pd.read_csv(os.path.join(path, filename), chunksize=chunksize),
                    total=n_chunks, desc='CHUNKS')

    ### creating empty dataset
    data = pd.DataFrame(index=groups, columns=renamed_columns) \
             .rename_axis(column_to_group)

    ### looping through chunks
    for df in iterator:
        df = df[[column_to_group]+columns_to_process].groupby(column_to_group) \
                                                     .count() \
                                                     .rename(columns=dict(zip(columns_to_process,
                                                                              renamed_columns))) \
                                                     .rename_axis(column_to_group)
        data = pd.concat([data, df], axis=0, ignore_index=False) \
                 .groupby(column_to_group) \
                 .sum()

    data.loc['TOTAL'] = data.sum().values
    data[renamed_columns] = data[renamed_columns].astype('int32')

    return data



def counter(path, filename, chunksize, n_rows, columns_to_process, renamed_columns, prep_f):
    """
    TODO
    """
    ### creating iterator
    n_chunks = math.ceil((n_rows - 1) / chunksize)
    iterator = tqdm(pd.read_csv(os.path.join(path, filename), chunksize=chunksize),
                    total=n_chunks, desc='CHUNKS')

    ### creating empty dataset
    data = pd.DataFrame(data=[[0 for _ in renamed_columns]], index=renamed_columns, columns=['count']) \
             .rename_axis('cols')

    ### looping through chunks
    for df in iterator:
        df = df[columns_to_process].rename(columns=dict(zip(columns_to_process, renamed_columns)))
        for col in renamed_columns:
            sum_ = df.loc[df[col].notnull(), col].apply(prep_f).sum()
            data.loc[col, 'count'] += sum_

    return data

def collector(path, filename, chunksize, n_rows, columns_to_process, renamed_columns, prep_f=None):
    """
    TODO
    """
    ### creating iterator
    n_chunks = math.ceil((n_rows - 1) / chunksize)
    iterator = tqdm(pd.read_csv(os.path.join(path, filename), chunksize=chunksize),
                    total=n_chunks, desc='CHUNKS')

    ### creating empty dataset
    data = pd.DataFrame(data=[[[] for _ in renamed_columns]], index=renamed_columns, columns=['items']) \
             .rename_axis('cols')

    ### looping through chunks
    for df in iterator:
        df = df[columns_to_process].rename(columns=dict(zip(columns_to_process, renamed_columns)))
        for col in renamed_columns:
            if prep_f is None:
                data.loc[col, 'items'] += df.loc[df[col].notnull(), col].values.tolist()
            else:
                data.loc[col, 'items'] += df.loc[df[col].notnull(), col].apply(prep_f).values.tolist()

    return data



def count_rows(path, filename, leave=True):
    """
    Counts the number of rows in csv file stored in 'path'

    Parameters
    ----------
    path : str
        Path to csv file

    filename : str
        Filename of csv file

    leave : bool
        Whether to keep tqdm iterator after counting

    Returns
    -------
    count : int
        Number of rows in csv file

    """
    with open(os.path.join(path, filename), 'r', encoding='utf8', newline='') as fp:
        reader = csv.reader(fp)
        count = sum([1 for _ in tqdm(reader, leave=leave, desc='counting rows...')])
    return count



def get_unique_symbols(documents):
    """
    Extracts unique symbols in a set of documents

    Parameters
    ----------
    documents : array_like of strings

    Returns
    -------
    symbols : str
        Unique symbols joined in str format

    """
    return ''.join(sorted(''.join(list(set(' '.join(list(documents)))))))

def count_unique_symbols(documents):
    """
    Count unique symbols in a set of documents

    Parameters
    ----------
    documents : array_like of strings

    Returns
    -------
    symbols : dict
        Sorted dictionary where keys = symbols, values = corresponding number of occurrences in documents

    """
    symbols = sorted(''.join(list(set(' '.join(list(documents))))))
    symbols = dict(zip(symbols, [0]*len(symbols)))
    symbols = {}
    for doc in tqdm(documents):
        symbls = sorted(list(set(doc)))
        counts = [doc.count(s) for s in symbls]
        for it, smb in enumerate(symbls):
            if smb in symbols.keys():
                symbols[smb] += counts[it]
            else:
                symbols[smb] = counts[it]
    symbols = dict(sorted(symbols.items(), key=lambda item: item[1], reverse=True))
    return symbols


def count_words_in_chunk(data, vocabulary):
    """
    TODO
    """
    counts1 = lil_matrix((len(data), len(vocabulary)), dtype='uint8')
    #counts2 = list(tqdm2(map(lambda x: Counter(ast.literal_eval(x)), data['lemmatized']), total=len(data), leave=False))
    counts2 = list(map(lambda x: Counter(ast.literal_eval(x)), data['lemmatized']))
    keys = [list(k.keys()) for k in counts2]
    values = [list(k.values()) for k in counts2]
    #for it in tqdm2(range(len(counts2)), leave=False, position=1):
    for it in range(len(counts2)):
        try:
            counts1[it, keys[it]] = values[it]
        except:
            pass
    #del keys, values, counts2
    return counts1.tocsr()


def count_words(path_to_read, path_to_write, path_to_vocabulary,
                filename_to_read, filename_of_vocabulary, filename_to_write, chunksize):
    """
    TODO
    """
    ### reading vocabulary
    vocabulary = pd.read_csv(os.path.join(path_to_vocabulary, filename_of_vocabulary))

    ### counting the number of rows to be processed
    n_rows = count_rows(path=path_to_read,
                        filename=filename_to_read,
                        leave=True)

    ### creating tqdm iterator over chunks
    iterator = tqdm(pd.read_csv(os.path.join(path_to_read, filename_to_read),
                                chunksize=chunksize),
                    total=math.ceil((n_rows - 1) / chunksize), desc='CHUNKS')
    iterator.set_postfix({'size in Kb': 0, 'non_zero': 0})
    non_zero = 0

    ### looping over chunks
    for it, df in enumerate(iterator):
        if it == 0:
            counts = count_words_in_chunk(data=df, vocabulary=vocabulary)
        else:
            counts_new = count_words_in_chunk(data=df, vocabulary=vocabulary)
            counts = csr_vappend(counts, counts_new)
            del counts_new

        ### update postfix
        non_zero += counts.count_nonzero()
        iterator.set_postfix({'size in Kb': get_size(counts, metric='Kb'), 'non_zero': non_zero})

    ### save data
    save_as_npz(path=path_to_write, filename=filename_to_write, data=counts)
    del counts


def log(label):
    """
    TODO
    """
    def decorator(function):
        def wrapper(*args, **kwargs):
            print(f'{label}... ', end='\n')
            result = function(*args, **kwargs)
            return result
        return wrapper
    return decorator



class TopWords:
    """
    TODO
    """
    @log('grouping')
    def grouping(self, main_data, col_to_group, text_col):
        """
        TODO
        """
        grouped = main_data.groupby([col_to_group])[text_col].apply(list).to_frame()
        return grouped

    @log('mapping column to group')
    def mapping_column_to_group(self, grouped, col_to_group, text_col, mapping_values, max_groups=5):
        """
        TODO
        """
        if mapping_values is not None:
            grouped.index = mapping_values
            grouped = grouped.rename_axis(col_to_group) \
                .groupby([col_to_group]) \
                .agg({text_col: lambda group: [item for items in group for item in items]})
            grouped['count'] = grouped[text_col].apply(len)
            grouped = grouped.sort_values(by='count', ascending=False) \
                          .iloc[:max_groups] \
                .drop('count', axis=1)
        return grouped

    @log('counting texts')
    def counting(self, grouped, text_col):
        """
        TODO
        """
        grouped['count'] = grouped[text_col].apply(lambda x: list(Counter(x).values()))
        grouped[text_col] = grouped[text_col].apply(lambda x: list(Counter(x).keys()))
        return grouped

    @log('converting texts to sets of tokens')
    def text_to_tokens(self, grouped, text_col, text_data, tokens_col='lemmatized', labels_col='label'):
        """
        TODO
        """
        grouped['tokens'] = grouped[text_col].apply(lambda x: text_data.loc[[k for k in x if type(k)==str], tokens_col].tolist())
        grouped['label'] = grouped[text_col].apply(lambda x: text_data.loc[[k for k in x if type(k)==str], labels_col].tolist())
        del grouped[text_col]
        return grouped

    @log('calculating tf')
    def calculating_tf(self, grouped, label=None, block_list=[]):
        """
        TODO
        """
        grouped['top_tokens_tf'] = None
        grouped['top_tokens_tf'] = grouped['top_tokens_tf'].astype(object)
        grouped['tf'] = None
        grouped['tf'] = grouped['tf'].astype(object)
        if label is None:
            f = lambda x: [(item, x['count'][it]) for it, items in enumerate(x['tokens']) for item in items if (item not in block_list)]
        else:
            f = lambda x: [(item, x['count'][it]) for it, items in enumerate(x['tokens']) for item in items if (x['label'][it]==label) and (item not in block_list)]
        for index, row in tqdm(grouped.iterrows(), total=len(grouped)):
            top_tokens = pd.DataFrame(f(row), columns=['token','count']) \
                .groupby('token') \
                .sum() \
                .sort_values(by='count', ascending=False)
            grouped.at[index, 'top_tokens_tf'] = top_tokens.index.tolist()
            grouped.at[index, 'tf'] = np.array(top_tokens['count'].tolist())/sum(top_tokens['count'])
            del top_tokens
        del grouped['count'], grouped['tokens'], grouped['label']
        return grouped

    @log('calculating tfidf')
    def calculating_tfidf(self, grouped, top):
        """
        TODO
        """
        tokens_set = []
        for index, row in tqdm(grouped.iterrows(), total=len(grouped)):
            tokens_set += grouped.at[index, 'top_tokens_tf']
        tokens_set = Counter(tokens_set)
        idf = pd.DataFrame(columns=['token', 'idf'])
        idf['token'] = list(tokens_set.keys())
        idf['idf'] = np.array(list(tokens_set.values()))
        del tokens_set
        idf['idf'] = np.log(len(grouped) / idf['idf'])
        idf = idf.set_index('token')
        grouped['tfidf'] = None
        grouped['tfidf'] = grouped['tfidf'].astype(object)
        grouped['top_tokens_tfidf'] = None
        grouped['top_tokens_tfidf'] = grouped['top_tokens_tfidf'].astype(object)
        grouped['df'] = None
        grouped['df'] = grouped['df'].astype(object)
        for index, row in tqdm(grouped.iterrows(), total=len(grouped)):
            grouped.at[index, 'tfidf'] = grouped.at[index, 'tf'] * np.array(idf.loc[grouped.at[index, 'top_tokens_tf'], 'idf'])
            grouped.at[index, 'df'] = pd.DataFrame(columns=['tokens','tfidf'])
            grouped.at[index, 'df']['tokens'] = grouped.at[index, 'top_tokens_tf']
            grouped.at[index, 'df']['tfidf'] = grouped.at[index, 'tfidf']
            grouped.at[index, 'df'] = grouped.at[index, 'df'].sort_values(by='tfidf', ascending=False)
            grouped.at[index, 'top_tokens_tfidf'] = grouped.at[index, 'df'].iloc[:top]['tokens'].tolist()
            grouped.at[index, 'tfidf'] = grouped.at[index, 'df'].iloc[:top]['tfidf'].tolist()
            grouped.at[index, 'top_tokens_tf'] = grouped.at[index, 'top_tokens_tf'][:top]
            grouped.at[index, 'tf'] = grouped.at[index, 'tf'][:top]
            grouped.at[index, 'df'] = None
        del idf, grouped['df']
        return grouped

    @log('converting tokens to words')
    def tokens_to_words(self, grouped, vocab):
        """
        TODO
        """
        tqdm.pandas()
        grouped['top_tokens_tf'] = grouped['top_tokens_tf'].progress_apply(lambda x: [vocab[k] if type(k)==int else k for k in x])
        grouped['top_tokens_tfidf'] = grouped['top_tokens_tfidf'].progress_apply(lambda x: [vocab[k] if type(k)==int else k for k in x])
        return grouped


    def find(self, main_data, text_data, col_to_group, text_col, vocab,
             tokens_col='lemmatized', labels_col='label', label=None, top=10,
             mapping_values=None, max_groups=5, block_list=[]):
        """
        TODO
        """
        grouped = self.grouping(main_data, col_to_group, text_col)
        grouped = self.mapping_column_to_group(grouped, col_to_group, text_col, mapping_values, max_groups)
        grouped = self.counting(grouped, text_col)
        grouped = self.text_to_tokens(grouped, text_col, text_data, tokens_col, labels_col)
        grouped = self.calculating_tf(grouped, label, block_list)
        grouped = self.calculating_tfidf(grouped, top)
        grouped = self.tokens_to_words(grouped, vocab)
        return grouped



class SentimentsRate:
    """
    TODO
    """
    @log('grouping')
    def grouping(self, main_data, col_to_group, text_col):
        """
        TODO
        """
        grouped = main_data.dropna(subset=[col_to_group, text_col], axis=0) \
                           .groupby(col_to_group)[text_col].apply(list).to_frame()
        return grouped

    @log('mapping column to group')
    def mapping_column_to_group(self, grouped, col_to_group, text_col, mapping_df, max_groups=5):
        """
        TODO
        """
        if mapping_df is not None:
          grouped = grouped.join(mapping_df[[f'reformed_{col_to_group}']], how='left') \
                            .reset_index(drop=True) \
                            .set_index(f'reformed_{col_to_group}')
          grouped = grouped.rename_axis(col_to_group) \
                          .groupby([col_to_group]) \
                          .agg({text_col: lambda group: [item for items in group for item in items]})
          grouped['count'] = grouped[text_col].apply(len)
          grouped = grouped.sort_values(by='count', ascending=False) \
                          .iloc[:max_groups] \
                          .drop('count', axis=1)
        return grouped

    @log('counting texts')
    def counting(self, grouped, text_col):
        """
        TODO
        """
        grouped['count'] = grouped[text_col].apply(lambda x: list(Counter(x).values()))
        grouped[text_col] = grouped[text_col].apply(lambda x: list(Counter(x).keys()))
        return grouped

    @log('getting labels')
    def get_labels(self, grouped, text_col, text_data, labels_col='label'):
      """
      TODO
      """
      tqdm.pandas()
      grouped['label'] = grouped[text_col].progress_apply(lambda x: text_data.loc[x, labels_col].tolist())
      return grouped

    @log('calculating rates of sentiments')
    def calculating_rates(self, grouped, text_col):
        """
        TODO
        """
        grouped['neg_count'], grouped['pos_count'] = None, None
        grouped['neg_rate'], grouped['pos_rate'] = None, None
        f0 = lambda x: [x['count'][it] for it, item in enumerate(x[text_col]) if (x['label'][it]==0)]
        f1 = lambda x: [x['count'][it] for it, item in enumerate(x[text_col]) if (x['label'][it]==1)]
        for index, row in tqdm(grouped.iterrows(), total=len(grouped)):
            row['neg_count'] = sum(f0(row))
            row['pos_count'] = sum(f1(row))
            row['neg_rate'] = row['neg_count']/(row['neg_count'] + row['pos_count'])
            row['pos_rate'] = row['pos_count']/(row['neg_count'] + row['pos_count'])
        del grouped['count'], grouped[text_col], grouped['label']
        print('')
        return grouped

    @log('normalizing rates')
    def normalize(self, data, scaling):
        """
        TODO
        """
        if scaling=='custom':
          data['count'] = (data['neg_count'] + data['pos_count'])
          data['weight'] = data['count']/sum(data['count'])
          data['weight'] = max(data['weight']) - data['weight']
        elif scaling=='minmax':
          data['weight'] = 1 - minmax_scale(data['neg_count'] + data['pos_count'])
        else:
          return data
        dx = (data['pos_rate'] - data['neg_rate'])*data['weight']
        data['pos_rate'] -= dx
        data['neg_rate'] += dx
        return data

    @log('reshaping dataframe')
    def reshape_df(self, df, col_to_group):
        """
        TODO
        """
        df0 = df.reset_index()[[col_to_group, 'neg_rate']] \
            .rename(columns={'neg_rate': 'rate'}) \
            .sort_values(by='rate')
        df0['label'] = 'negative'
        df1 = df.reset_index()[[col_to_group, 'pos_rate']] \
            .rename(columns={'pos_rate': 'rate'}) \
            .sort_values(by='rate')
        df1['label'] = 'non-negative'
        df_res = pd.concat([df0, df1], axis=0, ignore_index=True)
        df_res['rate'] = df_res['rate']*100
        return df_res

    def find(self, main_data, text_data, col_to_group, text_col,
           labels_col='label', mapping_df=None, max_groups=5,
           scaling='custom'):
        """
        TODO
        """
        grouped = self.grouping(main_data, col_to_group, text_col)
        grouped = self.mapping_column_to_group(grouped, col_to_group, text_col, mapping_df, max_groups)
        grouped = self.counting(grouped, text_col)
        grouped = self.get_labels(grouped, text_col, text_data, labels_col)
        grouped = self.calculating_rates(grouped, text_col)
        grouped = self.normalize(grouped, scaling)
        grouped = self.reshape_df(grouped, col_to_group)
        return grouped


class SentimentsRateMulti:
    """
    TODO
    """
    @log('grouping')
    def grouping(self, main_data, cols_to_group, text_col):
        """
        TODO
        """
        grouped = main_data.dropna(subset=cols_to_group + [text_col], axis=0) \
                           .groupby(cols_to_group)[text_col].apply(list).to_frame()
        return grouped

    @log('counting texts')
    def counting(self, grouped, text_col):
        """
        TODO
        """
        grouped['count'] = grouped[text_col].apply(lambda x: list(Counter(x).values()))
        grouped[text_col] = grouped[text_col].apply(lambda x: list(Counter(x).keys()))
        return grouped

    @log('getting labels')
    def get_labels(self, grouped, text_col, text_data, labels_col='label'):
        """
        TODO
        """
        tqdm.pandas()
        grouped['label'] = grouped[text_col].progress_apply(lambda x: text_data.loc[x, labels_col].tolist())
        return grouped

    @log('calculating rates of sentiments')
    def calculating_rates(self, grouped, text_col):
        """
        TODO
        """
        grouped['neg_count'], grouped['pos_count'] = None, None
        grouped['neg_rate'], grouped['pos_rate'] = None, None
        f0 = lambda x: [x['count'][it] for it, item in enumerate(x[text_col]) if (x['label'][it]==0)]
        f1 = lambda x: [x['count'][it] for it, item in enumerate(x[text_col]) if (x['label'][it]==1)]
        for index, row in tqdm(grouped.iterrows(), total=len(grouped)):
            row['neg_count'] = sum(f0(row))
            row['pos_count'] = sum(f1(row))
            row['neg_rate'] = row['neg_count']/(row['neg_count'] + row['pos_count'])
            row['pos_rate'] = row['pos_count']/(row['neg_count'] + row['pos_count'])
        del grouped['count'], grouped[text_col], grouped['label']
        print('')
        return grouped

    @log('normalizing rates')
    def normalize(self, data, scaling):
        """
        TODO
        """
        if scaling=='custom':
          data['count'] = (data['neg_count'] + data['pos_count'])
          data['weight'] = data['count']/sum(data['count'])
          data['weight'] = max(data['weight']) - data['weight']
        elif scaling=='minmax':
          data['weight'] = 1 - minmax_scale(data['neg_count'] + data['pos_count'])
        else:
          return data
        dx = (data['pos_rate'] - data['neg_rate'])*data['weight']
        data['pos_rate'] -= dx
        data['neg_rate'] += dx
        return data

    @log('reshaping dataframe')
    def reshape_df(self, df, cols_to_group):
        """
        TODO
        """
        df = df.reset_index()[cols_to_group+['neg_rate']]
        df['neg_rate'] = df['neg_rate']*100
        return df

    def find(self, main_data, text_data, cols_to_group, text_col, labels_col='label', scaling=None):
        """
        TODO
        """
        grouped = self.grouping(main_data, cols_to_group, text_col)
        grouped = self.counting(grouped, text_col)
        grouped = self.get_labels(grouped, text_col, text_data, labels_col)
        grouped = self.calculating_rates(grouped, text_col)
        grouped = self.normalize(grouped, scaling)
        grouped = self.reshape_df(grouped, cols_to_group)
        return grouped