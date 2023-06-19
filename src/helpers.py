import os
import sys
import math
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm2
from scipy.sparse import save_npz, load_npz


def get_project_dir():
    """
    TODO
    """
    return os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

def save_as_csv(path, filename, data, mode, index=False, header=True):
    """
    TODO
    """
    data.to_csv(os.path.join(path, filename), mode=mode, index=index, header=header)


def load_csv(path, filename, columns, chunksize, n_rows, condition=None, prefix=None,
             mapper=None, mapper_kwargs={}, column_to_process=None, ascending=False,
             ignore_index=True):
    """
    TODO
    """
    ### creating iterator
    iterator = tqdm2(pd.read_csv(os.path.join(path, filename), chunksize=chunksize),
                     total=math.ceil((n_rows - 1) / chunksize), desc=f'CHUNKS{str(prefix)}')
    iterator.set_postfix({'n_rows': 0})
    length = 0

    ### creating empty dataset
    data = pd.DataFrame(columns=columns)
    if condition in ['get_unique','count_unique']:
        data = []

    ### looping through chunks
    for df in iterator:
        if type(condition) == dict:
            df = df[df[condition['column']]==condition['value']][columns]
            if mapper is not None:
                df[column_to_process] = df[column_to_process].apply(lambda x: mapper(x, **mapper_kwargs))
            length += len(df)
            data = pd.concat([data, df], ignore_index=ignore_index)
        elif condition == 'count_unique':
            df = df[column_to_process].value_counts()
            if mapper is not None:
                df.index = df.index.map(lambda x: mapper(x, **mapper_kwargs))
                df = df.sort_index() \
                        .reset_index() \
                        .groupby(column_to_process) \
                        .sum() \
                        .reset_index()
            length += len(df)
            data.append(df)
        elif condition == 'get_unique':
            df = list(df[column_to_process].unique())
            if mapper is not None:
                df = list(map(lambda x: mapper(x, **mapper_kwargs), df))
            data = list(set(data + df))
            length = len(data)
        else:
            if mapper is not None:
                df[column_to_process] = df[column_to_process].replace(mapper)
            df = df[columns]
            length += len(df)
            data = pd.concat([data, df], ignore_index=ignore_index)

        iterator.set_postfix({'n_rows': length})

    if condition == 'count_unique':
        data = pd.concat(data).groupby(column_to_process).sum().sort_values(by=column_to_process,
                                                                            ascending=ascending)
    return data



def save_as_txt(path, filename, data, mode):
    """
    TODO
    """
    with open(os.path.join(path, filename), mode) as f:
        for line in data:
            f.write(f'{line}\n')



def save_as_npz(path, filename, data):
    """
    TODO
    """
    save_npz(file=os.path.join(path, filename), matrix=data.tocsr(), compressed=True)


def load_npz_sparse(path, filename):
    """
    TODO
    """
    return load_npz(os.path.join(path, filename))


def read_txt(path, filename):
    """
    TODO
    """
    with open(os.path.join(path, filename)) as f:
        data = f.readlines()
    data = list(map(lambda x: x[:-1], data))
    return data



def get_size(object, metric='Mb'):
    """
    TODO
    """
    if metric == 'Mb':
        return sys.getsizeof(object) / (1e+6)
    if metric == 'Kb':
        return sys.getsizeof(object) / (1e+3)



def csr_vappend(a, b):
    """
    TODO
    """
    a.data = np.hstack((a.data,b.data))
    a.indices = np.hstack((a.indices,b.indices))
    a.indptr = np.hstack((a.indptr,(b.indptr + a.nnz)[1:]))
    a._shape = (a.shape[0]+b.shape[0],b.shape[1])
    return a


def str_to_datetime(input_date, format='%Y/%m/%d'):
    """
    TODO
    """
    try:
        output_date = '/'.join(input_date.split('/')[:len(format.split('/'))])
        output_date = datetime.datetime.strptime(output_date, format)
    except:
        output_date = None
    return output_date


def smooth(scalars, weight=0.9, reverse=False):
    """
    TODO
    """
    if reverse:
        scalars = list(np.array(scalars)[::-1])
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    if reverse:
        smoothed = list(np.array(smoothed)[::-1])
    return smoothed