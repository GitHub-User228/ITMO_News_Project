import os
import csv
import math
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from src.preprocessors import init_preprocess
from src.aggregators import count_rows


class BaseSplitter:
    """
    This class can be used to split input Basis DataFrame by selected columns.
    For each selected column:
        - a new Identifier DataFrame is formed, where only unique values of the column with their IDs are stored
        - values in the column of Basis DataFrame are replaced with the corresponding IDs
    """

    @staticmethod
    def grouping(data, col):
        """
        Extracts unique values of column "col" from Basis DataFrame "data" with the IDs from column "id" grouped
        in a list for each unique value

        Parameters
        ----------
        data : pd.DataFrame
            Basis DataFrame

        col : str
            Name of a column for which the unique values with rows' IDs grouped in a list should be collected

        Returns
        -------
        field : pd.DataFrame
            Identifier DataFrame where:
                "index" - set of unique values of column 'col' from DataFrame "data"
                "post_id" - corresponding IDs of rows (combined in list)

        """
        field = data[[col, 'id']].groupby(col)['id'] \
                                 .apply(list) \
                                 .rename_axis('text') \
                                 .to_frame() \
                                 .rename(columns={'id': 'post_id'})
        return field

    @staticmethod
    def init_preprocessing(field, do_preprocessing, indexer1='', indexer2=''):
        """
        Performs initial preprocessing (if needed) of index in Identifier DataFrame created by function 'grouping'
        The idea is to reduces the size of Identifier DataFrame by reducing the number of unique indexes
        after preprocessing it

        Parameters
        ----------
        field : pd.DataFrame
            Identifier dataset (created by function 'grouping')

        do_preprocessing : bool
            Whether to do preprocessing or not

        indexer1 : int or str
            Optional indexer used to define IDs for rows of Identifier DataFrame

        indexer2 : int or str
            Optional indexer used to define IDs for rows of Identifier DataFrame

        Returns
        -------
        field : pd.DataFrame
            Identifier DataFrame where:
                "text" - set of unique initially preprocessed values of column 'col' from Basis DataFrame
                "id" - corresponding ID
                "post_id" - corresponding IDs of rows from Basis DataFrame (combined in list)

        """
        if do_preprocessing:
            field.index = list(tqdm(map(lambda x: init_preprocess(x), field.index),
                                    leave=False,
                                    total=len(field)))

            field = field.rename_axis('text') \
                         .groupby('text')['post_id'] \
                         .apply(lambda lists: [k for list_ in lists for k in list_]) \
                         .to_frame()
        field['id'] = [f'ID{indexer1}_{indexer2}_{k}' for k in range(len(field))]
        return field


    @staticmethod
    def join(data, field, col):
        """
        Joins Basis DataFrame 'data' and Identifier DataFrame 'field' on the unique values from
        column 'col' of 'data'

        Parameters
        ----------
        data : pd.DataFrame
            Basis DataFrame

        field : pd.DataFrame
            Identifier DataFrame where:
                "text" - set of unique initially preprocessed values of column 'col' from Basis DataFrame "data"
                "id" - corresponding ID
                "post_id" - corresponding IDs of rows from Basis DataFrame "data" (combined in list)


        col : str
            Name of a column for which an Identifier Dataset "field" is formed

        Returns
        -------
        data : pd.DataFrame
            Basis DataFrame with the column 'col' replaced by 'col_id'
            (IDs from Identifier DataFrame 'field' are used)

        field : pd.DataFrame
            Identifier DataFrame where:
                "text" - set of unique (optionally initially preprocessed) values of column 'col'
                         from Basis DataFrame "data"
                "id" - corresponding ID

        """
        data = data.drop(col, axis=1) \
            .join(field.explode('post_id') \
                  .set_index('post_id') \
                  .rename(columns={'id': f'{col}_id'}) \
                  .rename_axis('id'),
                  on='id', how='left')
        field = field.drop('post_id', axis=1)
        return data, field


    def split(self,
              data,
              cols=['title', 'summary', 'link'],
              all_cols=None,
              do_preprocessing=[True, True, False],
              bar_name='Set_0',
              indexer1=0,
              indexer2=0):
        """
        Main function, which splits input Basis DataFrame 'data' by selected 'cols' columns.
        For each selected column:
            - a new Identifier DataFrame is formed, where only unique values of the column with their IDs are stored
            - values in the column of Basis DataFrame are replaced with the corresponding IDs

        Parameters
        ----------
        data : pd.DataFrame
            Basis dataset, which should be split

        cols : list of strings
            Names of columns for which an Identifier DataFrame should be formed

        all_cols : list of strings
            Names of columns which should be kept in Basis DataFrame 'data'
            (in case there are redundant columns in 'data')

        do_preprocessing : list of bools
            Whether to perform initial preprocessing for selected columns in 'cols'

        bar_name : str
            Name of tqdm progress bar used to monitor the split process

        indexer1 : int or str
            Optional indexer used to define IDs

        indexer2 : int or str
            Optional indexer used to define IDs


        Returns
        -------
        data : pd.DataFrame
            Modified Basis DataFrame, where columns from 'cols' are replaced with corresponding identifier columns

        sets : dict of pd.DataFrame(s)
            A set of Identifier DataFrames created for each column in 'cols'
            Each Identifier DataFrame has the following columns:
                "text" - set of unique (optionally initially preprocessed) values of column 'col'
                         from Basis DataFrame "data"
                "id" - corresponding ID

        counter : numpy array
            An array with the length of each Identifier DataFrame

        """
        ### omitting redundant columns in Basis DataFrame
        if all_cols is not None:
            data = data[all_cols]

        ### defining 'id' column to index rows in Basis DataFrame
        data['id'] = [f'ID{indexer1}_{indexer2}_{k}' for k in range(len(data))]

        ### creating empty dict to store created Identifier DataFrames for each column in 'cols'
        sets = {}

        ### creating tqdm iterator
        iterator = tqdm(enumerate(cols), total=len(cols), desc=bar_name)
        counter = dict(zip(cols, [0 for _ in cols]))
        iterator.set_postfix(counter)

        ### looping through each column in cols
        for it, col in iterator:
            sets[col] = self.grouping(data, col)
            sets[col] = self.init_preprocessing(sets[col], do_preprocessing[it],
                                                indexer1, indexer2)
            data, sets[col] = self.join(data, sets[col], col)

            counter[col] = len(sets[col])
            iterator.set_postfix(counter)

        counter = np.array(list(counter.values()))

        return data.drop('id', axis=1), sets, counter



def main_splitter(path,
                  filename,
                  chunksize,
                  cols=['title', 'summary', 'link'],
                  all_cols=[],
                  do_preprocessing=[True, True, False],
                  indexer1=0,
                  n_rows=None):
    """
    Function, which splits Basis DataFrame stored in 'path' by selected 'cols' columns, when 'data'
    is very big to be split whole at once (there is also option available to split DataFrame in one go) -
    with the help of 'chunksize' parameter of pd.read_csv each chunk is split separately and sequentially.
    Function returns a result in the same Basis DataFrame and Identifier DataFrames, but indexing is
    different for each chunk. Function "split" from "BaseSplitter" class is used as a splitting function.
    How the splitting process is performed is shown in descriptions inside BaseSplitter class.

    Parameters
    ----------
    path : str
        Path to Basis Dataset in csv format which should be split

    filename : str
        Filename of Basis Dataset in csv format which should be split

    chunksize : int or None
        Size of each chunk of data if int (Basis DataFrame is split in one go in case of None)

    cols : list of strings
        Names of columns for which an Identifier DataFrame should be formed

    all_cols : list of strings
        Names of columns which should be kept in Basis Dataset (in case there are redundant columns)

    do_preprocessing : list of bools
        Whether to perform initial preprocessing for selected columns in 'cols'

    indexer1 : int or str
        Optional indexer used to define IDs

    Returns
    -------
    dfs : pd.DataFrame
        Modified Basis DataFrame, where columns from 'cols' are replaced with corresponding identifier columns
        In case of fragmentation of Basis DataFrame, values in identifier columns have the following format:
            - 'ID{indexer1}_{indexer2}_{id}
        where
            - 'indexer1' defines general index which the same for each chunk
            - 'indexer2' defines ID of a chunk
            - 'id' defines ID of a unique value from a column from chunk

    sets : dict of pd.DataFrame(s)
        A set of Identifier DataFrames created for each column in 'cols'
        In case of fragmentation of original dataset, each 'id' column have the following format:
            - 'ID{indexer1}_{indexer2}_{id}
        where
            - 'indexer1' defines general index which the same for each chunk
            - 'indexer2' defines ID of a chunk
            - 'id' defines ID of a unique value from a column from chunk

    """
    ### case when Basis DataFrame must be fragmented into chunks and split operation is performed for each chunk
    if chunksize is not None:
        ### counting the number of rows
        if n_rows is None:
            n_rows = count_rows(path=path, filename=filename)

        ### defining empty DataFrame to store modified Basis DataFrame after splitting each chunk
        dfs = pd.DataFrame(columns=[k for k in all_cols if k not in cols])

        ### defining a dict of empty DataFrames to store Identifier DataFrames after splitting each chunk
        sets = dict(zip(cols, [pd.DataFrame(columns=['id']).rename_axis('text') for _ in cols]))

        ### creating tqdm iterator over chunks
        iterator = tqdm(pd.read_csv(os.path.join(path, filename), chunksize=chunksize),
                        total=math.ceil((n_rows-1)/chunksize))
        counter = np.array([0 for _ in cols])
        iterator.set_postfix(dict(zip(cols, counter)))

        ### looping over chunks
        for it, df in enumerate(iterator):
            ### splitting chunk
            dfs_, sets_, counter_ = BaseSplitter().split(data=df,
                                                   cols=cols,
                                                   all_cols=all_cols,
                                                    do_preprocessing=do_preprocessing,
                                                    bar_name=f'Set_{it}',
                                                    indexer1=indexer1,
                                                    indexer2=it)
            counter += counter_
            iterator.set_postfix(dict(zip(cols, counter)))

            ### storing results
            for col in cols:
                sets[col] = pd.concat([sets[col], sets_[col]])
            dfs = pd.concat([dfs, dfs_])

    ### case when data can be split whole at once
    else:
        data = pd.read_csv(os.path.join(path, filename))
        dfs, sets, _ = BaseSplitter().split(data=data,
                                            cols=cols,
                                            all_cols=all_cols,
                                            do_preprocessing=do_preprocessing,
                                            bar_name=f'Set_{0}',
                                            indexer1=indexer1,
                                            indexer2=0)

    return dfs, sets


class PostSplitter:
    """
    This class is used to merge the result of function 'main_splitter' in case the Basis Dataset
    has been fragmented into chunks
    """

    @staticmethod
    def regroup_field(field, indexer1):
        """
        This function groups Identifier DataFrame 'field' by 'text' in order to
        keep only unique values.

        Parameters
        ----------
        field : pd.DataFrame
            Identifier DataFrame to be simplified

        indexer1 : int or str
            Optional indexer used to define IDs of rows after grouping

        Returns
        -------
        field : pd.DataFrame
            Modified Identifier DataFrame with the following columns:
                - text : unique values from corresponding column from initial Identifier DataFrame
                - ids : IDs from column 'id' from initial Identifier DataFrame grouped in list
                - id : new ID for each row with the following format:
                            - 'ID{indexer1}_{id}
                       where
                            - 'indexer1' defines general index
                            - 'id' defines ID of a unique value from 'text' column

        """
        field = field.groupby('text')['id'] \
            .apply(list) \
            .to_frame() \
            .rename(columns={'id': 'ids'}) \
            .sort_index()
        field['id'] = [f'ID{indexer1}_{k}' for k in range(len(field))]
        return field


    @staticmethod
    def join(field, data, col):
        """
        Joins Basis DataFrame 'data' and Identifier DataFrame 'field' on "old" IDS from field,
        and replacing old IDS with the new ones from 'field'

        Parameters
        ----------
        data : pd.DataFrame
            Basis DataFrame created by 'main_splitter' function with used fragmentation

        field : pd.DataFrame
            Identifier DataFrame created by 'regroup_field' function

        col : str
            Name of a column for which the Identifier DataFrame 'field' is formed

        Returns
        -------
        data : pd.DataFrame
            Modified Basis DataFrame, where rows with the same value of column 'col' have the same ID

        field : pd.DataFrame
            Modified Identifier DataFrame with the following columns:
                "text" - set of unique (optionally initially preprocessed) values of column 'col'
                         from Basis DataFrame "data"
                "id" - corresponding ID

        """
        data = data.join(field.explode('ids') \
                         .set_index('ids') \
                         .rename(columns={'id': f'{col}_id_new'}) \
                         .rename_axis(f'{col}_id'),
                         on=f'{col}_id', how='left') \
            .drop(f'{col}_id', axis=1) \
            .rename(columns={f'{col}_id_new': f'{col}_id'})
        field = field.drop('ids', axis=1)
        return field, data

    def merge(self, field, data, col, indexer1=''):
        """
        Main function, which merges the result of function 'main_splitter' in case the Basis DataFrame 'data'
        has been fragmented into chunks

        Parameters
        ----------
        data : pd.DataFrame
            Basis DataFrame created by 'main_splitter' function with used fragmentation

        field : pd.DataFrame
            Identifier DataFrame created by 'main_splitter' function with used fragmentation

        col : str
            Name of a column for which the Identifier DataFrame 'field' is formed

        indexer1 : int or str
            Optional indexer used to define IDs of rows after grouping

        Returns
        -------
        data : pd.DataFrame
            Modified Basis DataFrame, where rows with the same value of column 'col' have the same IDs

        field : pd.DataFrame
            Modified Identifier DataFrame with the following columns:
                "text" - set of unique (optionally initially preprocessed) values of column 'col'
                         from Basis DataFrame "data"
                "id" - corresponding ID

        """
        field = self.regroup_field(field, indexer1)
        field, data = self.join(field, data, col)
        return field, data


class NewDataMergerer:
    """
    This class is used to merge new dataset (which was split whole at once by main_splitter) with a main dataset
    """
    @staticmethod
    def find_matches(path, filename, field_new, col):
        """
        Finds matches between rows from New Identifier DataFrame 'field_new' and corresponding DataFrame stored
        in "path" with the "filename" filename

        Parameters
        ----------
        path : str
            Path to Identifier DataFrame from the main dataset

        filename : str
            Filename of Identifier DataFrame from the main dataset

        field_new : pd.DataFrame
            New Identifier DataFrame

        col : str
            Name of a column for which the New Identifier DataFrame 'field_new' is formed

        Returns
        -------
        data : pd.DataFrame
            Modified New Basis DataFrame, where IDs in identifier columns and IDs from corresponding
            columns in the Basis DataFrame from main dataset have the same indexation

        field : pd.DataFrame
            Modified New Identifier DataFrame with the following columns:
                "text" - set of unique (optionally initially preprocessed) values of column 'col'
                         from New Basis DataFrame "data"
                "id" - corresponding original ID
                "id_real" - corresponding ID having with the same indexation with main dataset
                "is_new" - whether a row is new or not (comparing to main dataset)

        """
        ### creating new column which will collect matches of text between field_new and corresponding main file
        field_new['id_real'] = None
        field_new['is_new'] = True

        ### reading field's main file
        with open(os.path.join(path, filename), 'r', encoding='utf8', newline='') as fp:
            reader = csv.reader(fp)

            ### tqdm iterator
            iterator = tqdm(reader, desc=col)
            counter = 0
            iterator.set_postfix({'Number of matches': counter})

            ### loop through rows of field's main file
            for it, (text_, id_) in enumerate(iterator):
                ### case when text_ is in 'field_new's text column
                if text_ in field_new.index:
                    field_new.loc[text_, 'id_real'] = id_
                    field_new.loc[text_, 'is_new'] = False
                    counter += 1
                    iterator.set_postfix({'Number of matches': counter})

            field_new.loc[field_new['is_new'], 'id_real'] = [f'ID_{k}' for k in range(it, it + sum(field_new['is_new']))]

        return field_new


    @staticmethod
    def join(data, field, col):
        """
        Joins New Basis DataFrame 'data' and New Identifier DataFrame 'field' on "old" IDS from field,
        and replacing old IDS with the new ones from 'field'

        Parameters
        ----------
        data : pd.DataFrame
            New Basis DataFrame created by 'main_splitter' function

        field : pd.DataFrame
            New Identifier DataFrame created by 'regroup_field' function

        col : str
            Name of a column for which the 'field' DataFrame is formed

        Returns
        -------
        data : pd.DataFrame
            Modified New Basis DataFrame, where IDs in identifier columns and IDs from corresponding
            columns in the Basis DataFrame from main dataset have the same indexation

        field : pd.DataFrame
            Dataset simular to input 'field', but column 'ids' is omitted and rows which occurs
            in the corresponding DataFrame from main dataset are also omitted

        """
        data = data.join(field[['id', 'id_real']].set_index('id') \
                              .rename_axis(f'{col}_id'),
                         on=f'{col}_id', how='left') \
                    .drop(f'{col}_id', axis=1) \
                    .rename(columns={'id_real': f'{col}_id'})
        field = field.drop('id', axis=1) \
                     .rename(columns={'id_real': 'id'}) \
                     .loc[field['is_new'], ['id']]
        return data, field

    def merge(self, path, filenames, data_new, fields_new):
        """
        Main function, which merges new dataset (which was split whole at once by main_splitter) with a main dataset

        Parameters
        ----------
        path : str
            Path to directory with the main dataset

        filenames : list of strings
            Names of files of Identifier DataFrames from main dataset corresponding to Identifier
            DataFrames stored in fields_new dictionary

        data_new : pd.DataFrame
            New Basis DataFrame which was split whole at once by 'main_splitter' function

        fields_new : dict of pd.DataFrame(s)
            New Identifier DataFrames which were created by 'main_splitter' function

        Returns
        -------
        data_new : pd.DataFrame
            Modified New Basis DataFrame 'data_new', where IDs in identifier columns and IDs from corresponding
            columns in the Basis DataFrame from main dataset have the same indexation

        fields_new : dict of pd.DataFrame(s)
            Modified 'fields_new', where IDs from each identifier DataFrame and IDs from corresponding
            DataFrame in the main dataset have the same indexation

        """
        iterator = tqdm(fields_new.keys(), total=len(fields_new))

        for it, key in enumerate(iterator):

            fields_new[key] = self.find_matches(path=path,
                                                filename=filenames[it],
                                                field_new=fields_new[key],
                                                col=key)

            data_new, fields_new[key] = self.join(data=data_new,
                                                  field=fields_new[key],
                                                  col=key)

        return data_new, fields_new