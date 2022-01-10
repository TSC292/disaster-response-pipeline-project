import pandas as pd
from sqlalchemy import create_engine
import sys

def load_data(messages_filepath, categories_filepath):
    """
    Loads data from two csv files. Merge operation depends on files containing 'id' column
    
    Parameters
    ----------
    messages_filepath : str
        csv file expected to contain messages data
    categories_filepath : str
        csv file expected to contain categories data
        
    Returns
    -------
    df : DataFrame
        pandas Dataframe with data from CSVs merged
    """
    # read two csv files as DataFrames then merge
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on='id')
    return df

def clean_data(df):
    """
    Custom clean operation for data loaded from two csv files. It has numerous dependencies on the content.
    
    Parameters
    ----------
    df : DataFrame
        pandas DataFrame expected to contain particular content
        
    Returns
    -------
    df : DataFrame
        pandas Dataframe with cleaning operations having been performed
    """
    """
    Custom clean operation for data loaded from two csv files. It has numerous dependencies on the content.
    
    Parameters
    ----------
    df : DataFrame
        pandas DataFrame expected to contain particular content
        
    Returns
    -------
    df : DataFrame
        pandas Dataframe with cleaning operations having been performed
    """
    # clean category column by splitting on ';'
    categories = df['categories'].str.split(";",expand=True)
    # retrieve column names from first row by removing the binary values from the strings
    columns = categories.iloc[0,].str.replace('(-\d)','',regex=True)
    # set column names to those extracted from the first row
    categories.columns=columns
    # remove the text preceding the binary 1/0 value then convert this binary value to integer
    categories = categories.apply(lambda x: x.str.replace('(\w+-)','',regex=True).astype(int))
    
    # replace the categories column in the original DataFrame with the cleaned categories dataframe
    df.drop('categories',axis=1,inplace=True)
    df = pd.concat([df,categories],axis=1)
    
    # identify rows where the category columns contain values which are not binary (0,1) and remove them
    notbinary = categories.apply(lambda x: ~x.isin([0,1])).any(axis=1)
    df = df.loc[~notbinary]
    
    # identify duplicates where entire rows are duplicated and keep only one of each set
    df = df.loc[~df.duplicated(keep='first')]
    # identify, out of remaining rows, those with duplicate IDs and exclude all of them
    df = df.loc[~df['id'].duplicated(keep=False)]
    return df

def save_data(df, database_filepath, table_name='DR_table'):
    """
    Save a pandas DataFrame to a specified database
    
    Parameters
    ----------
    df : DataFrame
        pandas DataFrame to be saved in the database
    database_filepath : str
        name of database. Expecting it to end with .db
    table_name : str, optional
        table name to use. Default 'DR_table'
    """
    # create an sqlite engine
    engine = create_engine(f'sqlite:///{database_filepath}')
    # write records to the sql database using the sqlite engine
    df.to_sql(table_name, engine, index=False,if_exists='replace')

    
def main():
    """
    Run all operations of: load csv data, clean data and save to a database.
    Depends on command line arguments for the parameters: messages_filepath, categories_filepath, database_filepath
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()