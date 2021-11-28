# import libraries
import sys
import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_data,categories_data):
    # load messages dataset
    messages = pd.read_csv(messages_data)
    messages.head()

    # load categories dataset
    categories = pd.read_csv(categories_data)
    categories.head()

    # merge datasets
    df = pd.merge(messages,categories, how='inner',on='id')
    return df

def clean_data(df):

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";",expand=True)

    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x:x.split('-')[0]).values

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x : x.split('-')[1])
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        # convert any values different than 0 and 1 to 1
        categories[column].loc[(categories[column] != 0) & (categories[column] != 1)] = 1


    # drop the original categories column from `df`
    df = df.drop(['categories','original'],axis = 1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis = 1)

    # drop duplicates
    df.drop_duplicates(inplace = True)
    
    #drop NA data
    df.dropna(axis = 0,how = 'all',subset = category_colnames,inplace = True)

    return df

def save_data(df,database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    table_name = os.path.basename(database_filepath).split('.')[0]
    df.to_sql(table_name, engine, index=False,if_exists = 'replace')
    print('table:',table_name)

def main():
    if len(sys.argv) == 4:

        messages_data, categories_data, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_data, categories_data))
        df = load_data(messages_data, categories_data)

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