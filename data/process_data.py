# Libraries import
import sys
import nltk
nltk.download(['punkt', 'wordnet'])
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC


# Function to load both input csv datasets, messages and categories
def load_data(messages_filepath, categories_filepath):
    # Load messages and categories files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge on a single dataframe
    df = messages.merge(categories, on='id', how='outer')
    
    return df
   

# Function to clean the data
def clean_data(df):
    
    # Split 'categories' into separate category columns
    categories = df['categories'].str.split(';', 0, expand=True)

    # select its first row and use it to extract a list of new column names
    row = categories.iloc[0, :]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = categories[column].apply(lambda x: int(x))

    # Replace 'categories' column in 'df' with new category columns
    df.drop('categories', axis=1, inplace=True)
    categories['id'] = df['id']
    df = df.merge(categories, on='id')

    # Remove duplicates
    df.drop_duplicates(subset=None, keep='first', inplace=True)

    return df 


# Saving the data into a SQLite database
def save_data(df, database_filepath):
    
    #Save the clean dataset into an sqlite database
    table_name = 'disasters_table'
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df.to_sql(tablename, engine, index=False)


    
def main():
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
