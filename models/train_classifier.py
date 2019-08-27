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


# Loading the data from the SQLite dataset previously created 
def load_data(database_filepath):
    
    table_name = 'disasters_table'
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(table_name, con=engine)
    
    X = df['message']  # Input variables
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)  # Targets (multiple outputs)
    
    return X, Y


# Tokenizing the messages in order to obtain separate words and clean to obtain them in the same format
def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


# Build the ML model
def build_model():
    
    # Pipeline building
    pipeline = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    
    # Parameters for the Grid Search
    parameters = {'tfidf__use_idf': (True, False), 
                  'clf__estimator__n_estimators': [50, 100], 
                  'clf__estimator__min_samples_split': [2, 3]} 
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


# Evaluation of the model
def evaluate_model(model, X_test, Y_test, category_names):
    
    y_pred = model.predict(X_test)
    
    for i, column in enumerate(Y_test):
        print(column)
        print(classification_report(Y_test[column], y_pred[:, i]))
        
    return model


# Save the model
def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(cv, f)


def main():
    if len(sys.argv) == 3:
        
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
