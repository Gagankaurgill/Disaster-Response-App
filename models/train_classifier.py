# import libraries
import sys
import os
import numpy as np
import pandas as pd
import re
import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pickle
import warnings
warnings.filterwarnings("ignore")

def load_data(database_filename):
    """Function to load cleaned data from app database as dataframe.
    
    Args: 
        database_filename (str): saved database table
    Returns:
        X (series):             messages to classify.
        Y (dataframe):          dataframe containing message categories.
        category_names:         column list of columns in Y dataframe
        
    """
   #create connection
    engine = create_engine('sqlite:///' + database_filename)    
    
    #read data from database
    table_name = os.path.basename(database_filename).split('.')[0]
    df = pd.read_sql_table(table_name,con=engine)
    
    #assign target and feature data to X and Y
    X = df['message']    
    Y = df.drop(['id','message','genre'], axis = 1)
    category_names = Y.columns

    return X,Y,category_names


def tokenize(text):
    """Function converts raw messages into tokens, cleans the tokens and removes
        stopwords.
    
    Args:
        text(str): raw message data to be classified.
    Returns:
        clean_tokens(list): cleaned list of tokens(words).
        
    """
    #check for urls in the text
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex,text)
    
    for url in detected_urls:
        text = text.replace(url,'urlplaceholder')
    
    #remove punctuation from the text
    text = re.sub(r"[^a-zA-Z0-9]"," ",text)
    
    #tokenization
    tokens = word_tokenize(text)
    
    # remove stop words
    tokens = [tok for tok in tokens if tok not in stopwords.words("english")]
    
    #lemmatization
    lem = WordNetLemmatizer()
    
    clean_tokens = []
    for token in tokens:
        clean_token = lem.lemmatize(token.lower().strip())
        clean_tokens.append(clean_token)
    
    return clean_tokens

def build_model():
    """Builds the machine learning pipeline containing transformers and 
        a final MultiOutput estimator.
        
    Args: 
        None.
    Returns:
        pipeline: defined machine learning pipeline.
    
    """

    pipeline = Pipeline([
                ('vect',CountVectorizer(tokenizer=tokenize)),
                ('tfidf',TfidfTransformer()),
                ('clf',MultiOutputClassifier(RandomForestClassifier(n_jobs=-1)))
             ])

    #   Pipeline Hyperparamenter tuning - remove '#' to include other parameters as you like. 
    #   Training could take several minutes or hours depending on your device and hyperparameter choice
    parameters = {
        'clf__estimator__n_estimators':[100,200],
        'clf__estimator__max_depth' :[5]
        }

    optimizer = GridSearchCV(pipeline, param_grid=parameters)

    return optimizer

def evaluate_model(model, X_test, y_test, category_names):
    """Evaluates the performance of trained model.
    
    Args:
        model:          trained model.
        X_test:         test data for prediction.
        y_test:         test classification data for evaluating model predictions.
        category_names: category names.
        
    Returns:
        prints out metric scores - Precision, Recall and Accuracy.
        
    """
    #   predict classes for X_test
    prediction = model.predict(X_test)

    #   print out model precision, recall and accuracy
    print(classification_report(y_test, prediction, target_names=category_names))


def save_model(model, model_filepath):
    """ This function saves the pipeline to local disk """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """ Run the script and handle user arguments.
       
    Args: 
        None
        
    Returns:
        None
       
    """
    if len(sys.argv) == 3:
        database_filename, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filename))
        X, y,category_names = load_data(database_filename)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state = 101)
        
        print('Building model...')
        model = build_model()


        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the name of table of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py DisasterResponse classifier.pkl')


if __name__ == '__main__':
    main()