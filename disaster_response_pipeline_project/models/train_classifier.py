# import libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
from joblib import dump

def load_data(database_filepath,table_name = 'DR_table'):
    """
    Load the disaster response data from the specified database
    Parameters
    ----------
    database_filepath : string
        file path to the database to be read from
    table_name : str, optional
        table name to use. Default 'DR_table'
    """
    # start sqlite engine for reading database
    engine = create_engine(f'sqlite:///{database_filepath}')
    # read table in the database as pandas DataFrame
    df = pd.read_sql_table(table_name,engine)
    X = df['message']
    y = df.iloc[:,4:]
    category_names = list(y.columns)
    return X, y, category_names


def tokenize(text):
    """
    Custom Tokenizer for cleaning and tokenizing strings
    Parameters
    ----------
    text : string
        Block of text to be tokenized
    """
    # convert to lower case, to letters and numbers only, remove stopwords, tokenize
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]"," ",text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [WordNetLemmatizer().lemmatize(w) for w in tokens]
    return tokens

def build_model():
    """
    ML pipeline with Gridsearch capability to return a classification model
    which was fitted with the custom tokenizer/vocabulary and with tuned hyperparameters.
    Parameters
    ----------
    None
    """
    # create pipeline with a TF-IDF created as X data using the custom tokenizer
    # max features is limited to reduce computational load
    # using a RandomForest classifer model
    pipeline = Pipeline([
    ('vect',CountVectorizer(tokenizer=tokenize, max_features=10000)),
    ('tfidf',TfidfTransformer()),
    ('clf',RandomForestClassifier())
    ])
    # tuning hyperparameters: trialling hyperparameters of two default value calculators for max features
    # also trialling three different values for number of trees (estimators)
    param_grid = {'clf__max_features':['sqrt', 'log2'],
                 'clf__n_estimators':[10, 100, 1000]}
    return GridSearchCV(pipeline, param_grid, cv=3)

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Print an evaluation of the fitted model using test data
    Parameters
    ----------
    model: sklearn estimator
        contains fitted model
    X_test: DataFrame
        X test data
    Y_test: DataFrame
        Y test data
    category_names: list
        list of category names to show results for
    """
    # predict y values on test data
    Y_pred = model.predict(X_test)
    # print performance metrics for the whole set of predictions
    print('accuracy',accuracy_score(Y_test, Y_pred))
    print('----micro average----')
    print('f1 score',f1_score(Y_test, Y_pred, average='micro'))
    print('precision',precision_score(Y_test, Y_pred, average='micro'))
    print('recall',recall_score(Y_test, Y_pred, average='micro'))
    # record and print performance metrics for each category
    results = pd.DataFrame(columns=['category','accuracy','f1','precision','recall'])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(len(category_names)):
            category = category_names[i]
            accuracy = accuracy_score(Y_test.iloc[:,i].to_numpy(), Y_pred[:,i])
            f1 = f1_score(Y_test.iloc[:,i].to_numpy(), Y_pred[:,i])
            precision = precision_score(Y_test.iloc[:,i].to_numpy(), Y_pred[:,i])
            recall = recall_score(Y_test.iloc[:,i].to_numpy(), Y_pred[:,i])
            new_row = pd.DataFrame(columns=['category','accuracy','f1','precision','recall'],data=[[category,accuracy,f1,precision,recall]])
            results = results.append(new_row,ignore_index=True)
    print('metrics for all categories:')
    print(results)

def save_model(model, model_filepath):
    """
    Save the fitted model to a pkl file
    Parameters
    ----------
    model: sklearn estimator
        contains fitted model
    model_filepath: string
        path to save location for pkl file
    """
    dump(model, model_filepath)

def main():
    """
    Function to activate load data, build model, train model, evaluate model, save model
    Parameters
    ----------
    None
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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