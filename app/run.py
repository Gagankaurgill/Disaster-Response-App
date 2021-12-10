import json
import plotly
import pandas as pd
import re
import joblib
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine


app = Flask(__name__)

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

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # message counts by genre
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # message counts by category
    category_counts = df[df.columns[3:]].sum().sort_values(ascending=False)
    category_name = list(category_counts.index)
    
    # categorized count of messages based on genre
    direct_cat_count = {}
    news_cat_count = {}
    social_cat_count = {}
    directdf = df[df['genre'] == 'direct']
    newsdf = df[df['genre'] == 'news']
    socialdf = df[df['genre'] == 'social']
    for col in df.columns[3:]:
        x = directdf[directdf[col] == True][col].count()
        y = newsdf[newsdf[col] == True][col].count()
        z = socialdf[socialdf[col] == True][col].count()
        direct_cat_count[col] = x 
        news_cat_count[col] = y 
        social_cat_count[col] = z 
    trace_direct = {
    'x' : list(direct_cat_count.keys()),
    'y' : list(direct_cat_count.values()),
    'name' : "Direct",
    'type' : "bar",
    'opacity' :0.5
    };
    trace_news = {
        'x' : list(news_cat_count.keys()),
        'y' : list(news_cat_count.values()),
        'name' : "News",
        'type' : "bar",
        'opacity' :0.5
    };
    trace_social = {
        'x' : list(social_cat_count.keys()),
        'y' : list(social_cat_count.values()),
        'name' : "Social",
        'type' : "bar",
        'opacity' :0.5
    };
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'template': "seaborn"
            }
        },
        {
            'data': [
                Bar(
                    x=category_name,
                    y=category_counts
                )],
            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                },
                'template': "seaborn"
            }
        },
        {
            'data': [trace_direct,trace_news,trace_social],

            'layout': {
                'title': 'Distribution of Message based on Genres ',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                },
                'barmode':"stack",
                'showlegend':True,
                'template': "seaborn"
            }
        }
        
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
