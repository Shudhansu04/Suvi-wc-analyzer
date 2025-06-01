# from transformers.utils.import_utils import candidates
from urlextract import  URLExtract
extract= URLExtract()
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import nltk
from nltk.corpus import stopwords
from gensim import corpora, models
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import tempfile

def fetch_starts(candidates,df):

    if candidates =='Full Analysis':
        #fetch total messages...........
        num_messages=df.shape[0]
        # total words typed.............
        words=[]
        for message in df['message']:
            words.extend(message.split())
        # total media messages..........
        total_media= df[df['message'] == '<Media omitted>\n'].shape[0]
    #total links shared.................
        links=[]
        for message in df['message']:
            links.extend(extract.find_urls(message))
        return num_messages,len(words),total_media,len(links)
    else:
        new_df = df[df['user'] == candidates]
        num_messages= new_df.shape[0]
        words = []
        for message in new_df['message']:
            words.extend(message.split())
        total_media =new_df[new_df['message'] == '<Media omitted>\n'].shape[0]
        links = []
        for message in new_df['message']:
            links.extend(extract.find_urls(message))
        return num_messages, len(words), total_media, len(links)

def fetch_active_users(df):
        x=df['user'].value_counts().head()
        df = round((df['user'].value_counts() / len(df)) * 100, 2).reset_index()
        df.columns = ['Name', 'Involved(%)']
        return x,df


def create_wordcloud(candidates, df):
    f = open('stopwords.txt', 'r')
    stopwords = f.read()
    if candidates != 'Full Analysis':
        df = df[df['user'] == candidates]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']
    def remove_stopwords(text):
        y=[]
        for word in text.lower().split():
            if word not in stopwords:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=800, height=400, background_color='white')
    temp['message']=temp['message'].apply(remove_stopwords)
    df_wc =wc.generate(temp['message'].str.cat(sep=""))
    return df_wc

def most_common_words(candidates,df):
    f=open('stopwords.txt','r')
    stopwords=f.read()
    if candidates !='Full Analysis':
        df=df[df['user'] == candidates]

    temp=df[df['user']!='group_notification']
    temp=temp[temp['message']!='<Media omitted>\n']

    words=[]
    for message in temp['message']:
        for word in message.lower().split():
            if word not in stopwords:
                words.append(word)

    repeated_df= pd.DataFrame(Counter(words).most_common(20))
    repeated_df.columns=['word','Frequency']
    return repeated_df

def emoji_calculator(candidates,df):
    if candidates !='Full Analysis':
        df=df[df['user'] == candidates]
    emojis=[]
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])
    emoji_df =pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    return emoji_df

def monthly_timeline(candidates,df):
    if candidates !='Full Analysis':
        df=df[df['user'] == candidates]

    timeline=df.groupby(['year','month_number','month']).count()['message'].reset_index()
    time=[]
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))
    timeline['time']=time
    return timeline
def daily_timeline(candidates,df):
    if candidates !='Full Analysis':
        df=df[df['user'] == candidates]
    daily=df.groupby('days').count()['message'].reset_index()
    return daily

def week_activity_map(candidates,df):
    if candidates !='Full Analysis':
        df=df[df['user'] == candidates]
    return df['day_name'].value_counts()

def month_activity_map(candidates,df):
    if candidates !='Full Analysis':
        df=df[df['user'] == candidates]

    return df['month'].value_counts()

def activity_map(candidates,df):
    if candidates !='Full Analysis':
        df=df[df['user'] == candidates]

    user_heatmap=df.pivot_table(index='day_name',columns='period',values='message',aggfunc='count').fillna(0)
    return user_heatmap

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'


nltk.download('stopwords')

def extract_messages(chat_lines):
    users = []
    messages = []
    for line in chat_lines:
        match = re.match(r'^\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}.*? - (.*?): (.+)$', line)
        if match:
            users.append(match.group(1))
            messages.append(match.group(2))
    return pd.DataFrame({'user': users, 'message': messages})


def preprocess_messages(messages, custom_stopwords):
    stop_words = set(stopwords.words('english')).union(custom_stopwords)
    clean_docs = []
    for msg in messages:
        msg = re.sub(r'[^a-zA-Z\s]', '', msg)
        words = msg.lower().split()
        words = [w for w in words if w not in stop_words and len(w) > 2]
        clean_docs.append(words)
    return clean_docs

def run_lda(clean_docs, num_topics=5):
    dictionary = corpora.Dictionary(clean_docs)
    corpus = [dictionary.doc2bow(text) for text in clean_docs]
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    return lda_model, corpus, dictionary


def get_user_data(candidates,df):
    if candidates != 'Full Analysis':
        df=df[df['user'] == candidates]
        return df
    else:
        new_df=df
        return new_df

def calculate_stats(user_df):
    messages_sent = user_df.shape[0]

    # Using above emoji_calculator function to calculate emoji.
    emoji_df =emoji_calculator('Full Analysis', user_df)

    # Average word length
    words = ' '.join(user_df['message']).split()
    avg_len = sum(len(word) for word in words) / max(len(words), 1)

    # Active hours
    active_hours = user_df['hour'].value_counts().sort_index()

    # Sentiment distribution
    if 'sentiment' in user_df.columns:
        sentiment_counts = user_df['sentiment'].value_counts()
    else:
        sentiment_counts = None

    return {
        'messages_sent': messages_sent,
        'emoji_df': emoji_df,
        'avg_word_length': avg_len,
        'active_hours': active_hours,
        'sentiment_counts': sentiment_counts
    }

def generate_pdf_report(df):
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(tmp_file.name, pagesize=letter)

    width, height = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawString(1 * inch, height - 1 * inch, "WhatsApp Chat Analysis Report")

    c.setFont("Helvetica", 12)
    c.drawString(1 * inch, height - 1.5 * inch, f"Total messages: {len(df)}")
    c.drawString(1 * inch, height - 1.75 * inch, f"Unique users: {df['user'].nunique()}")


    user_counts = df['user'].value_counts().head(5)
    c.drawString(1 * inch, height - 2.1 * inch, "Top 5 users by messages:")
    y = height - 2.35 * inch
    for user, count in user_counts.items():
        c.drawString(1.2 * inch, y, f"{user}: {count}")
        y -= 0.25 * inch

    c.save()
    return tmp_file.name
