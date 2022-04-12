import pandas as pd
import emojis
import string
import re
import emoji
from wordcloud import WordCloud
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from urlextract import URLExtract
extract = URLExtract()


def fetch_stats(selected_user, df):

    if selected_user != 'Overall':
        df = df[df['USERS'] == selected_user]
    # fetching number of messages
    num_messages = df.shape[0]
    # fetching number of words
    words = []
    for message in df['MESSAGES']:
        words.extend(message.split())

    # fetching number of media messages
    numb_media = df[df['MESSAGES'] == '<Media omitted>\n'].shape[0]

    # fetching number of links shared
    links = []
    for message in df['MESSAGES']:
        links.extend(extract.find_urls(message))

    # fetching number of emojis
    emo = []
    for message in df['MESSAGES']:
        emo.extend(emojis.get(message))

    return num_messages, len(words), numb_media, len(links), len(emo)


def most_busy(df):
    x = df['USERS'].value_counts().head()
    df = round((df['USERS'].value_counts()/df.shape[0])*100, 2).reset_index().rename(columns={'index': 'NAME',
                                                                                              'USERS': 'PERCENTAGE'})
    return x, df


def creating_wordcloud(selected_user, df):

    f = open('stop_word_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['USERS'] == selected_user]

    temp = df[df['USERS'] != 'group_notification']
    temp = temp[temp['MESSAGES'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    temp['MESSAGES'] = temp['MESSAGES'].apply(remove_stop_words)
    df_wc = wc.generate(temp['MESSAGES'].str.cat(sep=" "))
    return df_wc


def most_common_words(selected_user, df):

    f = open('stop_word_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['USERS'] == selected_user]

    temp = df[df['USERS'] != 'group_notif']
    temp = temp[temp['MESSAGES'] != '<Media omitted>\n']

    emo = []
    for message in df['MESSAGES']:
        emo.extend(emojis.get(message))

    mess = []
    for message in temp['MESSAGES']:
        mess.append(message)

    length = len(mess)

    text = ""
    for i in range(0, length):
        text = mess[i] + " " + text

    # converting to lowercase
    lower_case = text.lower()

    # Removing punctuations
    cleaned_text = lower_case.translate(str.maketrans('_', ' ', string.punctuation))
    # removing any integer values that might be present in the text
    cleaned_text = re.sub(r'[0-9]+', '', cleaned_text)
    tokenized_words = cleaned_text.split()

    # removing stop words
    final_words = [word for word in tokenized_words if word not in stop_words]
    emo = []
    for message in df['MESSAGES']:
        emo.extend(emojis.get(message))

    without_emo = [word for word in final_words if word not in emo]
    most_common_df = pd.DataFrame(Counter(without_emo).most_common(10))

    return most_common_df


def most_common_emoji(selected_user, df):

    if selected_user != 'Overall':
        df = df[df['USERS'] == selected_user]
    emo = []
    for message in df['MESSAGES']:
        emo.extend(emojis.get(message))
    most_common_emoj = pd.DataFrame(Counter(emo).most_common(10))

    return most_common_emoj


def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['USERS'] == selected_user]
    timeline = df.groupby(['YEAR', 'MONTH_NUM', 'MONTH']).count()['MESSAGES'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['MONTH'][i] + "-" + str(timeline['YEAR'][i]))

    timeline['TIME'] = time
    return timeline


def daily_timeline(selected_user, df):

    if selected_user != 'Overall':
        df = df[df['USERS'] == selected_user]

    daily_timeline = df.groupby('ONLY_DATE').count()['MESSAGES'].reset_index()

    return daily_timeline


def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['USERS'] == selected_user]

    return df['DAY_NAME'].value_counts()


def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['USERS'] == selected_user]

    return df['MONTH'].value_counts()


def activity_heatmap(selected_user, df):

    if selected_user != 'Overall':
        df = df[df['USERS'] == selected_user]

    user_heatmap = df.pivot_table(index='DAY_NAME', columns='PERIOD', values='MESSAGES', aggfunc='count').fillna(0)

    return user_heatmap


def sentiment_analysis(selected_user, df):

    if selected_user != 'Overall':
        df = df[df['USERS'] == selected_user]

    temp = df[df['USERS'] != 'group_notif']
    temp = temp[temp['MESSAGES'] != '<Media omitted>\n']
    mess = []
    for message in temp['MESSAGES']:
        mess.append(message)

    length = len(mess)

    text = ""
    for i in range(0, length):
        text = mess[i] + " " + text

    # converting to lowercase
    lower_case = text.lower()
    # extracting the emoji meaning
    lower_case = emoji.demojize(lower_case, delimiters=(" ", " "))
    # replacing punctuations generated from emoji conversion
    lower_case = lower_case.replace("_", " ")

    # Removing punctuations
    cleaned_text = lower_case.translate(str.maketrans('_', ' ', string.punctuation))
    tokenized_words = cleaned_text.split()

    f = open('stop_word_hinglish.txt', 'r')
    stop_words = f.read()
    # removing stop words
    final_words = [word for word in tokenized_words if word not in stop_words]
    # creating the emotion list
    emotion_list = []
    with open('emotions.txt', 'r') as file:
        for line in file:
            clear_line = line.replace('\n', '').replace(',', '').replace("'", '').strip()
            word, emotion = clear_line.split(':')
            if word in final_words:
                emotion_list.append(emotion)

    emotion = pd.DataFrame(Counter(emotion_list).most_common(5))

    score = SentimentIntensityAnalyzer().polarity_scores(cleaned_text)
    return emotion, score
