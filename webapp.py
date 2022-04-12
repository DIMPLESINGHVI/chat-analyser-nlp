import streamlit as st
import data_preprocessor
import helper
import matplotlib.pyplot as plt
import seaborn as sns
st.sidebar.title("CHAT ANALYSER")

uploaded_file = st.sidebar.file_uploader("SELECT A FILE")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = data_preprocessor.preprocess(data)

    # fetching unique users and creating a user list

    users_list = df['USERS'].unique().tolist()
    users_list.remove('group_notif')
    users_list.sort()
    users_list.insert(0, "Overall")
    selected_user = st.sidebar.selectbox("SHOW ANALYSIS W.R.T", users_list)

    if st.sidebar.button("SHOW ANALYSIS"):
        st.title("STATISTICS")
        st.write("Whatsapp claims that nearly 55 billion messages are sent each day. "
                 "The average user spends 195 minutes per week on Whatsapp, and is a member of plenty of groups."
                 "So, with this data right within our reach, we can gain insights on the messages our phones "
                 "are forced to bear witness to. Let's look at some interesting statistics below: ")
        number_messages, number_words, number_media, number_links, number_emojis = helper.fetch_stats(selected_user, df)
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.header("TOTAL MESSAGES")
            st.title(number_messages)
        with col2:
            st.header("TOTAL WORDS")
            st.title(number_words)
        with col3:
            st.header("TOTAL MEDIA SHARED")
            st.title(number_media)
        with col4:
            st.header("TOTAL LINKS SHARED")
            st.title(number_links)
        with col5:
            st.header("TOTAL EMOJIS USED")
            st.title(number_emojis)

        # timeline
        st.title("MONTHLY TIMELINE")
        st.write("This graph shows us the activity of the participant(s) of the group based "
                 "on the month, year and number of messages sent.")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['TIME'], timeline['MESSAGES'], color='green')
        plt.xticks(rotation='vertical')
        plt.xlabel('MONTH & YEAR')
        plt.ylabel('NUMBER OF MESSAGES')
        st.pyplot(fig)

        # daily timeline
        st.title("DAILY TIMELINE")
        st.write("This graph shows us the activity of the participant(s) of the group based "
                 "on the date and number of messages sent.")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['ONLY_DATE'], daily_timeline['MESSAGES'], color='orange')
        plt.xticks(rotation='vertical')
        plt.ylabel('NUMBER OF MESSAGES')
        st.pyplot(fig)

        # activity map
        st.title("ACTIVITY MAP")
        st.write("These graphs show us the most active day and month of the participant(s).")
        col1, col2 = st.columns(2)

        with col1:
            st.header("MOST BUSY DAY")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values)
            plt.xticks(rotation='vertical')
            plt.xlabel('DAY')
            plt.ylabel('NUMBER OF MESSAGES')
            st.pyplot(fig)
        with col2:
            st.header("MOST BUSY MONTH")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='purple')
            plt.xticks(rotation='vertical')
            plt.xlabel('MONTH')
            plt.ylabel('NUMBER OF MESSAGES')
            st.pyplot(fig)

        st.title("WEEKLY ACTIVITY MAP")
        st.write("This heat map shows us the participant(s) interaction based on time period and day of the week. "
                 "The lighter shade represents most activity at that time of the day and "
                 "the darkest shade represents least activity at that time of the day.")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        # finding the busiest user works only in case of overall
        if selected_user == 'Overall':
            st.title('MOST BUSY USERS')
            # removing rows that contain 'group_notif' in users
            df = df[df.USERS != 'group_notif']
            x, new_df = helper.most_busy(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                st.write("This graph shows us the most active "
                         "participants based on the number of messages they've sent:")
                ax.bar(x.index, x.values)
                plt.xticks(rotation='vertical')
                plt.xlabel('USERNAME / USER_NUMBER')
                plt.ylabel('NUMBER OF MESSAGES')
                st.pyplot(fig)
            with col2:
                st.write("This data frame shows the most interactive user and the percentage of interaction:")
                st.dataframe(new_df)

        # creating wordCloud
        st.title("WORD CLOUD")
        st.write("A word cloud is basically an image composed of words "
                 "used in a particular text(s), in which the size of each word indicates it's frequency.")
        # removing rows that contain 'media omitted' in users
        df = df[df.MESSAGES != '<Media omitted>\n']
        df_wc = helper.creating_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # most common words
        col1, col2 = st.columns(2)
        with col1:
            st.title('MOST USED WORDS')
            st.write("This graph shows us the frequency with which a word has been used by the participant(s).")
            most_common_df = helper.most_common_words(selected_user, df)

            fig, ax = plt.subplots(figsize=(5, 5.5))

            ax.barh(most_common_df[0], most_common_df[1])
            plt.xticks(rotation='vertical')
            plt.xlabel('WORD COUNT')
            plt.ylabel('WORDS')
            st.pyplot(fig, figsize=(5, 5.5))

        # most common emojis
        with col2:
            st.title('MOST USED EMOJIS')
            st.write("This pie chart shows us the frequency with which an emoji has been used by the participant(s).")
            emoji_df = helper.most_common_emoji(selected_user, df)
            fig, ax = plt.subplots()
            ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f")
            st.pyplot(fig)

        sentiment, sentiment_score = helper.sentiment_analysis(selected_user, df)
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(5, 4))
            st.header("SENTIMENT ANALYSIS")
            st.write("Sentiment Analysis identifies the emotional tone behind a body of text. "
                     "This is also a popular way for organizations to determine and categorize opinions "
                     "about a product, service, or idea.")
            ax.bar(sentiment[0], sentiment[1])
            plt.xticks(rotation='vertical')
            plt.xlabel('TYPE OF EMOTION')
            st.pyplot(fig, figsize=(5, 4))
        with col2:
            st.header("OVERALL SENTIMENT")
            st.write("This pie chart shows the overall sentiment of the participant(s). "
                     "It is used to determine whether the texts sent by the participant(s) are "
                     "positive, negative, neutral or compound")
            fig, ax = plt.subplots()
            ax.pie(sentiment_score.values(), labels=sentiment_score.keys(), autopct="%0.2f")
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
