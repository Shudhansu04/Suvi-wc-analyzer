import streamlit as st
import pyLDAvis.gensim_models
from matplotlib import pyplot as plt
from matplotlib import font_manager
import seaborn as sns
import functions
import preprocessor

st.set_page_config(page_title="WhatsApp Chat Analyzer", layout="wide")
st.sidebar.title("📱 WhatsApp Chat Analyzer")

# Upload chat and stopwords
uploaded_file = st.sidebar.file_uploader("Upload WhatsApp .txt file", type="txt")
with open("stopwords.txt", "r", encoding="utf-8") as f:
    stopword_file = f.read().splitlines()

search_query = st.sidebar.text_input("🔍 Search Messages")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Full Analysis")
    candidates = st.sidebar.selectbox("Group Members", user_list)

    #SEARCH
    if search_query:
        st.title("Search Results")
        if candidates != 'Full Analysis':
            filtered_df = df[(df['user'] == candidates) & (df['message'].str.contains(search_query, case=False))]
        else:
            filtered_df = df[df['message'].str.contains(search_query, case=False)]
        st.write(f"Found {filtered_df.shape[0]} messages")
        st.dataframe(filtered_df[['date', 'user', 'message']], height=300)

    #Sentiment Analysis
    if st.sidebar.checkbox("Show Sentiment Analysis"):
        st.title("📊 Sentiment Analysis")

        if uploaded_file is not None:
            raw_data = uploaded_file.read().decode("utf-8")
            df = preprocessor.preprocess(raw_data)

            if df.empty or 'message' not in df:
                st.warning("⚠️ No messages found. Please check your chat file format.")
            else:
                if candidates != 'Full Analysis':
                    filtered_df = df[df['user'] == candidates]
                    st.write(f"Analyzing messages of **{candidates}**")
                else:
                    filtered_df = df
                    st.write("Analyzing messages from **Whole Group Chat**")

                if filtered_df.empty:
                    st.warning("⚠️ No messages found for the selected user.")
                else:
                    # Apply sentiment analysis
                    filtered_df['sentiment'] = filtered_df['message'].apply(functions.analyze_sentiment)
                    sentiment_counts = filtered_df['sentiment'].value_counts()

                    # Display sentiment counts
                    st.subheader("📊 Sentiment Distribution")
                    st.write(sentiment_counts)
                    st.bar_chart(sentiment_counts)

                    # Pie chart
                    st.subheader("🥧 Sentiment Pie Chart")
                    fig, ax = plt.subplots()
                    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
                    ax.axis('equal')
                    st.pyplot(fig)

                    # Sentiment breakdown per user.
                    if candidates == 'Full Analysis':
                        st.subheader("👥 Individual User's Sentiment Breakdown")
                        user_sentiment = df.groupby(['user', 'sentiment']).size().unstack(fill_value=0)
                        st.dataframe(user_sentiment)

                    # Display messages with sentiment.
                    st.subheader("📝 Messages with Sentiment")
                    st.dataframe(filtered_df[['date', 'user', 'message', 'sentiment']])
        else:
            st.info("📁 Please upload a WhatsApp chat `.txt` file to begin.")

    #Topic Modeling with LDA
    if st.sidebar.checkbox("Show LDA Topic Modeling"):
        st.title("💬 LDA Topic Modeling")

        if uploaded_file is not None:
            chat_lines = data.splitlines()
            messages_df = functions.extract_messages(chat_lines)

            if df.empty or 'message' not in df:
                st.warning("⚠️ No messages found. Please check your chat file format.")
            else:
                if candidates != 'Full Analysis':
                    user_messages = df[df['user'] == candidates]['message'].tolist()
                    st.write(f"Showing topics for **{candidates}**")
                else:
                    user_messages = df['message'].tolist()
                    st.write("Showing topics for **Whole Group Chat**")

                clean_docs = functions.preprocess_messages(user_messages, set(stopword_file))
                if len(clean_docs) < 2:
                    st.warning("Not enough messages to extract topics.")
                else:
                    lda_model, corpus, dictionary = functions.run_lda(clean_docs)

                    # Display topics
                    st.subheader("Top Topics:")
                    for i, topic in lda_model.print_topics():
                        st.write(f"**Topic {i + 1}:** {topic}")

                    # Show visualization
                    st.subheader("📊 Topic Visualization")
                    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
                    st.components.v1.html(pyLDAvis.prepared_data_to_html(vis), height=800)
        else:
            st.info("Please upload a WhatsApp chat .txt file to begin.")

    if st.sidebar.checkbox("Show Chat Timeline View"):
        st.title("💬 Chat Timeline")

        if candidates != 'Full Analysis':
            chat_df = df[df['user'] == candidates]
        else:
            chat_df = df.copy()

        chat_df = chat_df.sort_values('date')

        # Assign pastel or dark colors
        pastel_colors = [
            "#a3d2ca", "#5eaaa8", "#056676", "#05445e", "#e3fdfd",
            "#9df3c4", "#f6dfeb", "#d6e6f2", "#ffb6b9", "#8ecae6"
        ]
        user_list = chat_df['user'].unique()
        user_colors = {user: pastel_colors[i % len(pastel_colors)] for i, user in enumerate(user_list)}

        # Using HTML for custom fonts
        st.markdown("""
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono&display=swap');

            .chat-bubble {
                font-family: 'Roboto Mono', monospace;
                padding: 12px 18px;
                border-radius: 16px;
                margin-bottom: 12px;
                line-height: 1.6;
                max-width: 70%;
                word-wrap: break-word;
                box-shadow: 1px 2px 6px rgba(0,0,0,0.15);
            }

            .user-left {
                margin-right: auto;
                background-color: #e3fdfd;
                color: #000;
            }

            .user-right {
                margin-left: auto;
                background-color: #d6e6f2;
                color: #000;
            }

            .timestamp {
                font-size: 0.75rem;
                color: #555;
                text-align: right;
                margin-top: 4px;
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown("---")

        for _, row in chat_df.iterrows():
            user = row['user']
            message = row['message']
            timestamp = row['date'].strftime('%d %b %Y %H:%M')

            align_class = "user-right" if candidates != 'Full Analysis' and user == candidates else "user-left"

            chat_bubble = f"""
            <div class='chat-bubble {align_class}'>
                <b>{user}</b><br>{message}
                 {timestamp} 
            </div>
            """
            st.markdown(chat_bubble, unsafe_allow_html=True)

    user_list = sorted(df['user'].unique())

    user1 = st.sidebar.selectbox("Select User 1 For Comparison", ['Full Analysis'] + user_list, key='user1')
    user2 = st.sidebar.selectbox("Select User 2 For Comparison ", ['Full Analysis'] + user_list, index=1 if len(user_list) > 1 else 0,
                                 key='user2')

    if st.sidebar.checkbox("Show User's Comparison"):
        st.title("📊 User's Comparison")
        user1_df = functions.get_user_data(user1, df)
        user2_df = functions.get_user_data(user2, df)


    stats1 = functions.calculate_stats(user1_df)
    stats2 = functions.calculate_stats(user2_df)

    col1, col2 = st.columns(2)

    with col1:
        st.header(f"Stats for {user1}")
        st.write(f"**Messages sent:** {stats1['messages_sent']}")
        st.write("**Emoji Usage:**")
        st.dataframe(stats1['emoji_df'])
        st.write(f"**Average Word Length:** {stats1['avg_word_length']:.2f}")
        st.write("**Active Hours:**")
        st.bar_chart(stats1['active_hours'])
        if stats1['sentiment_counts'] is not None:
            st.write("**Sentiment Distribution:**")
            st.bar_chart(stats1['sentiment_counts'])
        else:
            st.info("Sentiment data not available")

    with col2:
        st.header(f"Stats for {user2}")
        st.write(f"**Messages sent:** {stats2['messages_sent']}")
        st.write("**Emoji Usage:**")
        st.dataframe(stats2['emoji_df'])
        st.write(f"**Average Word Length:** {stats2['avg_word_length']:.2f}")
        st.write("**Active Hours:**")
        st.bar_chart(stats2['active_hours'])
        if stats2['sentiment_counts'] is not None:
            st.write("**Sentiment Distribution:**")
            st.bar_chart(stats2['sentiment_counts'])
        else:
            st.info("Sentiment data not available")

    if st.sidebar.button("Analyse"):

# Stats Area........
        num_messages, words, total_media, links = functions.fetch_starts(candidates, df)
        st.title("Statistics Of Chats")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.subheader("Total Conversations")
            st.title(num_messages)

        with col2:
            st.subheader("Typed Words")
            st.title(words)

        with col3:
            st.subheader("Total Media Shared")
            st.title(total_media)

        with col4:
            st.subheader("Links Shared")
            st.title(links)
# Monthly timleline......
        st.title('Monthly Timeline')
        timeline = functions.monthly_timeline(candidates, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='olive')
        plt.xticks(rotation=90)
        st.pyplot(fig)

# Daily timeline......
        st.title('Daily Timeline')
        daily = functions.daily_timeline(candidates, df)
        fig, ax = plt.subplots()
        ax.plot(daily['days'], daily['message'], color='olive')
        plt.xticks(rotation=90)
        st.pyplot(fig)

# Activity HeatMap.....
        st.title('Activity HeatMap')
        col1, col2 = st.columns(2)

        with col1:
          st.subheader("Most Busiest Days")
          busy_day = functions.week_activity_map(candidates, df)
          fig, ax = plt.subplots()
          ax.bar(busy_day.index, busy_day.values, color='olive')
          plt.xticks(rotation=90)
          st.pyplot(fig)

        with col2:
          st.subheader("Most Busiest Months")
          busy_month = functions.month_activity_map(candidates, df)
          fig, ax = plt.subplots()
          ax.bar(busy_month.index, busy_month.values, color='olive')
          plt.xticks(rotation=90)
          st.pyplot(fig)

        st.title('Weekly Activity HeatMap')
        user_heatmap=functions.activity_map(candidates, df)
        fig, ax = plt.subplots()
        ax=sns.heatmap(user_heatmap, cmap='RdYlBu')
        st.pyplot(fig)

# finding the most active person in the group(not for individual analysis)
        if candidates == 'Full Analysis':
            st.title("Most Active Users")
            x,rdf = functions.fetch_active_users(df)
            fig, ax = plt.subplots()
            col1, col2 = st.columns(2)

            with col1:
                ax.barh(x.index, x.values, color='green')
                st.pyplot(fig)

            with col2:
                st.dataframe(rdf, height=250)

# WordCloud Plot.
        st.title('WordCloud')
        wc_df = functions.create_wordcloud(candidates, df)
        fig, ax = plt.subplots()
        ax.imshow(wc_df)
        st.pyplot(fig)

# Most Used Words.
        st.title('Most Repeated 20 Words')
        repeated_df = functions.most_common_words(candidates, df)
        fig, ax = plt.subplots()
        ax.barh(repeated_df['word'], repeated_df['Frequency'], color='olive')
        st.subheader('Bar Chart')
        st.pyplot(fig)
        st.subheader('Tabular Format')
        st.dataframe(repeated_df, height=300)

# Emoji Analysis.......
        st.title('Emoji Used')

        emoji_df = functions.emoji_calculator(candidates, df)
        emoji_df.columns = ['Emoji', 'Frequency']

        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(emoji_df, height=350)
        
        with col2:
            sizes = emoji_df['Frequency'].head(20)
            labels = emoji_df['Emoji'].head(20)
            
            fig, ax = plt.subplots()
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.set_title("Top 20 Emoji Usage")
            
            st.pyplot(fig)

    if st.sidebar.button("Download Report"):
        pdf_path = functions.generate_pdf_report(df)

        with open(pdf_path, "rb") as pdf_file:
            pdf_doc = pdf_file.read()
        st.download_button(label="Download PDF", data=pdf_doc, file_name="chat_analysis_report.pdf",
                           mime='application/pdf')





