import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import io

nltk.download('vader_lexicon')

class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def transform_scale(self, score):
        return 5 * score + 5  # Convert the sentiment score from -1 to 1 scale to 0 to 10 scale

    def calculate_overall_sentiment(self, reviews):
        compound_scores = [self.sia.polarity_scores(review)["compound"] for review in reviews]
        overall_sentiment = sum(compound_scores) / len(compound_scores)
        return self.transform_scale(overall_sentiment)

    def analyze_sentiment(self, reviews):
        sentiments = [{'compound': self.transform_scale(self.sia.polarity_scores(review)["compound"]),
                       'pos': self.sia.polarity_scores(review)["pos"],
                       'neu': self.sia.polarity_scores(review)["neu"],
                       'neg': self.sia.polarity_scores(review)["neg"]}
                      for review in reviews]
        return sentiments

    def analyze_periodic_sentiment(self, reviews, period):
        period_reviews = [' '.join(reviews[i:i + period]) for i in range(0, len(reviews), period)]
        return self.analyze_sentiment(period_reviews)

    def interpret_sentiment(self, sentiments):
        avg_sentiment = sum([sentiment['compound'] for sentiment in sentiments]) / len(sentiments)
        if avg_sentiment >= 6.5:
            description = "Excellent progress, keep up the good work!"
        elif avg_sentiment >= 6.2:
            description = "Good progress, continue to work hard!"
        else:
            description = "Needs improvement, stay motivated and keep trying!"

        trend = "No change"
        if len(sentiments) > 1:
            first_half_avg = sum([sentiment['compound'] for sentiment in sentiments[:len(sentiments)//2]]) / (len(sentiments)//2)
            second_half_avg = sum([sentiment['compound'] for sentiment in sentiments[len(sentiments)//2:]]) / (len(sentiments)//2)
            if second_half_avg > first_half_avg:
                trend = "Improving"
            elif second_half_avg < first_half_avg:
                trend = "Declining"

        return description, trend


# Streamlit UI
st.title("Student Review Sentiment Analysis")

# Upload CSV file
csv_file = st.file_uploader("Upload your CSV file")

if csv_file:
    df = pd.read_csv(csv_file)
    st.write(df.head())  # Debug statement to check the loaded data

    students = df["Student"].tolist()
    selected_student = st.selectbox("Select a student:", ["All Students"] + students)
    review_period = st.selectbox("Review Period:", [1, 4])

    if selected_student != "All Students":
        student_data = df[df["Student"] == selected_student].squeeze()
        st.write(student_data.head())  # Debug statement to check the filtered data

        reviews = student_data[1:].tolist()
        analyzer = SentimentAnalyzer()

        if review_period == 1:
            sentiments = analyzer.analyze_sentiment(reviews)
        else:
            sentiments = analyzer.analyze_periodic_sentiment(reviews, review_period)
        st.write(sentiments)  # Debug statement to check the sentiments

        overall_sentiment = analyzer.calculate_overall_sentiment(reviews)
        st.subheader(f"Overall Sentiment for {selected_student}: {overall_sentiment:.2f}")
        st.subheader("Sentiment Analysis")

        # Plotting sentiment
        weeks = list(range(1, len(sentiments) + 1))
        sentiment_scores = [sentiment['compound'] for sentiment in sentiments]
        pos_scores = [sentiment['pos'] for sentiment in sentiments]
        neu_scores = [sentiment['neu'] for sentiment in sentiments]
        neg_scores = [sentiment['neg'] for sentiment in sentiments]

        fig, ax = plt.subplots()
        ax.plot(weeks, sentiment_scores, label="Overall", color="blue")
        ax.fill_between(weeks, sentiment_scores, color="blue", alpha=0.1)
        ax.plot(weeks, pos_scores, label="Positive", color="green")
        ax.plot(weeks, neu_scores, label="Neutral", color="gray")
        ax.plot(weeks, neg_scores, label="Negative", color="red")

        ax.set_xlabel('Week')
        ax.set_ylabel('Sentiment Score')
        ax.set_title(f'Sentiment Analysis for {selected_student}')
        ax.legend()
        st.pyplot(fig)

        description, trend = analyzer.interpret_sentiment(sentiments)
        st.subheader("Progress Description")
        st.write(f"Sentiment Trend: {trend}")
        st.write(f"Description: {description}")

        # Breakdown of analysis
        st.subheader("Breakdown of Analysis")
        breakdown_df = pd.DataFrame(sentiments, index=list(range(1, len(sentiments) + 1)))
        st.write(breakdown_df)

        # Add a button to download student performance data
        if st.button("Download Student Performance Data"):
            output_df = pd.DataFrame({'Student': students})
            output_df['Overall Sentiment'] = [analyzer.calculate_overall_sentiment(df[df["Student"] == student].squeeze()[1:].tolist()) for student in students]
            output_df['Sentiment Description'] = [analyzer.interpret_sentiment(analyzer.analyze_sentiment(df[df["Student"] == student].squeeze()[1:].tolist()))[0] for student in students]

            # Create a streamlit.io object to store the CSV data
            output_data = io.StringIO()
            output_df.to_csv(output_data, index=False)
            output_data.seek(0)

            # Display a download button for the CSV file
            st.download_button(
                label="Click here to download student performance data",
                data=output_data.getvalue().encode('utf-8'),
                file_name="student_performance.csv",
                key='download_button'
            )
