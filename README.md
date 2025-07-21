# Suvi WC Analyzer

Suvi WC Analyzer is a powerful Streamlit-based web application for analyzing and visualizing WhatsApp group or individual chat data. The tool provides insights into user activity, sentiment, word usage patterns, emoji trends, and conversation topics using natural language processing (NLP) and machine learning (ML) techniques.

## Features

- Upload and parse WhatsApp chat files (`.txt` format).
- Generate user-wise and overall chat statistics.
- Sentiment analysis (positive, negative, neutral) for entire chats or individual participants.
- Timeline visualizations (monthly, daily, and weekly trends).
- Activity heatmaps and peak usage times.
- Word cloud generation and most common words.
- Emoji usage analysis.
- LDA-based topic modeling to extract key conversation topics.
- Support for custom stopwords file.
- Downloadable reports and visualizations.

## Live Demo

[Click here to try the application](https://suvi-wc-analyzer-sgbpypassvbdg2c8bddaqh.streamlit.app/)

## Technologies Used

- **Frontend & App Framework:** Streamlit
- **Programming Language:** Python
- **Libraries & Tools:**
  - pandas, numpy
  - matplotlib, seaborn, wordcloud
  - NLTK, spaCy (for NLP)
  - scikit-learn (for sentiment classification)
  - gensim (for LDA topic modeling)
  - dateutil (for flexible timestamp parsing)

## File Structure
```
Suvi-wc-analyzer/
│
├── functions.py # Core logic for statistics, emoji and NLP
├── preprocessor.py # Data cleaning and chat parsing logic
├── user_interface.py # Streamlit UI components
├── stopwords.txt # Custom stopwords for NLP
├── requirements.txt # Python dependencies
├── .gitignore
├── README.md
└── .devcontainer/ # Optional configuration for containerized environments

```


## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Shudhansu04/Suvi-wc-analyzer.git
   cd Suvi-wc-analyzer
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   pip install -r requirements.txt
   streamlit run user_interface.py
   
   ```

