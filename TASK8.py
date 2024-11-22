import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
import nltk

# Download the VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

# Load the dataset
apps_df = pd.read_csv(r"C:\Users\nandi\OneDrive\Desktop\oasis\TASK8\apps.csv")  # Replace with your file path
reviews_df = pd.read_csv(r"C:\Users\nandi\OneDrive\Desktop\oasis\TASK8\user_reviews.csv")  # Replace with your file path

# Data Preparation: Cleaning
apps_df.drop_duplicates(inplace=True)
apps_df.dropna(inplace=True)

# Convert Reviews column to numeric
apps_df['Reviews'] = pd.to_numeric(apps_df['Reviews'], errors='coerce')

# Clean and convert Installs column
apps_df['Installs'] = apps_df['Installs'].str.replace('[+,]', '', regex=True).astype(float)

# Clean and convert Price column
apps_df['Price'] = apps_df['Price'].str.replace('$', '', regex=True).astype(float, errors='ignore')

# Clean and convert Size column
# First, replace non-string values like NaN with an empty string and handle any unexpected types
apps_df['Size'] = apps_df['Size'].fillna('').astype(str)

# Remove any text "Varies with device" and handle sizes with "k" or "M"
apps_df['Size'] = apps_df['Size'].replace("Varies with device", np.nan)

# Extract numeric value and multiplier (k or M) for Size column
size_multiplier = apps_df['Size'].str.extract(r'([\d\.]+)([kM]+)', expand=True)

# Convert size to numeric, stripping the suffixes, and apply multiplier
apps_df['Size'] = apps_df['Size'].str.replace(r'[kM]+$', '', regex=True)  # Remove any 'k' or 'M'
apps_df['Size'] = pd.to_numeric(apps_df['Size'], errors='coerce')  # Convert to numeric

# Now apply the correct multiplier (1 for no multiplier, 10^3 for k, 10^6 for M)
apps_df['Size'] = apps_df['Size'] * size_multiplier[1].replace({'k': 10**3, 'M': 10**6}).fillna(1).astype(float)

# Convert Last Updated column to datetime
apps_df['Last Updated'] = pd.to_datetime(apps_df['Last Updated'], errors='coerce')

# Category Exploration
plt.figure(figsize=(12, 6))
sns.countplot(data=apps_df, y='Category', order=apps_df['Category'].value_counts().index, palette='viridis')
plt.title("App Distribution Across Categories")
plt.xlabel("Number of Apps")
plt.ylabel("Category")
plt.tight_layout()
plt.show()

# Metrics Analysis
# Ratings Distribution
plt.figure(figsize=(8, 5))
sns.histplot(apps_df['Rating'], bins=20, kde=True, color='skyblue')
plt.title("App Ratings Distribution")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.show()

# Size vs. Rating
plt.figure(figsize=(10, 6))
sns.scatterplot(data=apps_df, x='Size', y='Rating', hue='Category', alpha=0.7)
plt.title("Size vs Rating Across Categories")
plt.xlabel("Size (in MB)")
plt.ylabel("Rating")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Sentiment Analysis
reviews_df.dropna(subset=['Translated_Review'], inplace=True)
sid = SentimentIntensityAnalyzer()
reviews_df['Sentiment_Score'] = reviews_df['Translated_Review'].apply(lambda x: sid.polarity_scores(x)['compound'])
reviews_df['Sentiment'] = reviews_df['Sentiment_Score'].apply(lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral')

# Sentiment Distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=reviews_df, x='Sentiment', palette='coolwarm')
plt.title("User Review Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# Word Cloud for User Reviews
text = " ".join(review for review in reviews_df['Translated_Review'])
wordcloud = WordCloud(background_color="white", max_words=500, contour_width=3, contour_color='steelblue').generate(text)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Interactive Visualization (Pair Plot)
sns.pairplot(apps_df[['Rating', 'Reviews', 'Installs', 'Price']], diag_kind='kde', palette='viridis')
plt.suptitle("Pair Plot for Key Metrics", y=1.02)
plt.show()

# Label Encoding for Categorical Data (Optional)
le = LabelEncoder()
apps_df['Category'] = le.fit_transform(apps_df['Category'])

# Insights from Visualization
print("Data Preparation and Analysis Complete!")
