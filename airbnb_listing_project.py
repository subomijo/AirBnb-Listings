#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install imbalanced-learn


# In[2]:


pip install xgboost


# In[79]:


#importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from imblearn.pipeline import Pipeline
import xgboost as xgb
import time


# In[4]:


#loading the dataset into a pandas dataframe
airbnb_data = pd.read_csv('airbnb_berlin.csv')

#showing the first few rows
airbnb_data.head()


# In[5]:


#programmatic assessment of dataset
airbnb_data.info()


# ### DATA CLEANING AND PREPROCESSING

# In[6]:


#checking for missing values
missing_values = airbnb_data.isnull().sum()

#checking data types
data_types = airbnb_data.dtypes

#summary of missing values and data types
missing_values_summary = pd.DataFrame({'Missing Values': missing_values, 'Data Type': data_types})
missing_values_summary.sort_values(by='Missing Values', ascending=False)


# In[7]:


#checking for duplicate rows
duplicates = airbnb_data.duplicated().sum()
duplicates


# In[8]:


#dropping columns that might not be relevant for analyzing customer satisfaction, more columns might be added upon further investigation of the dataset
columns_to_drop = ['Host URL', 'Listing URL', 'Country', 'Country Code']
airbnb_data = airbnb_data.drop(columns=columns_to_drop)


# In[9]:


#converting 'Price' to numeric, handling non-numeric characters
airbnb_data['Price'] = pd.to_numeric(airbnb_data['Price'], errors='coerce')


# In[10]:


#ensuring that prices are non-negative
airbnb_data = airbnb_data[airbnb_data['Price'] >= 0]


# In[11]:


#checking the count and descriptive statistics of the 'Price' column
num_rows_price = airbnb_data['Price'].shape[0]
price_description  = airbnb_data['Price'].describe()


# In[12]:


num_rows_price, price_description


# In[13]:


#contextual analysis of zero-priced listings
#filtering the dataset for zero-priced listings
zero_priced_listings = airbnb_data[airbnb_data['Price'] == 0]
rating_columns = ['Overall Rating', 'Accuracy Rating', 'Cleanliness Rating', 
                  'Checkin Rating', 'Communication Rating', 'Location Rating', 
                  'Value Rating']

#analyzing common characteristics of these listings
zero_price_common_chars = {
    'Number of Zero Priced Listings': zero_priced_listings.shape[0],
    'Unique Hosts': zero_priced_listings['Host ID'].nunique(),
    'Common Neighborhoods': zero_priced_listings['neighbourhood'].value_counts().head(5),
    'Common Property Types': zero_priced_listings['Property Type'].value_counts().head(5),
    'Average Ratings': zero_priced_listings[rating_columns].mean()
}
zero_price_common_chars


# We can infer that the presence of zero-priced listings are not errors, and they represent special promotions or peculiar hosting situations. The high average ratings across all rating categories indicate positive guest experiences, which could be influenced by the zero pricing. We have chosen to include this in the analysis based on these findings as they represent comprehensive analysis of customer satisfaction, and are also unique yet valid scenarios within the Airbnb market in Berlin.

# In[14]:


#dropping missing values in the ratings column
airbnb_data = airbnb_data.dropna(subset=rating_columns)


# In[15]:


#checking the summary stats of the ratings column
airbnb_data[rating_columns].describe()


# In[16]:


#creating histograms for further analysis of the ratings columns
for column in rating_columns:
    airbnb_data[column].hist()
    plt.title(column)
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.show()


# In[17]:


#defining percentiles for categorizing ratings
high_rating_threshold = airbnb_data[rating_columns].quantile(0.75)
medium_rating_threshold = airbnb_data[rating_columns].quantile(0.25)

#creating a function to categorize ratings
def categorize_ratings(row, high_threshold, medium_threshold):
    if row > high_threshold:
        return 'High'
    elif row < medium_threshold:
        return 'Low'
    else:
        return 'Medium'

#applying the categorization to each ratings column
categorized_ratings = airbnb_data[rating_columns].apply(lambda x: x.apply(categorize_ratings, 
                                                                                  args=(high_rating_threshold[x.name], 
                                                                                        medium_rating_threshold[x.name])))

#displaying the first few rows of the categorized ratings
categorized_ratings.head()


# In[18]:


medium_rating_threshold


# From these observed values we can see how our contextual analysis of the zero priced listings are further validated as the ratings are genrally on the higher side

# In[19]:


#downloading necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# In[20]:


nltk.download('omw-1.4')


# In[21]:


#text data preprocessing
def preprocess_text(text):
    #checking if the text is not a string (e.g., NaN or None)
    if not isinstance(text, str):
        return ""

    #removing special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)

    #normalizing text to lowercase
    text = text.lower()

    #tokenize text
    tokens = word_tokenize(text)

    #remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]

    #lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    return ' '.join(lemmatized_text)


# In[22]:


#applying text preprocessing to the Comments column
airbnb_data['Processed_Comments'] = airbnb_data['Comments'].apply(preprocess_text)
airbnb_data['Processed_Comments'].head(10)


# In[23]:


airbnb_data.info()


# In[24]:


columns_with_missing_values = ['Property Type', 'Room Type', 'Host Response Time', 
                               'Is Superhost', 'Is Exact Location', 'Processed_Comments']

#dropping rows with missing values in the specified columns
airbnb_data = airbnb_data.dropna(subset=columns_with_missing_values)


# We will use one hot encoder for the property type and room type columns because they do not have too many categories and are also nominal while for the neighborhood column we would be using the label encoder beacause of the amount of categories in the variable

# In[25]:


#one hot encoder
one_hot_encoder = OneHotEncoder(sparse=False)
one_hot_encoded = one_hot_encoder.fit_transform(airbnb_data[['Property Type', 'Room Type','Host Response Rate', 'Is Superhost']])

#converting one-hot encoded data to a DataFrame
columns_one_hot_encoded = one_hot_encoder.get_feature_names_out(['Property Type', 'Room Type', 'Host Response Rate', 'Is Superhost'])
df_one_hot_encoded = pd.DataFrame(one_hot_encoded, columns=columns_one_hot_encoded)

#combining the encoded columns with the original dataset
airbnb_data = airbnb_data.join(df_one_hot_encoded)


# In[26]:


#label encoder
label_encoder = LabelEncoder()
airbnb_data['neighbourhood_encoded'] = label_encoder.fit_transform(airbnb_data['neighbourhood'])


# In[27]:


#combining the one-hot encoded columns with the original dataset
airbnb_data_encoded = airbnb_data.join(df_one_hot_encoded, lsuffix='_left', rsuffix='_right')

#display the first few rows of the new dataset
airbnb_data_encoded.head()


# In[28]:


#saving progress
#airbnb_data.to_csv('airbnb_data_cleaned.csv', index=False)
#airbnb_data_encoded.to_csv('airbnb_data_encoded.csv', index=False)


# ### EXPLORATORY DATA ANALYSIS

# In[29]:


#selecting only numerical columns for correlation analysis
numerical_columns = [
    'Overall Rating', 'Accuracy Rating', 'Cleanliness Rating', 'Checkin Rating', 
    'Communication Rating', 'Location Rating', 'Value Rating', 'Price', 
    'Latitude', 'Longitude', 'Reviews'
]

#creating a dataframe with only the numerical columns
numerical_data = airbnb_data_encoded[numerical_columns]

#calculating the correlation matrix
correlation_matrix = numerical_data.corr()

#plotting the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix of Selected Numerical Variables")
plt.show()


# From the graphic we can see that Overall Rating has strong positive correlations with all other rating variables (Accuracy, Cleanliness, Checkin, Communication, and Value), indicating that higher scores in these categories generally contribute to a higher overall rating.
# 
# Price shows a very low correlation with all the rating variables, suggesting that the cost of the accommodation is not strongly linked to how guests rate their experience in these aspects.
# 
# Geographical Coordinates (Latitude and Longitude) do not show significant correlations with the rating variables, which might imply that the exact location within Berlin is not a major factor in determining the ratings.
# 
# We would like to delve further into these findings

# In[30]:


#selecting the columns with strong correlations to 'Overall Rating' for further analysis
#we consider strong correlation to be above 0.5
strong_correlation_columns = correlation_matrix['Overall Rating'][correlation_matrix['Overall Rating'] > 0.5].index.tolist()

#filtering the data for these columns
strong_correlation_data = numerical_data[strong_correlation_columns]

#creating a pairplot to visualize the relationships between 'Overall Rating' and other strongly correlated variables
sns.pairplot(strong_correlation_data)
plt.suptitle("Pairplot of Variables Strongly Correlated with Overall Rating", y=1.02)
plt.show()


# From the pairplot, we can observe that there are clear linear relationships between 'Overall Rating' and other ratings like 'Value Rating', 'Cleanliness Rating', etc. This indicates that improvements in these specific areas are likely to lead to higher overall satisfaction. For this dataset the distribution of each rating variable shows a concentration of high ratings, suggesting that most customers tend to give high satisfaction scores. 

# In[31]:


#box plot for 'Overall Rating' across different 'Room Type's
plt.figure(figsize=(12, 6))
sns.boxplot(x='Room Type', y='Overall Rating', data=airbnb_data_encoded)
plt.title('Overall Rating Across Different Room Types')
plt.show()


# In[32]:


#calculaing summary statistics for 'Overall Rating' grouped by 'Room Type' for a clearer  understanding of the visualization
summary_stats_by_property_type = airbnb_data_encoded.groupby('Room Type')['Overall Rating'].describe()

#displaying the summary statistics
summary_stats_by_property_type


# From the above plots and summary statistics we can see that the private room is the most preffered room type as it has the highest median rating. It also has the loswest standard deviation which shows a more consistent set of ratings. The shared room has the lowest median rating showing people might not be overly delighted about sharing their privacy and it also has the highest standard deviation which shows a greater variability in guest satisfaction.

# In[33]:


#since 'Property Type' has many unique values, we'll focus on the most common types for clarity in visualization
#filtering the top 10 most common property types
top_property_types = airbnb_data_encoded['Property Type'].value_counts().head(10).index

#creating a new dataframe with only the top property types
filtered_data_property_type = airbnb_data_encoded[airbnb_data_encoded['Property Type'].isin(top_property_types)]

#box plot for 'Overall Rating' across the top property types
plt.figure(figsize=(15, 8))
sns.boxplot(x='Property Type', y='Overall Rating', data=filtered_data_property_type)
plt.xticks(rotation=45)
plt.title('Overall Rating Across Different Property Types')
plt.show()


# In[34]:


#calculating summary statistics for 'Overall Rating' grouped by 'Property Type'
summary_stats_by_property_type = airbnb_data_encoded.groupby('Property Type')['Overall Rating'].describe()

#displaying the summary statistics
summary_stats_by_property_type


# The property types 'Apartment', 'Loft', 'Condominium', 'House', and 'Serviced Apartment' stand out for their high guest satisfaction, with median ratings consistently above 95, indicating a strong preference among guests for these types of accommodations. Apartments, the most common type, show a high median rating of 96, reflecting their popularity and guest satisfaction. Lofts and Condominiums follow closely, also displaying high satisfaction levels. Houses and Serviced Apartments, while slightly less common, maintain similarly high ratings, suggesting that guests value the space, amenities, and experience these properties offer.
# 
# Conversely, Hostels, show a notably lower median rating of 91, indicating varied guest experiences, possibly due to the shared nature of the accommodations. Guest Suites and Guesthouses present a more consistent and positive experience, as indicated by their higher median ratings around 96. Townhouses also fare well in guest satisfaction, aligning with the trend favoring residential-style accommodations. Bed and Breakfasts, with their unique charm, maintain a respectable median rating, though with slightly more variability in guest experiences.
# 
# This highlights a clear preference for private, well-equipped accommodations, with traditional options like Hostels and Hotels receiving more varied responses, underscoring the importance of property type in shaping guest experiences in Airbnb stays.

# In[35]:


#since 'neighbourhood' column has many unique values, we'll focus on the most common neighbourhoods for clarity in visualization
#filtering the top 10 most common neighbourhoods
top_neighbourhoods = airbnb_data_encoded['neighbourhood'].value_counts().head(10).index

#creating a new dataframe with only the top neighbourhoods
filtered_data_neighbourhood = airbnb_data_encoded[airbnb_data_encoded['neighbourhood'].isin(top_neighbourhoods)]

#creating a box plot for 'Overall Rating' across the top neighbourhoods
plt.figure(figsize=(15, 8))
sns.boxplot(x='neighbourhood', y='Overall Rating', data=filtered_data_neighbourhood)
plt.xticks(rotation=45)
plt.title('Overall Rating Across Different Neighbourhoods')
plt.show


# In[36]:


#calculating the average 'Overall Rating' for each neighbourhood
average_ratings_by_neighbourhood = airbnb_data_encoded.groupby('neighbourhood')['Overall Rating'].mean().sort_values(ascending=False)

#displaying the top and bottom 5 neighbourhoods in terms of average 'Overall Rating'
top_neighbourhoods = average_ratings_by_neighbourhood.head(5)
bottom_neighbourhoods = average_ratings_by_neighbourhood.tail(5)

top_neighbourhoods, bottom_neighbourhoods


# This analysis provides a clear picture of how guest satisfaction varies across different areas of Berlin. The top-rated neighbourhoods might be offering unique experiences, better amenities, or other desirable features that lead to higher guest satisfaction. Conversely, the lower-rated neighbourhoods might have certain drawbacks or lack features that are important to Airbnb guests.

# In[37]:


#calculating the average 'Overall Rating' for each postal code
average_ratings_by_postal_code = airbnb_data_encoded.groupby('Postal Code')['Overall Rating'].mean().sort_values(ascending=False)

#displaying the top and bottom 5 postal codes in terms of average 'Overall Rating'
top_postal_codes = average_ratings_by_postal_code.head(5)
bottom_postal_codes = average_ratings_by_postal_code.tail(5)

top_postal_codes, bottom_postal_codes


# This analysis provides a snapshot of the geographical distribution of customer satisfaction across different postal codes in Berlin. It's important to consider other factors like the number of listings and reviews in each postal code to fully understand these trends.

# In[38]:


#cross-analysis of other variables like price, property type, and room type across different postal codes
#calculating the average price for each postal code
average_price_by_postal_code = airbnb_data_encoded.groupby('Postal Code')['Price'].mean().sort_values(ascending=False)

#most common property and room types for each postal code
#for simplicity, we select the most common type in each postal code
most_common_property_type_by_postal_code = airbnb_data_encoded.groupby('Postal Code')['Property Type'].agg(lambda x: x.value_counts().index[0])
most_common_room_type_by_postal_code = airbnb_data_encoded.groupby('Postal Code')['Room Type'].agg(lambda x: x.value_counts().index[0])


# In[39]:


#combining these results into a single dataframe
postal_code_analysis = pd.DataFrame({
    'Average Rating': average_ratings_by_postal_code,
    'Average Price': average_price_by_postal_code,
    'Most Common Property Type': most_common_property_type_by_postal_code,
    'Most Common Room Type': most_common_room_type_by_postal_code
})

#displaying the combined data for the top 5 and bottom 5 postal codes in terms of average rating
top_postal_codes_analysis = postal_code_analysis.loc[top_postal_codes.index]
bottom_postal_codes_analysis = postal_code_analysis.loc[bottom_postal_codes.index]

top_postal_codes_analysis, bottom_postal_codes_analysis


# For postal codes with an average rating of 100, the average prices range from relatively low to moderate, suggesting that high satisfaction is not solely dependent on higher pricing.
# The most common property type in these high-rated areas is predominantly 'Apartment', with a mix of room types like 'Entire home/apt', 'Private room', and even 'Shared room'. 
# In the lower-rated postal codes, the average prices also vary, with some areas having higher average prices despite lower ratings. Similar to the top-rated areas, 'Apartment' is the most common property type in these areas, with a mix of 'Entire home/apt' and 'Private room' being the most common room types.
# This analysis indicates that while there's some variation in the type of accommodation and pricing across different postal codes, these factors alone may not fully explain the variations in guest satisfaction.

# In[40]:


#analyzing the impact of Superhost status on overall ratings
superhost_ratings = airbnb_data_encoded.groupby('Is Superhost')['Overall Rating'].mean()

superhost_ratings


# The average overall rating for listings hosted by non-Superhosts is approximately 92.93 while the average overall rating for listings hosted by Superhosts is significantly higher, at approximately 96.85. This indicates that Superhosts, on average, tend to receive higher satisfaction ratings from guests. This could be due to various factors such as better service, experience, or quality of listings, which are criteria for obtaining Superhost status on Airbnb.

# In[41]:


#analyzing the impact of Host Response Time on overall ratings
#since Host Response Time is a categorical variable, we will calculate the average rating for each category
response_time_ratings = airbnb_data_encoded.groupby('Host Response Time')['Overall Rating'].mean().sort_values(ascending=False)

#analyzing the impact of Host Response Rate on overall ratings
#converting Host Response Rate to a numeric value (it might be in percentage format)
airbnb_data_encoded['Host Response Rate'] = airbnb_data_encoded['Host Response Rate'].str.rstrip('%').astype('float') / 100
response_rate_ratings = airbnb_data_encoded.groupby(pd.cut(airbnb_data_encoded['Host Response Rate'], bins=5))['Overall Rating'].mean()

response_time_ratings, response_rate_ratings


# Hosts with different response rates (grouped into bins) show varying average overall ratings. The ratings range from about 92.87 to 94.63, with the highest ratings observed in the highest response rate bin (0.8 to 1.0) and a notable dip in the middle range (0.2 to 0.4).
# These findings suggest that responsiveness, both in terms of response time and rate, does have an impact on guest satisfaction, with faster and more consistent responses correlating with higher ratings.

# #### SENTIMENT ANALYSIS

# In[42]:


nltk.download('averaged_perceptron_tagger')


# In[43]:


def analyze_sentiment(comment):
    #check if the comment is a string
    if isinstance(comment, str):
        analysis = TextBlob(comment)
        return analysis.sentiment.polarity
    else:
        #return None or a neutral sentiment score like 0 for non-string inputs
        return 0 


#applying the function to your comments column
airbnb_data_encoded['Sentiment Score'] = airbnb_data_encoded['Processed_Comments'].apply(analyze_sentiment)


# In[44]:


#calculating the average and median sentiment scores, and plotting a histogram for visualization
average_sentiment = airbnb_data_encoded['Sentiment Score'].mean()
median_sentiment = airbnb_data_encoded['Sentiment Score'].median()

#plotting a histogram
airbnb_data_encoded['Sentiment Score'].hist(bins=50)
plt.title("Distribution of Sentiment Score")
plt.ylabel("Sentiment Score")


# In[45]:


average_sentiment, median_sentiment


# Both the average and median being positive suggest that the majority of reviews are positive, or at least more reviews are positive than negative. The close values of the mean and median indicate a relatively consistent sentiment across the dataset without extreme outliers skewing the results significantly.

# In[46]:


#categorizing sentiments into Positive, Neutral, and Negative.
def categorize_sentiment(score):
    if score > 0.1:
        return 'Positive'
    elif score < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

airbnb_data_encoded['Sentiment Category'] = airbnb_data_encoded['Sentiment Score'].apply(categorize_sentiment)


# In[47]:


#for the analysis, we will compare sentiment scores across Room Type, Host Status, and Property Type
#grouping by Room Type and calculating the average sentiment score
average_sentiment_by_room = airbnb_data_encoded.groupby('Room Type')['Sentiment Score'].mean()

#grouping by Host Status and calculating the average sentiment score
average_sentiment_by_host_status = airbnb_data_encoded.groupby('Is Superhost')['Sentiment Score'].mean()

#grouping by Property Type and calculating the average sentiment score
average_sentiment_by_property_type = airbnb_data_encoded.groupby('Property Type')['Sentiment Score'].mean()


# In[48]:


#displaying the results
average_sentiment_by_room


# In[49]:


average_sentiment_by_host_status


# In[50]:


average_sentiment_by_property_type


# In[51]:


#plotting the average sentiment by room type
average_sentiment_by_room = airbnb_data_encoded.groupby('Room Type')['Sentiment Score'].mean()
average_sentiment_by_room.plot(kind='bar')
plt.title('Average Sentiment Score by Room Type')
plt.ylabel('Average Sentiment Score')
plt.xlabel('Room Type')
plt.show()


# In[52]:


#average Sentiment by Host Status (Superhost)
average_sentiment_by_host_status = airbnb_data_encoded.groupby('Is Superhost')['Sentiment Score'].mean()
average_sentiment_by_host_status.plot(kind='bar', color='green')
plt.title('Average Sentiment Score by Host Status')
plt.ylabel('Average Sentiment Score')
plt.xlabel('Is Superhost')
plt.show()


# In[53]:


#correlation between sentiment scores and numerical ratings
correlation = airbnb_data_encoded['Sentiment Score'].corr(airbnb_data_encoded['Overall Rating'])
correlation


# In[54]:


#defining thresholds for extremely high and low sentiment scores
high_threshold = airbnb_data_encoded['Sentiment Score'].quantile(0.95) 
low_threshold = airbnb_data_encoded['Sentiment Score'].quantile(0.05)   

#filtering the dataset to find comments with extremely high sentiment scores
extremely_high_sentiment = airbnb_data_encoded[airbnb_data_encoded['Sentiment Score'] >= high_threshold]

#filtering the dataset to find comments with extremely low sentiment scores
extremely_low_sentiment = airbnb_data_encoded[airbnb_data_encoded['Sentiment Score'] <= low_threshold]

#displaying the comments with extremely high sentiment scores
print("Comments with Extremely High Sentiment Scores:")
print(extremely_high_sentiment[['Processed_Comments', 'Sentiment Score']])


# In[55]:


#displaying the comments with extremely low sentiment scores
print("\nComments with Extremely Low Sentiment Scores:")
print(extremely_low_sentiment[['Comments', 'Sentiment Score']])


# In[56]:


sentiment_counts = airbnb_data_encoded['Sentiment Category'].value_counts()
sentiment_counts.plot(kind='bar')


# The analysis of sentiment scores across various room types and host statuses in Airbnb listings reveals interesting insights into guest satisfaction. For room types, entire homes or apartments lead with an average sentiment score of 0.316, indicating a generally positive sentiment, likely attributed to the privacy and space they offer. Private rooms follow closely with a score of 0.309, showing that guests are nearly as satisfied as those staying in entire homes or apartments. However, shared rooms score the lowest at 0.3, but still maintain a positive sentiment, possibly reflecting a slightly less satisfying experience due to reduced privacy or space.
# 
# When considering the impact of host status, there's a notable difference in guest satisfaction between Superhosts and non-Superhosts. Superhosts, recognized for their exceptional hospitality, achieve a higher average sentiment score of 0.326, while non-Superhosts have a slightly lower score of 0.303. This difference underscores the positive impact of the Superhost program in enhancing guest experiences.
# 
# Diving into the sentiment scores based on property types, there's a diverse range of guest satisfaction. Some property types like Aparthotels, Boats, and Barns have lower scores (ranging from 0.16 to 0.21), suggesting less guest satisfaction. In contrast, Boutique hotels, Casas particulares (Cuba), and Lofts score higher (0.34 to 0.41), indicating greater guest satisfaction. Interestingly, unconventional property types such as Tipis, Earth houses, and Treehouses show varying scores, with Tipis scoring remarkably high at 0.496, possibly due to their unique and exceptional experiences. Meanwhile, more conventional options like Apartments, Condominiums, and Serviced Apartments show scores around 0.315, aligning with typical guest satisfaction expectations.
# 
# Overall, these results paint a nuanced picture of guest experiences on Airbnb. They highlight the importance of privacy, host quality, and property uniqueness in driving positive guest sentiment. While private accommodations tend to receive higher satisfaction scores, the host's status as a Superhost also plays a significant role in enhancing guest experiences. Moreover, the type of property has a varied impact on satisfaction, with unique or unconventional properties often providing highly satisfactory experiences, possibly owing to their novelty or distinctive features.

# In[57]:


#defining percentiles for categorizing ratings
high_rating_threshold = airbnb_data_encoded['Overall Rating'].quantile(0.75)
medium_rating_threshold = airbnb_data_encoded['Overall Rating'].quantile(0.25)

#creating a function to categorize ratings
def categorize_ratings(row, high_threshold, medium_threshold):
    if row > high_threshold:
        return 'High'
    elif row < medium_threshold:
        return 'Low'
    else:
        return 'Medium'

#applying the categorization to the 'Overall Rating' column
airbnb_data_encoded['new_categorized_rating'] = airbnb_data_encoded['Overall Rating'].apply(categorize_ratings, args=(high_rating_threshold, medium_rating_threshold))

#displaying the first few rows of the airbnb_data_encoded with the new categorized rating
airbnb_data_encoded.head()


# ### Model Building

# In[58]:


#editing the columns for data consistency
airbnb_data_encoded.columns = airbnb_data_encoded.columns.str.lower().str.replace(' ', '_')
airbnb_data_encoded.head()


# In[59]:


#summary statistics of target variable
airbnb_data_encoded['new_categorized_rating'].describe()


# In[60]:


#dropping irrelevant columns
colss = ['index', 'review_id', 'reviewer_id', 'reviewer_name', 'comments','listing_id', 'host_id', 'host_name', 
         'city', 'postal_code', 'latitude', 'longitude', 'neighbourhood', 'instant_bookable', 'business_travel_ready', 
         'last_review', 'first_review', 'square_feet', 'listing_name', 'neighborhood_group', 'property_type', 
         'host_since', 'host_response_time', 'is_superhost', 'is_exact_location', 'room_type', 'processed_comments', 
         'host_response_rate', 'sentiment_category', 'overall_rating', 'value_rating', 'accuracy_rating', 'cleanliness_rating',
         'communication_rating', 'checkin_rating', 'location_rating', 'review_date']

df = airbnb_data_encoded.drop(columns=colss)
df.info()


# In[61]:


missing_values = df.isnull().mean() * 100
columns_with_missing_values = missing_values[missing_values > 0]
columns_with_missing_values


# In[62]:


df = df.dropna()


# In[ ]:





# In[63]:


#seperating my columns for standardization
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
categorical_columns = df.select_dtypes(include=['object', 'category', 'bool']).columns

#initializing the standard scaler
scaler = StandardScaler()

#standardizing only numerical variables
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

#recombining with categorical
df = pd.concat([df[numerical_columns], df[categorical_columns]], axis=1)


# In[64]:


#selecting categorical and boolean columns
categorical_data = df.select_dtypes(include=['object', 'category', 'bool'])

#creating a dictionary to hold category counts for each categorical column
category_counts = {col: categorical_data[col].value_counts() for col in categorical_data.columns}

#converting the dictionary to a DataFrame for a better display
category_counts_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in category_counts.items()]))

#displaying the DataFrame with category counts
category_counts


# In[65]:


#splitting the dataset both vertically and horizontally
X = df.drop(columns= 'new_categorized_rating')
y = df['new_categorized_rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# #### RANDOM FOREST

# In[66]:


train_accuracies = []
test_accuracies = []
max_depth_range = range(20, 41)  

for depth in max_depth_range:
    #start the timer
    start_time = time.time()
    
    #initializing the Random Forest Classifier with varying max_depth
    rfc = RandomForestClassifier(class_weight='balanced', criterion='entropy', max_depth=depth, random_state=42)
    
    #training the model
    rfc.fit(X_train, y_train)
    
    #evaluate on the training set
    train_pred_rfc = rfc.predict(X_train)
    train_accuracy_rfc = accuracy_score(y_train, train_pred_rfc)
    train_accuracies.append(train_accuracy_rfc)
    
    #evaluate on the test set
    test_pred_rfc = rfc.predict(X_test)
    test_accuracy_rfc = accuracy_score(y_test, test_pred_rfc)
    test_accuracies.append(test_accuracy_rfc)
    
    #end the timer and print the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Depth: {depth}, Time elapsed: {elapsed_time:.2f} seconds")


# In[67]:


#plotting the training and test accuracies
plt.figure(figsize=(10, 6))
plt.plot(max_depth_range, train_accuracies, label='Training Accuracy', marker='o')
plt.plot(max_depth_range, test_accuracies, label='Test Accuracy', marker='o')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Random Forest Training and Test Accuracies')
plt.legend()
plt.show()


# In[68]:


#defining the parameter grid
param_grid = {'max_depth': range(27, 38)}

#initializing the Grid Search with cross-validation
grid_search = GridSearchCV(RandomForestClassifier(class_weight='balanced', criterion='entropy', random_state=42), 
                           param_grid, cv=5, scoring='accuracy')

#fitting Grid Search to the data
grid_search.fit(X_train, y_train)


# In[69]:


#getting the best parameters
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

#evaluating the best model found by Grid Search
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# #### DECISION TREE 

# In[70]:


train_accuracies = []
test_accuracies = []
max_depth_range = range(15, 41)

for depth in max_depth_range:
    #start the timer
    start_time = time.time()
    
    #initialize the Decision Tree Classifier with varying max_depth
    dtc = DecisionTreeClassifier(class_weight='balanced', criterion='entropy', max_depth=depth, random_state=42)
    
    #training the model
    dtc.fit(X_train, y_train)
    
    #evaluate on the training set
    train_pred_dtc = dtc.predict(X_train)
    train_accuracy_dtc = accuracy_score(y_train, train_pred_dtc)
    train_accuracies.append(train_accuracy_dtc)
    
    #evaluate on the test set
    test_pred_dtc = dtc.predict(X_test)
    test_accuracy_dtc = accuracy_score(y_test, test_pred_dtc)
    test_accuracies.append(test_accuracy_dtc)
    
    #end the timer and print the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Depth: {depth}, Time elapsed: {elapsed_time:.2f} seconds")


# In[71]:


#plotting the training and test accuracies
plt.figure(figsize=(10, 6))
plt.plot(max_depth_range, train_accuracies, label='Training Accuracy', marker='o')
plt.plot(max_depth_range, test_accuracies, label='Test Accuracy', marker='o')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree Training and Test Accuracies')
plt.legend()
plt.show()


# In[72]:


#defining the parameter grid
param_grid = {'max_depth': [2, 5, 10, 20, 25, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]}

#grid search
grid_search = GridSearchCV(estimator=dtc, param_grid=param_grid, cv=5, n_jobs=1, verbose=2)
grid_search.fit(X_train, y_train)

#best parameters
print("Best Parameters:", grid_search.best_params_)


# In[73]:


#evaluating the best model found by Grid Search
best_params = grid_search.best_params_
dtc_full = DecisionTreeClassifier(**best_params, criterion='entropy', random_state=42)
dtc_full.fit(X_train, y_train)


# In[74]:


y_pred = dtc_full.predict(X_test)
print("Accuracy:", accuracy_score(y_test, test_pred_dtc))
print("Classification Report:\n", classification_report(y_test, test_pred_dtc))
print("Confusion Matrix:\n", confusion_matrix(y_test, test_pred_dtc))


# #### XGBOOST

# In[75]:


#initializing the LabelEncoder
le = LabelEncoder()

#fitting and transforming the labels to numeric values
y_encoded = le.fit_transform(y)

#performing splitting again
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


# In[76]:


train_accuracies = []
test_accuracies = []
max_depth_range = range(1, 16)

for depth in max_depth_range:
    #start the timer
    start_time = time.time()
    
    #initialize the XGBoost Classifier with varying max_depth
    xgb_clf = xgb.XGBClassifier(max_depth=depth, eval_metric='logloss', use_label_encoder=False, random_state=42)
    
    #train the model
    xgb_clf.fit(X_train, y_train)
    
    #evaluate on the training set
    train_pred_xgb = xgb_clf.predict(X_train)
    train_accuracy_xgb = accuracy_score(y_train, train_pred_xgb)
    train_accuracies.append(train_accuracy_xgb)
    
    #evaluate on the test set
    test_pred_xgb = xgb_clf.predict(X_test)
    test_accuracy_xgb = accuracy_score(y_test, test_pred_xgb)
    test_accuracies.append(test_accuracy_xgb)
    
    #end the timer and print the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Depth: {depth}, Time elapsed: {elapsed_time:.2f} seconds")


# In[77]:


#plotting the training and test accuracies
plt.figure(figsize=(10, 6))
plt.plot(max_depth_range, train_accuracies, label='Training Accuracy', marker='o')
plt.plot(max_depth_range, test_accuracies, label='Test Accuracy', marker='o')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('XGBoost: Training vs Test Accuracies')
plt.legend()
plt.show()


# In[80]:


max_depth_range = range(1, 16)  
mean_cv_scores = []

for depth in max_depth_range:
    #start the timer
    start_time = time.time()
    
    #initialize the XGBoost Classifier with varying max_depth
    xgb_clf = xgb.XGBClassifier(max_depth=depth, eval_metric='logloss', use_label_encoder=False, random_state=42)
    
    #perform cross-validation
    cv_scores = cross_val_score(xgb_clf, X_train, y_train, cv=5, scoring='accuracy')
    mean_cv_scores.append(np.mean(cv_scores))
    
    #print the mean CV score and the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Depth: {depth}, Mean CV Accuracy: {np.mean(cv_scores):.4f}, Time elapsed: {elapsed_time:.2f} seconds")


# In[82]:


print(mean_cv_scores)


# In[83]:


xgb_clf = xgb.XGBClassifier(max_depth=10, eval_metric='logloss', use_label_encoder=False, random_state=42)

#training the model
xgb_clf.fit(X_train, y_train)

#predicting on the training set
train_pred_xgb = xgb_clf.predict(X_train)
train_accuracy_xgb = accuracy_score(y_train, train_pred_xgb)

#predict on the test set
test_pred_xgb = xgb_clf.predict(X_test)
test_accuracy_xgb = accuracy_score(y_test, test_pred_xgb)

#classification Report and Confusion Matrix
report = classification_report(y_test, test_pred_xgb)
conf_matrix = confusion_matrix(y_test, test_pred_xgb)

#output results
print(f"Training Accuracy: {train_accuracy_xgb}")
print(f"Test Accuracy: {test_accuracy_xgb}")
print("Classification Report:\n", report)
print("Confusion Matrix:\n", conf_matrix)


# In[ ]:




