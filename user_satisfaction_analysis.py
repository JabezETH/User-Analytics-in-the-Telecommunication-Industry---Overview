import database
import data_processing
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model

def calculate_customer_engagment(df):
    try:
        df['session_duration_hours(hr)'] = df['Dur. (ms)'] / (1000 * 60 * 60)
        df['total_dl_ul'] = (df['Total UL (Bytes)'] + df['Total DL (Bytes)'])
        df['total_dl_ul(GB)'] = df['total_dl_ul']/(1024**3)
        customer_engagement = df.groupby('MSISDN/Number').agg(
            session_frequency=('Bearer Id', 'nunique'),  # Count the number of unique sessions
            session_duration=('session_duration_hours(hr)', 'sum'),       # Sum of session durations
            session_traffic=('total_dl_ul(GB)', 'sum'),  # Sum of uplink session traffic
        ).reset_index()
        return customer_engagement
    except Exception as e:
        print(f"Error calculating customer engagement: {e}")
        return None 

def normalize_materics(customer_engagment):
    try:
        scaler = MinMaxScaler()
        normalized_engagement = scaler.fit_transform(customer_engagment[['session_frequency', 'session_duration', 'session_traffic']])

        # Run k-means clustering (k=3)
        kmeans = KMeans(n_clusters=3, random_state=42)
        customer_engagment['cluster'] = kmeans.fit_predict(normalized_engagement)
        
        return customer_engagment
    except Exception as e:
        print(f"Error normalizing customer engagement: {e}")
        return None

# Load data
df = database.connect_to_database()

# Preprocess data
if df is not None:
    df = data_processing.handle_missing_values(df)
    if df is not None:
        df = data_processing.fill_missing_values(df)
        if df is not None:
            df = data_processing.impute_categorical_values(df)
            if df is not None:
                df = data_processing.replace_outliers(df)
               

# Calculate customer engagement
customer_engagement = calculate_customer_engagment(df)

# Normalize engagement metrics
if customer_engagement is not None:
    normalized_metrics = normalize_materics(customer_engagement)
    print(normalized_metrics)

# Calculate engagement scores
if normalized_metrics is not None:
    # Calculate centroids of each cluster
    cluster_centroids = {}
    for cluster_label in normalized_metrics['cluster'].unique():
        cluster_data = normalized_metrics[normalized_metrics['cluster'] == cluster_label]
        centroid = cluster_data[['session_frequency', 'session_duration', 'session_traffic']].mean()
        cluster_centroids[cluster_label] = centroid.values

    # Function to calculate Euclidean distance between a user data point and a centroid
    def calculate_euclidean_distance(user_data_point, centroid):
        return np.linalg.norm(user_data_point - centroid)

    # Dictionary to store engagement scores per user
    engagement_scores_per_user = {}

    # Iterate over each user (MSISDN/Number)
    for msisdn, user_data in normalized_metrics.groupby('MSISDN/Number'):
        user_engagement_scores = []
        # Iterate over each user data point
        for index, row in user_data.iterrows():
            user_data_point = row[['session_frequency', 'session_duration', 'session_traffic']].values
            # Calculate Euclidean distance between the user data point and the centroid of the least engaged cluster
            least_engaged_cluster = min(cluster_centroids.keys())
            least_engaged_centroid = cluster_centroids[least_engaged_cluster]
            euclidean_distance = calculate_euclidean_distance(user_data_point, least_engaged_centroid)
            user_engagement_scores.append(euclidean_distance)
        # Assign the minimum engagement score as the user's engagement score
        engagement_scores_per_user[msisdn] = min(user_engagement_scores)

    # Create a DataFrame to display the MSISDN/Number and corresponding engagement score
    engagement_scores_df = pd.DataFrame(list(engagement_scores_per_user.items()), columns=['MSISDN/Number', 'Engagement Score'])

    # Now engagement_scores_df contains the MSISDN/Number and the corresponding engagement score for each user
    #print(engagement_scores_df)


df['average_TCP'] = df['TCP DL Retrans. Vol (Bytes)'] + df['TCP UL Retrans. Vol (Bytes)']
df['average_RTT'] = df['Avg RTT DL (ms)'] + df['Avg RTT UL (ms)']
df['average_throughput'] = df['Avg Bearer TP DL (kbps)'] + df['Avg Bearer TP UL (kbps)']

# Group by 'MSISDN/Number' and aggregate metrics
experience_analytics = df.groupby('MSISDN/Number').agg(
    average_TCP=('average_TCP', 'mean'),
    hand_set=('Handset Type', lambda x: x.mode().iloc[0]),  # Mode for categorical column
    average_RTT=('average_RTT', 'mean'),
    average_throughput=('average_throughput', 'mean')
)
def normalize_experience_metrics(experience_analytics):
    try:
        scaler = MinMaxScaler()
        normalized_experience = scaler.fit_transform(experience_analytics[['average_TCP', 'average_RTT', 'average_throughput']])

        # Run k-means clustering (k=3)
        kmeans = KMeans(n_clusters=3, random_state=42)
        experience_analytics['cluster'] = kmeans.fit_predict(normalized_experience)
        
        return experience_analytics  # Return the modified DataFrame
    
    except Exception as e:
        print(f"Error normalizing experience metrics: {e}")
        return None

# Normalize experience metrics


customer_engagment = normalize_experience_metrics(experience_analytics)

# Calculate centroids of each cluster
cluster_centroids = {}
for cluster_label in customer_engagment['cluster'].unique():
    cluster_data = customer_engagment[customer_engagment['cluster'] == cluster_label]
    centroid = cluster_data[['average_TCP', 'average_RTT', 'average_throughput']].mean()
    cluster_centroids[cluster_label] = centroid.values

# Function to calculate Euclidean distance between a user data point and a centroid
def calculate_distance(user_data_point, centroid):
    return np.linalg.norm(user_data_point - centroid)

# Dictionary to store engagement scores per user
experiance_scores_per_user = {}

# Iterate over each user (MSISDN/Number)
for msisdn, user_data in customer_engagment.groupby('MSISDN/Number'):
    user_scores = []
    # Iterate over each user data point
    for index, row in user_data.iterrows():
        user_data_point = row[['average_TCP', 'average_RTT', 'average_throughput']].values
        # Calculate distance between the user data point and the centroid of the least engaged cluster
        least_engaged_cluster = min(cluster_centroids.keys())
        least_engaged_centroid = cluster_centroids[least_engaged_cluster]
        distance = calculate_distance(user_data_point, least_engaged_centroid)
        user_scores.append(distance)
    # Assign the minimum score as the user's engagement score
    experiance_scores_per_user[msisdn] = min(user_scores)

# Create a DataFrame to display the MSISDN/Number and corresponding engagement score
experiance_scores_df = pd.DataFrame(list(experiance_scores_per_user.items()), columns=['MSISDN/Number', 'Experiance Score'])

# Now engagement_scores_df contains the MSISDN/Number and the corresponding engagement score for each user
print(experiance_scores_df)


merged_df = pd.merge(experiance_scores_df, engagement_scores_df, on='MSISDN/Number', how='inner')


merged_df['mean_satisfactory_score'] = merged_df[['Experiance Score', 'Engagement Score']].mean(axis=1)

# Get the top 10 satisfied users
top_satisfied_users = merged_df.sort_values(by='mean_satisfactory_score', ascending=False).head(10)

# Plot the top 10 satisfied users using a bar plot with Seaborn
sns.barplot(data=top_satisfied_users, y='mean_satisfactory_score', x='MSISDN/Number', order=top_satisfied_users.sort_values('mean_satisfactory_score', ascending=False)['MSISDN/Number'])
plt.xlabel('MSISDN/Number')
plt.ylabel('Mean Satisfactory Score')
plt.title('Top 10 Satisfied Customers')
plt.show()

# Display the top satisfied users DataFrame
print(top_satisfied_users)


from sklearn.model_selection import train_test_split
from pandas.core.common import random_state
from sklearn.linear_model import LinearRegression
# Splitting variables
merged_df = pd.merge(merged_df, customer_engagement[['MSISDN/Number','session_frequency', 'session_duration', 'session_traffic']], on='MSISDN/Number', how='inner')

# Assuming X is the feature matrix and y is the target variable (Satisfaction_Score)
X = merged_df[['session_frequency', 'session_duration', 'session_traffic']]
y = merged_df['mean_satisfactory_score']

# Handle missing values by filling with the mean
X.fillna(X.mean(), inplace=True)
y.fillna(y.mean(), inplace=True)

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (optional but recommended for some models)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build a linear regression model
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
predictions = model.predict(X_test_scaled)

# Evaluate the model
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Display the model performance
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Optionally, you can inspect the coefficients and intercept
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)


# Plot actual vs. predicted values
plt.scatter(y_test, predictions, color='blue', label='Actual vs. Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Perfect Prediction')
plt.title('Actual vs. Predicted Satisfaction Score')
plt.xlabel('Actual Satisfaction Score')
plt.ylabel('Predicted Satisfaction Score')
plt.legend()
plt.grid(True)
plt.show()


satisfaction_scores = merged_df 

import psycopg2

# Replace these values with your database credentials
hostname = 'localhost'
username = 'postgres'
password = ';'  # Replace with your actual password
new_database = 'satisfaction_scores_db'

# Establish a connection to the default PostgreSQL database
conn = psycopg2.connect(host=hostname, user=username, password=password)

# Create a new database
conn.autocommit = True
cur = conn.cursor()
cur.execute(f"CREATE DATABASE {new_database}")
cur.close()
conn.close()

# Establish a connection to the new database
conn = psycopg2.connect(host=hostname, database=new_database, user=username, password=password)


satisfaction_scores.to_sql(name='satisfaction_scores_table', con=conn, if_exists='replace', index=False, method='postgresql')

# Close the connection
conn.close()






