import database
import data_processing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


if __name__ == "__main__":
    # Code to be executed when the script is run directly
    df = database.connect_to_database()
    if df is not None:
        df = data_processing.handle_missing_values(df)
        if df is not None:
            df = data_processing.fill_missing_values(df)
            if df is not None:
                df = data_processing.impute_categorical_values(df)
                if df is not None:
                    df = data_processing.replace_outliers(df)
                    if df is not None:
                        print(df)


# Calculating customer engagment 
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

customer_engagment= calculate_customer_engagment(df)
print(customer_engagment)


#top 10 customers per engagement metric 
def plot_top_customer_materics(customer_engagment):
    try:
        fig, axes = plt.subplots(3, figsize=(15, 15))
        top_ten_session_frequency = customer_engagment['session_frequency'].nlargest(10)
        top_ten_duration_session = customer_engagment['session_duration'].nlargest(10)
        top_ten_traffic = customer_engagment['session_traffic'].nlargest(10)

        top_ten_session_frequency.plot(kind='bar', ax=axes[0], title='Top 10 session frequency per Customer')
        top_ten_duration_session.plot(kind='bar', ax=axes[1], title='Top 10 session duration(per hr) per Customer')
        top_ten_traffic.plot(kind='bar', ax=axes[2], title='Top 10 trafic (per GB) per Customer')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error ploting customer engagement: {e}")
        return None

print(plot_top_customer_materics)


#Normalize each engagement metric
def normalize_engagment_materics(customer_engagment):
    try:
        scaler = MinMaxScaler()
        normalized_engagement = scaler.fit_transform(customer_engagment[['session_frequency', 'session_duration', 'session_traffic']])

        # Run k-means clustering (k=3)
        kmeans = KMeans(n_clusters=3, random_state=42)
        customer_engagment['cluster'] = kmeans.fit_predict(normalized_engagement)
    
    except Exception as e:
        print(f"Error to normalize customer engagement: {e}")
        return None

normalized_matrics = normalize_engagment_materics(customer_engagment)
print(normalized_matrics)


# Scatter plot for session frequency vs session duration
def plot_cluster_scatter(customer_engagment, x_column, y_column):
    try:
        plt.figure(figsize=(10, 6))
        
        # Plot scatter plot for each cluster
        for cluster_label in customer_engagment['cluster'].unique():
            cluster_data = customer_engagment[customer_engagment['cluster'] == cluster_label]
            plt.scatter(cluster_data[x_column], cluster_data[y_column], label=f'Cluster {cluster_label}')

        # Add labels, title, legend, and grid
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title(f'{x_column} vs {y_column}')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    except Exception as e:
        print(f"Error plotting cluster scatter: {e}")
plot_duration_frequency = plot_cluster_scatter(customer_engagment, 'session_duration', 'session_frequency')
print(plot_duration_frequency)
plot_duration_frequency = plot_cluster_scatter(customer_engagment, 'session_duration', 'session_traffic')
print(plot_duration_frequency)
plot_duration_frequency = plot_cluster_scatter(customer_engagment, 'session_traffic', 'session_frequency')
print(plot_duration_frequency)


#The minimum, maximum, average & total non-normalized metrics for each cluster
cluster_stats = customer_engagment.groupby('cluster').agg(
    min_session_frequency=('session_frequency', 'min'),
    max_session_frequency=('session_frequency', 'max'),
    avg_session_frequency=('session_frequency', 'mean'),
    total_session_frequency=('session_frequency', 'sum'),
    min_session_duration=('session_duration', 'min'),
    max_session_duration=('session_duration', 'max'),
    avg_session_duration=('session_duration', 'mean'),
    total_session_duration=('session_duration', 'sum'),
    min_session_traffic=('session_traffic', 'min'),
    max_session_traffic=('session_traffic', 'max'),
    avg_session_traffic=('session_traffic', 'mean'),
    total_session_traffic=('session_traffic', 'sum')
)

print(cluster_stats)


# user total traffic per application
def calculate_engagment_per_app(df):
    df['social_media_traffic'] = (df['Social Media DL (Bytes)']+df['Social Media UL (Bytes)'])/1048576
    df['google_traffic'] = (df['Google DL (Bytes)']+df['Google UL (Bytes)'])/1048576
    df['email_traffic'] = (df['Email DL (Bytes)']+df['Email UL (Bytes)'])/1048576
    df['youtube_traffic'] = (df['Youtube DL (Bytes)']+df['Youtube UL (Bytes)'])/1048576
    df['netflix_traffic'] = (df['Netflix DL (Bytes)']+df['Netflix UL (Bytes)'])/1048576
    df['gaming_traffic'] = (df['Gaming DL (Bytes)']+df['Gaming UL (Bytes)'])/1048576
    df['other_traffic'] = (df['Other DL (Bytes)']+df['Other UL (Bytes)'])/1048576
    try:
        engagment_per_app = df.groupby('MSISDN/Number').agg(
            social_media_traffic=('social_media_traffic', 'sum'),
            google_traffic=('google_traffic', 'sum'),
            email_traffic=('email_traffic', 'sum'),
            youtube_traffic=('youtube_traffic', 'sum'),
            netflix_traffic=('netflix_traffic', 'sum'),
            gaming_traffic=('gaming_traffic', 'sum'),
            other_traffic=('other_traffic', 'sum'),

        ).reset_index()
        return engagment_per_app
    
    except Exception as e:
        print(f"Error calculating customer engagement: {e}")
        return None 

engagment_per_app= calculate_engagment_per_app(df)
print(engagment_per_app)


# top 10 most engaged users per application
def plot_top_engagment_per_app(engagment_per_app):
    try:
        fig, axes = plt.subplots(7, figsize=(15, 15))
        top_ten_social_media_traffic= engagment_per_app['social_media_traffic'].nlargest(10)
        top_ten_google_traffic= engagment_per_app['google_traffic'].nlargest(10)
        top_ten_email_traffic= engagment_per_app['email_traffic'].nlargest(10)
        top_ten_youtube_traffic= engagment_per_app['youtube_traffic'].nlargest(10)
        top_ten_netflix_traffic= engagment_per_app['netflix_traffic'].nlargest(10)
        top_ten_gaming_traffic= engagment_per_app['gaming_traffic'].nlargest(10)
        top_ten_other_traffic= engagment_per_app['other_traffic'].nlargest(10)

        top_ten_social_media_traffic.plot(kind='bar', ax=axes[0], title='Top 10 social media traffic (MB) per Customer')
        top_ten_google_traffic.plot(kind='bar', ax=axes[1], title='Top 10 Google traffic (MB) per Customer')
        top_ten_email_traffic.plot(kind='bar', ax=axes[2], title='Top 10 Email traffic (MB) per Customer')
        top_ten_youtube_traffic.plot(kind='bar', ax=axes[3], title='Top 10 Youtube traffic (MB) per Customer')
        top_ten_netflix_traffic.plot(kind='bar', ax=axes[4], title='Top 10 Netflix traffic (MB) per Customer')
        top_ten_gaming_traffic.plot(kind='bar', ax=axes[5], title='Top 10 Gaming traffic (MB) per Customer')
        top_ten_other_traffic.plot(kind='bar', ax=axes[6], title='Top 10 Other traffic (MB) App per Customer')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error ploting customer engagement: {e}")
        return None
top_engagment_per_app = plot_top_engagment_per_app(engagment_per_app)
print(top_engagment_per_app)

#Plot the top 3 most used applications using appropriate charts
def plot_top_3_apps_traffic(engagment_per_app):
    try:
        # Sum the traffic for each application across all customers
        total_traffic_per_app = {
            'Social Media': engagment_per_app['social_media_traffic'].sum(),
            'Google': engagment_per_app['google_traffic'].sum(),
            'Email': engagment_per_app['email_traffic'].sum(),
            'Youtube': engagment_per_app['youtube_traffic'].sum(),
            'Netflix': engagment_per_app['netflix_traffic'].sum(),
            'Gaming': engagment_per_app['gaming_traffic'].sum(),
            'Other': engagment_per_app['other_traffic'].sum()
        }

        # Sort the total traffic in descending order and select the top 3
        top_3_apps_traffic = dict(sorted(total_traffic_per_app.items(), key=lambda item: item[1], reverse=True)[:3])

        # Plot the top 3 most used applications
        plt.bar(top_3_apps_traffic.keys(), top_3_apps_traffic.values())
        plt.xlabel('Application')
        plt.ylabel('Total Traffic (MB)')
        plt.title('Top 3 Most Used Applications by Traffic')
        plt.show()

    except Exception as e:
        print(f"Error plotting top 3 apps traffic: {e}")
plot_top_3_apps_traffic(engagment_per_app)



#  optimized value of k using elbow method 
selected_metrics = customer_engagment
def find_optimal_k_elbow_method(data):
    try:
        sse = []
        k_values = range(1, 11)

        # Fit k-means clustering for each value of k
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data)
            sse.append(kmeans.inertia_)

        # Plot the elbow plot
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, sse, marker='o')
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Sum of Squared Distances (SSE)')
        plt.show()

    except Exception as e:
        print(f"Error occurred: {e}")


find_optimal_k_elbow_method(selected_metrics)

