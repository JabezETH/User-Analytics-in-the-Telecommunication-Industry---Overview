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
                        
                        
# Aggregate, per customer Average TCP retransmission, Average RTT, Handset type, Average throughput

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

print(experience_analytics)

# Top Experiance Analytics 
def top_experience_analytics(experience_analytics):
    try:
        fig, axes = plt.subplots(3, figsize=(15, 15))
        top_ten_average_TCP = experience_analytics['average_TCP'].nlargest(10)
        top_ten_average_RTT_session = experience_analytics['average_RTT'].nlargest(10)
        top_ten_average_throughput = experience_analytics['average_throughput'].nlargest(10)

        top_ten_average_TCP.plot(kind='bar', ax=axes[0], title='Top 10 Average TCP per Customer')
        top_ten_average_RTT_session.plot(kind='bar', ax=axes[1], title='Top 10  Average RTT session per Customer')
        top_ten_average_throughput.plot(kind='bar', ax=axes[2], title='Top 10 Average Throughput per Customer')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error ploting customer engagement: {e}")
        return None
plot = top_experience_analytics(experience_analytics)
print(plot)

top_ten_average_throughput = df.groupby('Handset Type')['average_throughput'].mean().nlargest(10)
print(top_ten_average_throughput)


# Bottom Experiance Analytics 
def bottom_experience_analytics(experience_analytics):
    try:
        fig, axes = plt.subplots(3, figsize=(15, 15))
        bottom_ten_average_TCP = experience_analytics['average_TCP'].nsmallest(10)
        bottom_ten_average_RTT_session = experience_analytics['average_RTT'].nsmallest(10)
        bottom_ten_average_throughput = experience_analytics['average_throughput'].nsmallest(10)

        bottom_ten_average_TCP.plot(kind='bar', ax=axes[0], title='Bottom 10 Average TCP per Customer')
        bottom_ten_average_RTT_session.plot(kind='bar', ax=axes[1], title='Bottom 10 Average RTT session per Customer')
        bottom_ten_average_throughput.plot(kind='bar', ax=axes[2], title='Bottom 10 Average Throughput per Customer')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error plotting customer engagement: {e}")
        return None
    
bottom_experiance_plot = bottom_experience_analytics(experience_analytics)
print(bottom_experience_analytics)


# The distribution of the average throughput  per handset type
top_ten_average_throughput = df.groupby('Handset Type')['average_throughput'].mean().nlargest(10)
print(top_ten_average_throughput)

# Create a scatter plot
plt.scatter(top_ten_average_throughput.index, top_ten_average_throughput.values)
plt.xlabel('Handset Type')
plt.ylabel('Average Throughput')
plt.title('Top 10 Average Throughput per Handset Type')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()


# The distribution of the average throughput  per handset type
top_ten_average_throughput = df.groupby('Handset Type')['average_TCP'].mean().nlargest(10)
print(top_ten_average_throughput)

# Create a scatter plot
plt.scatter(top_ten_average_throughput.index, top_ten_average_throughput.values)
plt.xlabel('Handset Type')
plt.ylabel('Average TCP')
plt.title('Top 10 Average TCP per Handset Type')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()


#k-means clustering (where k = 3)

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
normalized_metrics = normalize_experience_metrics(experience_analytics)
print(normalized_metrics)

# Scatter plot for session frequency vs session duration
def plot_cluster_scatter(experience_analytics, x_column, y_column):
    try:
        plt.figure(figsize=(10, 6))
        
        # Plot scatter plot for each cluster
        for cluster_label in experience_analytics['cluster'].unique():
            cluster_data = experience_analytics[experience_analytics['cluster'] == cluster_label]
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

# Plot scatter plot for average TCP vs average RTT
plot_average_TCP = plot_cluster_scatter(normalized_metrics, 'average_TCP', 'average_RTT')
plot_average_TCP = plot_cluster_scatter(normalized_metrics, 'average_TCP', 'average_throughput')
plot_average_TCP = plot_cluster_scatter(normalized_metrics, 'average_throughput', 'average_RTT')
