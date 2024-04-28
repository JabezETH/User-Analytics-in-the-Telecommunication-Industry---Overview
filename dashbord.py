import streamlit as st
import database
import data_processing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import database
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


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
                   

# Define functions for each page
def page_home():
    st.title('User Analytics for Tellco')
    st.write('User Overview')
    no_of_users=df['MSISDN/Number'].nunique()
    no_of_users = format(no_of_users, ',') 
    total_session_duration = int((df['Dur. (ms)'].sum())/(1000 * 60 * 60))
    df['total_dl_ul'] = (df['Total UL (Bytes)'] + df['Total DL (Bytes)'])
    total_traffic = int((df['total_dl_ul'].sum())/ (1024 ** 4)) 

    

    st.markdown(
        f"""
        <div style="
            background-color: black;
            color: white;
            padding: 5px; 
            border-radius: 5px;
            width: 600px; 
        ">
            <h1 style="font-size: 20px; color: white;">Total Number of users analyzed: {no_of_users}</h1>
            <h1 style="font-size: 20px; color: white;">Total Session Hours Analyzed: {total_session_duration}</h1>
            <h1 style="font-size: 20px; color: white;">Total session traffic analyzed: {total_traffic} (TB)</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    
    df_corr = df[['Google DL (Bytes)', 'Google UL (Bytes)', 'Email DL (Bytes)',
                'Email UL (Bytes)', 'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
                'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 'Gaming DL (Bytes)',
                'Gaming UL (Bytes)', 'Other DL (Bytes)', 'Other UL (Bytes)',]].corr()

    # Create a Matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the heatmap using Seaborn
    sns.heatmap(df_corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)

    # Set title and axis labels
    ax.set_title('Correlation Matrix Heatmap')
    ax.set_xlabel('Features')
    ax.set_ylabel('Features')

    # Display the plot in Streamlit
    st.pyplot(fig)
                
    
    df.dropna(inplace=True)

    # Identify and drop non-numeric columns
    non_numeric_columns = df.select_dtypes(exclude=['float64', 'int64']).columns
    df_numeric = df.drop(non_numeric_columns, axis=1)

    # Standardize the data (important for PCA)
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(df_numeric)

    # Perform PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_standardized)

    # Explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = explained_variance_ratio.cumsum()

    ig, ax = plt.subplots()
    st.plot(range(1, len(explained_variance_ratio) + 1), cumulative_explained_variance, marker='o')
    st.set_title('Cumulative Explained Variance')
    st.set_xlabel('Number of Principal Components')
    st.set_ylabel('Cumulative Explained Variance')

    # Display the plot in Streamlit
    st.pyplot(fig)
        
    
    

def user_engagment():
    st.title('User Engagment Analysis')
    st.write('This is findings of user engagment analysis.')
    
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
            fig = px.bar(customer_engagment.nlargest(10, 'session_frequency'),
                     x='MSISDN/Number', y='session_frequency', title='Top 10 session frequency per Customer')
            fig.update_layout(yaxis_title='Session Frequency', xaxis_title='Customer ID')
            st.plotly_chart(fig)

            fig = px.bar(customer_engagment.nlargest(10, 'session_duration'),
                        x='MSISDN/Number', y='session_duration', title='Top 10 session duration (per hr) per Customer')
            fig.update_layout(yaxis_title='Session Duration (hours)', xaxis_title='Customer ID')
            st.plotly_chart(fig)

            fig = px.bar(customer_engagment.nlargest(10, 'session_traffic'),
                        x='MSISDN/Number', y='session_traffic', title='Top 10 traffic (per GB) per Customer')
            fig.update_layout(yaxis_title='Session Traffic (GB)', xaxis_title='Customer ID')
            st.plotly_chart(fig)
                
        except Exception as e:
                print(f"Error ploting customer engagement: {e}")
                return None

    top_ten_engaged_customers = plot_top_customer_materics(customer_engagment)
    print(top_ten_engaged_customers)


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
            fig = px.scatter(customer_engagment, x=x_column, y=y_column, color='cluster',
                            title=f'{x_column} vs {y_column}', labels={'cluster': 'Cluster'})
            fig.update_layout(showlegend=True, legend_title_text='Cluster')
            st.plotly_chart(fig)
            
        except Exception as e:
            st.error(f"Error plotting cluster scatter: {e}")

 


    st.header("Plot Cluster Scatter")
    x_column = st.selectbox("Select X-axis column", options=customer_engagment.columns[:-1])
    y_column = st.selectbox("Select Y-axis column", options=customer_engagment.columns[:-1])
    plot_cluster_scatter(customer_engagment, x_column, y_column)

  

def experiance_analytics():
    st.title('Experiance Analysis for Tellco')
    st.write('Experiance Analysis.')
    
def satisfaction_analysis():
    st.title('Satisfaction Analysis for Tellco')
    st.write('Satisfaction Analysis.')
    


# Create sidebar with page selection
page = st.sidebar.radio("Select Page", ('Home', 'User Engagment', 'Experiance Analysis', 'Satisfaction Analysis'))

# Display selected page in the main content area
if page == 'Home':
    page_home()
elif page == 'User Engagment':
    user_engagment()
elif page == 'Contact':
    experiance_analytics()
elif page == 'Contact':
    satisfaction_analysis()