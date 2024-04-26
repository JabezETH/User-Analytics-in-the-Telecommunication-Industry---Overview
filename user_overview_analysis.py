import database
import data_processing
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Overview of the users’ behaviour
def calculate_user_behavior(df):
    try:
        sessions_per_user = df.groupby('MSISDN/Number')['Bearer Id'].nunique().reset_index()
        session_duration_per_user = df.groupby('MSISDN/Number')['Dur. (ms)'].sum().reset_index()
        Total_DL_UL_per_user = df.groupby('MSISDN/Number')[['Total UL (Bytes)', 'Total DL (Bytes)']].sum().reset_index()
        SocialMedia_DL_UL_per_user = df.groupby('MSISDN/Number')[['Social Media DL (Bytes)', 'Social Media UL (Bytes)']].sum().reset_index()
        Google_DL_UL_per_user = df.groupby('MSISDN/Number')[['Google DL (Bytes)', 'Google UL (Bytes)']].sum().reset_index()
        Email_DL_UL_per_user = df.groupby('MSISDN/Number')[['Email DL (Bytes)', 'Email UL (Bytes)']].sum().reset_index()
        Youtube_DL_UL_per_user = df.groupby('MSISDN/Number')[['Youtube DL (Bytes)', 'Youtube UL (Bytes)']].sum().reset_index()
        Netflix_DL_UL_per_user = df.groupby('MSISDN/Number')[['Netflix DL (Bytes)', 'Netflix UL (Bytes)']].sum().reset_index()
        Gaming_DL_UL_per_user = df.groupby('MSISDN/Number')[['Gaming DL (Bytes)', 'Gaming UL (Bytes)']].sum().reset_index()
        Other_DL_UL_per_user = df.groupby('MSISDN/Number')[['Other DL (Bytes)', 'Other UL (Bytes)']].sum().reset_index()

        merged_df = sessions_per_user
        for data in [session_duration_per_user, Total_DL_UL_per_user, SocialMedia_DL_UL_per_user, Google_DL_UL_per_user, Email_DL_UL_per_user,
                     Youtube_DL_UL_per_user, Netflix_DL_UL_per_user, Gaming_DL_UL_per_user, Other_DL_UL_per_user]:
            merged_df = merged_df.merge(data, on='MSISDN/Number', how='outer')

        merged_df.fillna(0, inplace=True)
        return merged_df
    except Exception as e:
        print(f"Error calculating user behavior: {e}")
        return None


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
                


# Data type
df_data_type = df.info()

# Basic metricical analysis
df_basic_metric = df.describe() 

# Non-Graphical Univariate Analysis
df_basic_metric

# Graphical Univariate Analysis
numeric_columns = 'Bearer Id', 'Start ms', 'End ms', 'Dur. (ms)', 'IMSI', 'MSISDN/Number',
'IMEI', 'Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)',
'Avg Bearer TP UL (kbps)', 'TCP DL Retrans. Vol (Bytes)',
'TCP UL Retrans. Vol (Bytes)', 'DL TP < 50 Kbps (%)',
'50 Kbps < DL TP < 250 Kbps (%)', '250 Kbps < DL TP < 1 Mbps (%)',
'DL TP > 1 Mbps (%)', 'UL TP < 10 Kbps (%)',
'10 Kbps < UL TP < 50 Kbps (%)', '50 Kbps < UL TP < 300 Kbps (%)',
'UL TP > 300 Kbps (%)', 'HTTP DL (Bytes)', 'HTTP UL (Bytes)',
'Activity Duration DL (ms)', 'Activity Duration UL (ms)', 'Dur. (ms).1',
'Nb of sec with 125000B < Vol DL',
'Nb of sec with 1250B < Vol UL < 6250B',
'Nb of sec with 31250B < Vol DL < 125000B',
'Nb of sec with 37500B < Vol UL',
'Nb of sec with 6250B < Vol DL < 31250B',
'Nb of sec with 6250B < Vol UL < 37500B',
'Nb of sec with Vol DL < 6250B', 'Nb of sec with Vol UL < 1250B',
'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
'Google DL (Bytes)', 'Google UL (Bytes)', 'Email DL (Bytes)',
'Email UL (Bytes)', 'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 'Gaming DL (Bytes)',
'Gaming UL (Bytes)', 'Other DL (Bytes)', 'Other UL (Bytes)',
'Total UL (Bytes)', 'Total DL (Bytes)'
# Set up subplots
fig, axes = plt.subplots(nrows=len(numeric_columns), ncols=1, figsize=(10, 5 * len(numeric_columns)))
# Loop through numeric columns and create box plots
for i, column in enumerate(numeric_columns):
    sns.boxplot(x=df[column], ax=axes[i])
    axes[i].set_title(f'Box Plot of {column}')
plt.tight_layout()
plt.show()

#Bivariate Analysis 
def scatter_plot(x_column):
    df['total_UL_DL'] = df['Total UL (Bytes)'] + df['Total DL (Bytes)']
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=x_column, y='total_UL_DL', ax=ax)
    plt.show()
    
def plot():   
    Social_media = scatter_plot(df['Social Media DL (Bytes)'])
    Youtube =scatter_plot(df['Youtube DL (Bytes)'])
    Email = scatter_plot(df['Email DL (Bytes)'])
    Netfix = scatter_plot(df['Netflix DL (Bytes)'])
    Gaming = scatter_plot(df['Gaming DL (Bytes)'])
    Google = scatter_plot(df['Google DL (Bytes)'])
    Other = scatter_plot(df['Other DL (Bytes)'])



# Calculate total duration per user
total_duration_per_user = df.groupby('MSISDN/Number')['Dur. (ms)'].sum().reset_index()

# Calculate deciles based on total duration
deciles = total_duration_per_user['Dur. (ms)'].quantile([0, 0.2, 0.4, 0.6, 0.8])

# Function to assign decile class to each user
def assign_decile_class(duration):
    if duration <= deciles[0]:
        return 'D1'
    elif duration <= deciles[0.2]:
        return 'D2'
    elif duration <= deciles[0.4]:
        return 'D3'
    elif duration <= deciles[0.6]:
        return 'D4'
    elif duration <= deciles[0.8]:
        return 'D5'


# Assign decile class to each user
total_duration_per_user['Decile Class'] = total_duration_per_user['Dur. (ms)'].apply(assign_decile_class)

# Calculate total data (DL+UL) per decile class
total_data_per_decile = total_duration_per_user.groupby('Decile Class')['Dur. (ms)'].sum()

print("Total Data (DL+UL) per Decile Class:")
print(total_data_per_decile)

# Correlation Analysis

df_corr = df[['Google DL (Bytes)', 'Google UL (Bytes)', 'Email DL (Bytes)',
       'Email UL (Bytes)', 'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
       'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 'Gaming DL (Bytes)',
       'Gaming UL (Bytes)', 'Other DL (Bytes)', 'Other UL (Bytes)',]].corr()
plt.figure(figsize=(20, 15))  
sns.heatmap(df_corr, annot=True)  
plt.show()

# PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# Drop any rows with missing values for simplicity
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

# Plot the explained variance
plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_explained_variance, marker='o')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()

# Choose the number of components based on the plot or a desired threshold 
num_components = 3  

# Retain the selected number of components
X_pca_selected = X_pca[:, :num_components]

# Interpretation of principal components
principal_components_df = pd.DataFrame(pca.components_, columns=df_numeric.columns)
print("Principal Component Loadings:")
print(principal_components_df)



# Visualize in the reduced-dimensional space (for 2D and 3D)
if num_components == 2:
    plt.scatter(X_pca_selected[:, 0], X_pca_selected[:, 1], marker='o')
    plt.title('PCA: Reduced 2D Space')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()
elif num_components == 3:
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_pca_selected[:, 0], X_pca_selected[:, 1], X_pca_selected[:, 2], marker='o')
    ax.set_title('PCA: Reduced 3D Space')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    plt.show()