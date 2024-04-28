import pandas as pd
import database

# Handling Missing Values 
def handle_missing_values(df):
    try:
        threshold = len(df) * 0.05
        cols_to_drop = df.columns[df.isna().sum() <= threshold]
        df.dropna(subset=cols_to_drop, inplace=True)
        return df
    except Exception as e:
        print(f"Error handling missing values: {e}")
        return None


def fill_missing_values(df):
    try:
        for column_mean in ['Bearer Id', 'Start ms',  'End ms', 'Dur. (ms)',
                            'MSISDN/Number', 'Avg RTT DL (ms)', 'Avg RTT UL (ms)',
                            'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)',
                            'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)',
                            'DL TP < 50 Kbps (%)', '50 Kbps < DL TP < 250 Kbps (%)',
                            '250 Kbps < DL TP < 1 Mbps (%)', 'DL TP > 1 Mbps (%)',
                            'UL TP < 10 Kbps (%)', '10 Kbps < UL TP < 50 Kbps (%)',
                            '50 Kbps < UL TP < 300 Kbps (%)', 'UL TP > 300 Kbps (%)',
                            'HTTP DL (Bytes)', 'HTTP UL (Bytes)', 'Activity Duration DL (ms)',
                            'Activity Duration UL (ms)', 'Dur. (ms).1',
                            'Nb of sec with 125000B < Vol DL',
                            'Nb of sec with 1250B < Vol UL < 6250B',
                            'Nb of sec with 31250B < Vol DL < 125000B',
                            'Nb of sec with 37500B < Vol UL',
                            'Nb of sec with 6250B < Vol DL < 31250B','IMEI','IMSI',
                            'Nb of sec with 6250B < Vol UL < 37500B',
                            'Nb of sec with Vol DL < 6250B', 'Nb of sec with Vol UL < 1250B',
                            'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
                            'Google DL (Bytes)', 'Google UL (Bytes)', 'Email DL (Bytes)',
                            'Email UL (Bytes)', 'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
                            'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 'Gaming DL (Bytes)',
                            'Gaming UL (Bytes)', 'Other DL (Bytes)', 'Other UL (Bytes)',
                            'Total UL (Bytes)', 'Total DL (Bytes)']:
            try:
                df[column_mean] = pd.to_numeric(df[column_mean], errors='coerce')
                mean_value = df[column_mean].mean(skipna=True)
                df[column_mean] = df[column_mean].fillna(mean_value)
            except Exception as e:
                print(f"Error processing column '{column_mean}': {e}")
        return df
    except Exception as e:
        print(f"Error filling missing values: {e}")
        return None


def impute_categorical_values(df):
    try:
        for column in ['Start', 'End', 'Last Location Name', 'Handset Type', 'Handset Manufacturer']:
            mode_value = df[column].mode()[0]
            df[column] = df[column].fillna(mode_value)
        return df
    except Exception as e:
        print(f"Error imputing categorical values: {e}")
        return None
    
    
def replace_outliers(df):
    try:
    # Convert numerical columns to numeric type
        numerical_columns = ['Bearer Id', 'Start ms',  'End ms', 'Dur. (ms)',
                        'MSISDN/Number', 'Avg RTT DL (ms)', 'Avg RTT UL (ms)',
                        'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)',
                        'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)',
                        'DL TP < 50 Kbps (%)', '50 Kbps < DL TP < 250 Kbps (%)',
                        '250 Kbps < DL TP < 1 Mbps (%)', 'DL TP > 1 Mbps (%)',
                        'UL TP < 10 Kbps (%)', '10 Kbps < UL TP < 50 Kbps (%)',
                        '50 Kbps < UL TP < 300 Kbps (%)', 'UL TP > 300 Kbps (%)',
                        'HTTP DL (Bytes)', 'HTTP UL (Bytes)', 'Activity Duration DL (ms)',
                        'Activity Duration UL (ms)', 'Dur. (ms).1',
                        'Nb of sec with 125000B < Vol DL',
                        'Nb of sec with 1250B < Vol UL < 6250B',
                        'Nb of sec with 31250B < Vol DL < 125000B',
                        'Nb of sec with 37500B < Vol UL',
                        'Nb of sec with 6250B < Vol DL < 31250B','IMEI','IMSI',
                        'Nb of sec with 6250B < Vol UL < 37500B',
                        'Nb of sec with Vol DL < 6250B', 'Nb of sec with Vol UL < 1250B',
                        'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
                        'Google DL (Bytes)', 'Google UL (Bytes)', 'Email DL (Bytes)',
                        'Email UL (Bytes)', 'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
                        'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 'Gaming DL (Bytes)',
                        'Gaming UL (Bytes)', 'Other DL (Bytes)', 'Other UL (Bytes)',
                        'Total UL (Bytes)', 'Total DL (Bytes)']

        df[numerical_columns] = df[numerical_columns].apply(pd.to_numeric, errors='coerce')

        # Handle missing or invalid values
        df.fillna(0, inplace=True)  # Replace missing values with 0

        # Replace outliers with mean for numerical columns
        for column in numerical_columns:
            mean_value = df[column].mean()
            df[column] = df[column].apply(lambda x: mean_value if x < 0 else x)

        # Replace outliers with mode for categorical columns
        categorical_columns = ['Start', 'End', 'Last Location Name', 'Handset Type', 'Handset Manufacturer']
        for column in categorical_columns:
            mode_value = df[column].mode()[0]
            df[column] = df[column].apply(lambda x: mode_value if not isinstance(x, str) else x)

        return df

    except Exception as e:
        print(f"Error occurred: {e}")
        return None

# Apply outlier replacement function to DataFrame
df = database.connect_to_database()
df_cleaned = replace_outliers(df)