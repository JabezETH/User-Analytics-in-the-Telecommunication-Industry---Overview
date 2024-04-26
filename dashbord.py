import streamlit as st
import database
import data_processing
import user_overview_analysis

def main():
    st.title('User Overview Analysis')

    # Connect to the database
    st.subheader('Connect to Database')
    df = database.connect_to_database()
    if df is not None:
        st.success('Connected to database successfully!')
    else:
        st.error('Failed to connect to the database.')
        return

    
    # Calculate user behavior
    st.subheader('User Behavior Analysis')
    merged_df = user_overview_analysis.calculate_user_behavior(df)
    if merged_df is not None:
        st.write(merged_df)
    else:
        st.error('Failed to calculate user behavior.')

    # Visualization
    st.subheader('Data Visualization')

    # Box Plot of Numeric Columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_columns) > 0:
        st.write('Box Plot of Numeric Columns')
        for column in numeric_columns:
            st.write(f'Box Plot of {column}')
            user_overview_analysis.plot_boxplot(df, column)
    else:
        st.warning("No numeric columns found for visualization.")

    # Scatter Plots
    st.write('Scatter Plots')
    for column in ['Social Media DL (Bytes)', 'Youtube DL (Bytes)', 'Email DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Google DL (Bytes)', 'Other DL (Bytes)']:
        st.write(f'Scatter Plot of {column}')
        user_overview_analysis.scatter_plot(df, column)

if __name__ == "__main__":
    main()