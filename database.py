import pandas as pd



from sqlalchemy import create_engine

def connect_to_database(database_name='telecom', table_name='xdr_data', connection_params=None):
    try:
        if connection_params is None:
            connection_params = {
                "host": "localhost",
                "user": "postgres",
                "password": ";",
                "port": "5432",
                "database": database_name
            }
        engine = create_engine(f"postgresql+psycopg2://{connection_params['user']}:{connection_params['password']}@{connection_params['host']}:{connection_params['port']}/{connection_params['database']}")
        sql_query = f'SELECT * FROM {table_name}'
        df = pd.read_sql(sql_query, con=engine)
        return df
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None