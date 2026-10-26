import sqlite3
import pandas as pd

def run_sql():
    conn = sqlite3.connect("data/instamart.db")
    
    with open("sql/queries.sql", "r") as f:
        sql_file = f.read()
        
    # Split queries by semicolon
    queries = sql_file.split(';')
    
    for i, query in enumerate(queries):
        if query.strip():
            print(f"\n" + "="*40)
            print(f"QUERY Q{i+1} RESULTS")
            print("="*40)
            df = pd.read_sql_query(query, conn)
            print(df.head(10)) # Print top 10 rows
            print(f"Total Rows Returned: {len(df)}")
            
    conn.close()

if __name__ == "__main__":
    run_sql()