import pyodbc
import pandas as pd

def runQueryString(query,dsn):
    con = pyodbc.connect("DSN="+dsn, autocommit = True)
    df= pd.read_sql_query(query, con, index_col=None, coerce_float=True, params=None, parse_dates=None, chunksize=None)
    con.close()
    return(df)
	
def runQueryFile(file_name,dsn):
	with open(file_name, 'r') as myfile:
		return runQueryString(myfile.read(),dsn)
		
def runQueryName(query_name,dsn):
	return runQueryFile('data/'+query_name+'.sql',dsn)