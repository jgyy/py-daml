"""
Data Input and Output
"""
from requests import get
from sqlalchemy import create_engine
from pandas import read_csv, read_excel, read_html, read_sql

# CSV
df = read_csv("example")
print(df)
df.to_csv("example", index=False)

# Excel
print(read_excel("Excel_Sample.xlsx", sheet_name="Sheet1"))
df.to_excel("Excel_Sample.xlsx", sheet_name="Sheet1")

# HTML
header = {
    "User-Agent": "Mozilla/5.0",
    "X-Requested-With": "XMLHttpRequest",
}
df = read_html(get("https://datatables.net", headers=header).text)[0]
print(df)

# SQL
engine = create_engine("sqlite:///:memory:")
df.to_sql("data", engine)
sql_df = read_sql("data", con=engine)
print(sql_df)
