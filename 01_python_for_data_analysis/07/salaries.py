"""
SF Salaries Exercise
"""
from pandas import read_csv, DataFrame

# Read Salaries.csv as a dataframe called sal
sal = DataFrame(read_csv("Salaries.csv"))
print(sal.head())
print(sal.info())
print(sal["BasePay"].mean())
print(sal["OvertimePay"].mean())
print(sal["OvertimePay"].max())
print(sal[sal["EmployeeName"] == "JOSEPH DRISCOLL"]["JobTitle"])
print(sal[sal["EmployeeName"] == "JOSEPH DRISCOLL"]["TotalPayBenefits"])
print(sal[sal["TotalPayBenefits"] == sal["TotalPayBenefits"].max()])
print(sal[sal["TotalPayBenefits"] == sal["TotalPayBenefits"].min()])
print(sal.groupby("Year").mean()["BasePay"])
print(sal["JobTitle"].nunique())
print(sal["JobTitle"].value_counts().head(5))
print(sum(sal[sal["Year"] == 2013]["JobTitle"].value_counts() == 1))

# How many people have the word Chief in their job title?
chief_string = lambda x: "chief" in x.lower()
print(sum(sal["JobTitle"].apply(chief_string)))
sal['title_len'] = sal['JobTitle'].apply(len)
print(sal[['title_len','TotalPayBenefits']].corr())
