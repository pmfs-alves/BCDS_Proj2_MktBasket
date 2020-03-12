import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# Display output fully
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# ------------------------------------------- Import file and initial check --------------------------------------------
df1 = pd.read_csv('./data/AsianRestaurant_Cyprus_2018.txt', sep=';')
df2 = pd.read_csv('./data/AsianRestaurant_Cyprus_2018_partial.txt', sep=';')
df1.dtypes
# correct TotalAmount data type
df1.TotalAmount = df1.TotalAmount.apply(lambda x: float(str(x.replace(',', '.'))))
df1.describe(include="all")
# Checking missing values
# só customer city e customer since têm NULLS
# CustomerCity - 52 861
# CustomerSince - 54967
df1.isnull().sum()

# Customers that ate at the restaurant have missing values in CustomerCity and CustomerSince
df1.IsDelivery.value_counts()
df1.loc[df1["IsDelivery"] == 1,].isnull().sum()
df1.loc[df1["IsDelivery"] == 0,].isnull().sum()

# 11.147 invoices
df1['DocNumber'].nunique()

# Distribution of items per invoice
df1["DocNumber"].value_counts().hist(bins=38)
df1["DocNumber"].value_counts().describe()
sns.boxplot(df1["DocNumber"].value_counts())
# E em gráficos

a = df1["DocNumber"].value_counts().describe()

maxval = df1["DocNumber"].value_counts().value_counts(ascending=True).values.max()
linwhismin = a[4] - (a[6] - a[4])
linwhismax = a[6] + (a[6] - a[4])

a = df1["DocNumber"].value_counts().describe()

maxval = df1["DocNumber"].value_counts().value_counts(ascending=True).values.max()
linwhismin = a[4] - (a[6] - a[4])
linwhismax = a[6] + (a[6] - a[4])

sns.set()
sns.set_style("white")
fig, ax1 = plt.subplots(1, figsize=(16, 6))
ax1.scatter(df1["DocNumber"].value_counts().value_counts(ascending=True).index,
            df1["DocNumber"].value_counts().value_counts(ascending=True).values)

plt.plot([a[1], a[1]], [0, maxval / 4], color="red", linewidth=1)  # plot mean
plt.plot([linwhismin, linwhismin], [0, maxval / 8], color="black", linewidth=1)  # plot 1st whisker
plt.plot([a[4], a[4]], [0, maxval / 4], color="black", linewidth=1)  # plot 1st quart
plt.plot([a[5], a[5]], [0, maxval / 4], color="black", linewidth=1)  # plot median
plt.plot([a[6], a[6]], [0, maxval / 4], color="black", linewidth=1)  # plot 3rd quart

plt.plot([linwhismax, linwhismax], [0, maxval / 8], color="black", linewidth=1)  # plot 2nd whisker
# ax2.boxplot(df1["DocNumber"].value_counts())
ax1.set_title("Study of Items per Document", fontsize=20)

sns.despine()
plt.show()

# Number of customers in the database
df1["CustomerID"].nunique()
df1.loc[df1["IsDelivery"] == 0, "CustomerID"].value_counts()  # Customers that eat at restaurant have CustomerID = 0
# How many lines per customer that eat away?
df1.loc[df1["IsDelivery"] == 1, "CustomerID"].value_counts(ascending=False)
# How many lines per Invoice of IsDelivery = 1? - Comparing with overall distribution
df1["DocNumber"].value_counts().hist(bins=38)
# df1.loc[df1["IsDelivery"] == 0, "DocNumber"].value_counts().hist(bins=38)  # - 6281 customers
df1.loc[df1["IsDelivery"] == 1, "DocNumber"].value_counts().hist(bins=22)  # it seems that people that eat away have
# less variance in the amount of items they order - 4866 customers
# How many items per CustomerID and DocNumber
df1.loc[df1["IsDelivery"] == 1, ["CustomerID", "DocNumber"]].groupby(["CustomerID", "DocNumber"]).size()
# Distribution of number of orders per customer
df1.loc[df1["IsDelivery"] == 1,].drop_duplicates(subset=["DocNumber"]).CustomerID.value_counts().describe()
# Number of customers with more than one order
(df1.loc[df1["IsDelivery"] == 1,].drop_duplicates(subset=["DocNumber"]).CustomerID.value_counts() > 1).value_counts()
# Distribution of number of orders per customer - Pedro's graphic
a = df1.loc[df1["IsDelivery"] == 1,].drop_duplicates(subset=["DocNumber"]).CustomerID.value_counts().describe()
maxval = df1.loc[df1["IsDelivery"] == 1,].drop_duplicates(
    subset=["DocNumber"]).CustomerID.value_counts().value_counts().values.max()
linwhismin = a[4] - (a[6] - a[4])
linwhismax = a[6] + (a[6] - a[4])

sns.set()
sns.set_style("white")
fig, ax1 = plt.subplots(1, figsize=(16, 6))
# ax1.scatter(df1["DocNumber"].value_counts().value_counts(ascending=True).index,df1["DocNumber"].value_counts().value_counts(ascending=True).values )
ax1.scatter(
    df1.loc[df1["IsDelivery"] == 1,].drop_duplicates(subset=["DocNumber"]).CustomerID.value_counts().value_counts(
        ascending=True).index,
    df1.loc[df1["IsDelivery"] == 1,].drop_duplicates(subset=["DocNumber"]).CustomerID.value_counts().value_counts(
        ascending=True).values)

plt.plot([a[1], a[1]], [0, maxval / 4], color="red", linewidth=1)  # plot mean
plt.plot([linwhismin, linwhismin], [0, maxval / 8], color="black", linewidth=1)  # plot 1st whisker
plt.plot([a[4], a[4]], [0, maxval / 4], color="black", linewidth=1)  # plot 1st quart
plt.plot([a[5], a[5]], [0, maxval / 4], color="black", linewidth=1)  # plot median
plt.plot([a[6], a[6]], [0, maxval / 4], color="black", linewidth=1)  # plot 3rd quart

plt.plot([linwhismax, linwhismax], [0, maxval / 8], color="black", linewidth=1)  # plot 2nd whisker
# ax2.boxplot(df1["DocNumber"].value_counts())
ax1.set_title("Study of Number of Orders by customer", fontsize=20)

sns.despine()
plt.show()

a = df1.loc[df1["IsDelivery"] == 1,].drop_duplicates(subset=["DocNumber"]).CustomerID.value_counts()
print("number of customers with 1 order is: " + str(a.where(a == 1).count()) +
      " about " + str(round(a.where(a == 1).count() / 2315 * 100, 2)) + "% of customers")
print("number of customers with 2 or less orders is: " + str(a.where(a <= 2).count()) +
      " about " + str(round(a.where(a <= 2).count() / 2315 * 100, 2)) + "% of customers")
print("number of customers with 3 or less orders is: " + str(a.where(a <= 3).count()) +
      " about " + str(round(a.where(a <= 3).count() / 2315 * 100, 2)) + "% of customers")

# Analysing Product Family Qty and TotalAmount
familydf = pd.concat([df1.groupby(["ProductFamily"])["Qty"].sum(), df1.groupby(["ProductFamily"])["TotalAmount"].sum()],
                     axis=1, sort=False)
familydf

# In the same invoice we can have duplicates (e.g. ordering a Coke at the beginning and middle of a meal)
df1[(df1.DocNumber == 110044742018) & (df1.ProductDesignation == 'COKE')]

# How many Pax in a delivery order?
df1[df1.IsDelivery == 1].Pax.value_counts()  # always Pax=1 for IsDelivery=1 customers

# What are the existing Products?
df1.ProductDesignation.unique()

# Special Requests
SpRequestsNO = []
for product in list(df1.ProductDesignation.unique()):
    if ' NO ' in product:
        product_n = product.strip()
        SpRequestsNO.append(product_n)
SpRequestsNO

SpRequestsEXTRA = []
for product in list(df1.ProductDesignation.unique()):
    if ' EXTRA ' in product:
        product_n = product.strip()
        SpRequestsEXTRA.append(product_n)
SpRequestsEXTRA

# Delivery charges
df1[df1.ProductDesignation == 'DELIVERY CHARGE'].head()
df1.loc[df1.ProductDesignation == 'DELIVERY CHARGE', "IsDelivery"].value_counts()
df1.loc[(df1["IsDelivery"] == 1) & (df1["ProductDesignation"] == "DELIVERY CHARGE")].groupby("DocNumber").size()
# - 3892 invoices
df1.loc[(df1["IsDelivery"] == 1)].groupby("DocNumber").size()  # - 4866 invoices
# Not every IsDelivery=1 invoice has Delivery Charge


#----------------------
# FEATURE ENGINEERING
#----------------------

# CREATE WEEKDAY ATTRIBUTE

# Create attribute InvoiceDateHour_time -> turn InvoiceDateHour to type Datetime
df1['InvoiceDateHour_time'] =  pd.to_datetime(df1['InvoiceDateHour'], format='%Y-%m-%d %H:%M:%S.%f')
# Drop original column
df1.drop(columns=['InvoiceDateHour'],inplace=True)
# Create list with weekdays
weekDays = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
# Create new column with weekday name
df1['Weekday'] = df1.InvoiceDateHour_time.apply(lambda x: weekDays[x.weekday()])

# CREATE MEAL (LUNCH/DINNER) ATTRIBUTE

# Create attribute which only shows the hour of the invoice
df1['InvoiceHour'] = df1.InvoiceDateHour_time.apply(lambda x: x.hour)
# See distribution of hours, given if it is delivery or dine inn
df1.groupby(by='IsDelivery')['InvoiceHour'].value_counts()
# Create attribute
df1['Meal'] = df1.InvoiceHour.apply(lambda x: 'Lunch' if x<=17 and x>10 else 'Dinner')

# CREATE HOLIDAY ATTRIBUTE

# Create attribute without the hour, min, sec
df1['Date'] = df1.InvoiceDateHour_time.apply(lambda x: x.date())
# Create list with dates of holidays for 2018 in cyprus
holidays = ['01 Jan 2018','06 Jan 2018','19 Feb 2018','25 Mar 2018','01 Apr 2018','06 Apr 2018','07 Apr 2018','08 Apr 2018','09 Apr 2018','01 May 2018','28 May 2018','15 Aug 2018','01 Oct 2018','28 Oct 2018','24 Dec 2018','25 Dec 2018','26 Dec 2018','31 Dec 2018']
holidays_date = pd.to_datetime(holidays,format='%d %b %Y')
# Create final attribute
df1['Holiday'] = df1.Date.apply(lambda x: 1 if x in holidays_date else 0)

# CREATE SEASON

# Create function to get the season given a certain date
def get_season(data):
    
    if data >= pd.to_datetime('2018-03-20') and data < pd.to_datetime('2018-06-21'):
        return 'Spring'
    elif data >= pd.to_datetime('2018-06-21') and data < pd.to_datetime('2018-09-23'):
        return 'Summer'
    elif data >= pd.to_datetime('2018-09-23') and data < pd.to_datetime('2018-12-22'):
        return 'Autumn'
    else:
        return 'Winter'

# Create final attribute
df1['Season'] = df1.Date.apply(get_season)

# CREATE ATTRIBUTES RELATED TO WEATHER

# do web scrapping
# https://www.wunderground.com/history/monthly/cy/τύμβου/LCEN/date/2018-7

#----------------------
# MERGE ROWS
#----------------------

# Sum quantity and total amount for lines of an invoice which have the same product
qty_amount = df1[['DocNumber','ProductDesignation','Qty','TotalAmount']].groupby(['DocNumber','ProductDesignation']).sum()

# drop duplicates from the df (keep first) and create df with columns 'DocNumber','ProductDesignation','ProductFamily','IsDelivery'
family_deliv = df1.drop_duplicates(subset=['DocNumber','ProductDesignation'],keep= 'first')[['DocNumber','ProductDesignation','ProductFamily','IsDelivery']]

# merge the 2 df's created above
df_clean = family_deliv.merge(qty_amount, how='left',on=['DocNumber','ProductDesignation'])


#----------------------
# ONE HOT ENCODING
#----------------------




#--------------------
# ASSOCIATION RULES - simplest approach
#--------------------

df_clean_prod = df_clean[['DocNumber','ProductDesignation']]
df_clean_family = df_clean[['DocNumber','ProductFamily']]

pivot_family = pd.pivot_table(df_clean_family, index='DocNumber', columns='ProductFamily', aggfunc=lambda x: 1 if len(x)>0 else 0).fillna(0)
pivot_family.head()

pivot_prod = pd.pivot_table(df_clean_prod, index='DocNumber', columns='ProductDesignation', aggfunc=lambda x: 1 if len(x)>0 else 0).fillna(0)
pivot_prod.head()

# Rules supported in at least 5% of the transactions (more info at http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/)
frequent_itemsets_fam = apriori(pivot_family, min_support=0.05, use_colnames=True)
frequent_itemsets_prod = apriori(pivot_prod, min_support=0.05, use_colnames=True)

##### EXPLORE FREQUENT_ITEMSETS FOR FamilyProduct #####
# Add a column with the length
frequent_itemsets_fam['length'] = frequent_itemsets_fam['itemsets'].apply(lambda x: len(x))

# Length=2 and Support>=0.2
frequent_itemsets_fam[(frequent_itemsets_fam['length'] > 1) & (frequent_itemsets_fam['support'] >= 0.2)].sort_values(by='support', ascending=False)


##### EXPLORE FREQUENT_ITEMSETS FOR ProductDesignation #####
# Add a column with the length
frequent_itemsets_prod['length'] = frequent_itemsets_prod['itemsets'].apply(lambda x: len(x))

# Length=2 and Support>=0.2
frequent_itemsets_prod[(frequent_itemsets_prod['length'] > 1) & (frequent_itemsets_prod['support'] >= 0.2)].sort_values(by='support', ascending=False)

# Generate the association rules FOR PRODUCT - by confidence
rulesConfidence_prod = association_rules(frequent_itemsets_prod, metric="confidence", min_threshold=0.50)
rulesConfidence_prod.sort_values(by='confidence', ascending=False, inplace=True)
rulesConfidence_prod.head(10)

# Generate the association rules FOR FAMILY - by confidence
rulesConfidence_fam = association_rules(frequent_itemsets_fam, metric="confidence", min_threshold=0.50)
rulesConfidence_fam.sort_values(by='confidence', ascending=False, inplace=True)
rulesConfidence_fam.head(10)

# Generate the association rules FOR PRODUCT - by lift
rulesLift_prod = association_rules(frequent_itemsets_prod, metric="lift", min_threshold=1.5)
rulesLift_prod.sort_values(by='lift', ascending=False, inplace=True)
rulesLift_prod.head(10)

# Generate the association rules FOR FAMILY - by lift
rulesLift_fam = association_rules(frequent_itemsets_fam, metric="lift", min_threshold=1.5)
rulesLift_fam.sort_values(by='lift', ascending=False, inplace=True)
rulesLift_fam.head(10)