import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
