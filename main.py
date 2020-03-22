import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import networkx as nx


# Display output fully
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# ------------------------------------------------- Data Understanding -------------------------------------------------
df = pd.read_csv('./data/AsianRestaurant_Cyprus_2018.txt', sep=';')
# df2 = pd.read_csv('./data/AsianRestaurant_Cyprus_2018_partial.txt', sep=';')
df.dtypes

# remove spaces at the end/beginning of strings
df['CustomerCity'] = df.CustomerCity.str.strip()

# there are several ways to address the same city so we will normalize the cities names
# import file with each way and its normalized form + lat/long + distance to Nicosia
norm_cities = pd.read_excel(r'./data/CityChanges.xlsx')
# all cities in our dataframe are upper case
norm_cities['City'] = norm_cities['City'].str.upper()
# merge norm_cities with df
df1 = df.merge(norm_cities, how='left', left_on='CustomerCity', right_on='City')
# drop unnecessary columns and rename the normalized city column
df1.drop(columns=['CustomerCity', 'City'], inplace=True)
df1.rename(columns={'new_name': 'CustomerCity'}, inplace=True)

# correct TotalAmount data type
df1.TotalAmount = df1.TotalAmount.apply(lambda x: float(str(x.replace(',', '.'))))

# after taking a first look at the data we noticed there is more than 1 product family related to wine
# as they individually, are not so important, we decided to join them in an unique family 'WINE'
df1.ProductFamily.unique()
# replace
df1.ProductFamily = df1.ProductFamily.apply(lambda x: 'WINE' if 'WINE' in x else x)
# check if everything is ok
df1.ProductFamily.unique()

# We want one single family for all Sushi
df1.ProductFamily = df1.ProductFamily.apply(lambda x: 'SUSHI' if 'SUSHI' in x else x)

# We want one single family for all indian food (maybe who eats indian, eats something else?)
df1.ProductFamily = df1.ProductFamily.apply(lambda x: 'INDIAN' if 'IND' in x else x)

# --------------------EXPLORATORY ANALYSIS-----------
df1.describe(include="all")
# Checking missing values
# só customer city e customer since têm NULLS
# CustomerCity - 52 861
# CustomerSince - 54967
df1.isnull().sum()

# check that for delivery, the pax is always 1
df1[df1.IsDelivery == 1].Pax.value_counts()

# there are rows that are said to be from the restaurant, but the pax is 0,
# this might mean they are take away clients!
df1[df1.Pax == 0].IsDelivery.value_counts()  # 52 invoice lines corresponding to IsDelivery=0
df1[df1.Pax == 0].DocNumber.unique()  # 6 different invoices

# but there are no clients from the restaurant or take away who have an "account"
df1[(df1.IsDelivery == 0) & (df1.CustomerID != 0)]

# All customers that ate at the restaurant have missing values in CustomerCity and CustomerSince
df1.IsDelivery.value_counts()
df1.loc[df1["IsDelivery"] == 1,].isnull().sum()
df1.loc[df1["IsDelivery"] == 0,].isnull().sum()

# 11.147 invoices
df1['DocNumber'].nunique()

# Distribution of number of items per invoice
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
df1["CustomerID"].nunique() - 1  # -1 because of value 0 (unregisted customer)
df1.loc[df1["IsDelivery"] == 0, "CustomerID"].value_counts()  # Customers that eat at restaurant have CustomerID = 0

# How many lines per customer that eat away?
df1.loc[df1["IsDelivery"] == 1, "CustomerID"].value_counts(ascending=False)
# How many lines per Invoice of IsDelivery = 1? - Comparing with overall distribution
df1["DocNumber"].value_counts().hist(bins=38)
df1.loc[df1["IsDelivery"] == 1, "DocNumber"].value_counts().hist(bins=22)  # it seems that people that eat away have
# less variance in the amount of items they order - 4866 customers
# df1.loc[df1["IsDelivery"] == 0, "DocNumber"].value_counts().hist(bins=38)  # - 6281 customers

# How many items per CustomerID and DocNumber
df1.loc[df1["IsDelivery"] == 1, ["CustomerID", "DocNumber"]].groupby(["CustomerID", "DocNumber"]).size()

# Distribution of number of orders per customer
df1.loc[df1["IsDelivery"] == 1,].drop_duplicates(subset=["DocNumber"]).CustomerID.value_counts().describe()
# Number of customers with more than one order
(df1.loc[df1["IsDelivery"] == 1,].drop_duplicates(subset=["DocNumber"]).CustomerID.value_counts() > 1).value_counts()
# Distribution of number of orders per customer - Pedro's graphic
a = df1.loc[df1["IsDelivery"] == 1,].drop_duplicates(subset=["DocNumber"]).CustomerID.value_counts().describe()
maxval = df1.loc[df1["IsDelivery"] == 1,].drop_duplicates(subset=["DocNumber"]).CustomerID.value_counts(). \
    value_counts().values.max()
linwhismin = a[4] - (a[6] - a[4])
linwhismax = a[6] + (a[6] - a[4])

sns.set()
sns.set_style("white")
fig, ax1 = plt.subplots(1, figsize=(16, 6))
# ax1.scatter(df1["DocNumber"].value_counts().value_counts(ascending=True).index,df1["DocNumber"].value_counts().
# value_counts(ascending=True).values )
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
familydf = df1.loc[:, ["ProductFamily", "Qty", "TotalAmount"]].groupby(["ProductFamily"]).sum()
familydf

# In the same invoice we can have duplicates (e.g. ordering a Coke at the beginning and middle of a meal)
df1[(df1.DocNumber == 'TK0110000022018') & (df1.ProductDesignation == 'CARLSBERG 33CL')]
# We will be grouping these invoice lines into a single one

# What are the existing Products?
df1.ProductDesignation.unique()
# How many?
df1.ProductDesignation.nunique()  # 255

# See Special Requests with No
SpRequestsNO = []
for product in list(df1.ProductDesignation.unique()):
    if ' NO ' in product:
        product_n = product.strip()
        SpRequestsNO.append(product_n)
SpRequestsNO

# See Special Requests with Extra
SpRequestsEXTRA = []
for product in list(df1.ProductDesignation.unique()):
    if ' EXTRA ' in product:
        product_n = product.strip()
        SpRequestsEXTRA.append(product_n)
SpRequestsEXTRA

# Delivery charges
df1[df1.ProductDesignation == 'DELIVERY CHARGE'].head()
df1.loc[df1.ProductDesignation == 'DELIVERY CHARGE', "IsDelivery"].value_counts()  # Delivery Charge corresponds
# to IsDelivery=1
df1.loc[(df1["IsDelivery"] == 1) & (df1["ProductDesignation"] == "DELIVERY CHARGE")].groupby("DocNumber").size(). \
    value_counts()  # 3892 invoices with Delivery Charge of which 5 have duplicate lines of Delivery Charge
df1.loc[(df1["IsDelivery"] == 1)].groupby("DocNumber").size()  # - 4866 delivery invoices of which only 3892 had
# Delivery Charge - Not every IsDelivery=1 invoice has Delivery Charge

# -------------------------------------------------- Data Preparation --------------------------------------------------
# MERGE ROWS -----------------------------------------------------------------------------------------------------------

# Sum quantity and total amount for lines of an invoice which have the same product
qty_amount = df1[['DocNumber', 'ProductDesignation', 'Qty', 'TotalAmount']].groupby(
    ['DocNumber', 'ProductDesignation']).sum()

# get columns to use
columns = df1.columns.drop(["TotalAmount", "Qty"]).to_list()
# drop duplicates from the df (keep first) and create df with all columns except TotalAmount and Qty - drop 4770 rows
family_deliv = df1.drop_duplicates(subset=['DocNumber', 'ProductDesignation'], keep='first')[columns]

# merge the 2 df's created above
df_clean = family_deliv.merge(qty_amount, how='left', on=['DocNumber', 'ProductDesignation'])

del SpRequestsEXTRA, SpRequestsNO,a,ax1, familydf, fig, linwhismax, linwhismin, maxval,product,product_n
# FEATURE ENGINEERING --------------------------------------------------------------------------------------------------

# CREATE WEEKDAY ATTRIBUTE
# Create attribute InvoiceDateHour_time -> turn InvoiceDateHour to type Datetime
df_clean['InvoiceDateHourTime'] = pd.to_datetime(df_clean['InvoiceDateHour'], format='%Y-%m-%d %H:%M:%S.%f')
# Drop original column
df_clean.drop(columns=['InvoiceDateHour'], inplace=True)
# Create list with weekdays
weekDays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
# Create new column with weekday name
df_clean['Weekday'] = df_clean.InvoiceDateHourTime.apply(lambda x: weekDays[x.weekday()])

# CREATE MEAL (LUNCH/DINNER) ATTRIBUTE
# Create attribute which only shows the hour of the invoice
df_clean['InvoiceHour'] = df_clean.InvoiceDateHourTime.apply(lambda x: x.hour)
# See distribution of hours, given if it is delivery or dine in
pd.pivot_table(df_clean[["DocNumber", "InvoiceHour", "IsDelivery"]], index="InvoiceHour",
               columns="IsDelivery", aggfunc="count").plot(kind="bar")
# We can conclude that the peak of deliveries is before the one from dine ins (maybe because the kitchen closes
# somewhere around 10pm and the invoice is emitted when the food leaves the restaurant)
# Create attribute
df_clean['Meal'] = df_clean.InvoiceHour.apply(lambda x: 'Lunch' if 17 >= x > 10 else 'Dinner')

# CREATE HOLIDAY ATTRIBUTE
# Create attribute without the hour, min, sec
df_clean['Date'] = df_clean.InvoiceDateHourTime.apply(lambda x: x.date())
# Create list with dates of holidays for 2018 in cyprus - https://www.timeanddate.com/holidays/cyprus/2018
holidays = ['01 Jan 2018', '06 Jan 2018', '19 Feb 2018', '25 Mar 2018', '01 Apr 2018', '06 Apr 2018', '07 Apr 2018',
            '08 Apr 2018', '09 Apr 2018', '01 May 2018', '28 May 2018', '15 Aug 2018', '01 Oct 2018', '28 Oct 2018',
            '24 Dec 2018', '25 Dec 2018', '26 Dec 2018', '31 Dec 2018']
holidays_date = pd.to_datetime(holidays, format='%d %b %Y')
# Create final attribute
df_clean['Holiday'] = df_clean.Date.apply(lambda x: 1 if x in holidays_date else 0)


# CREATE SEASON
# Create function to get the season given a certain date
def get_season(data):
    if pd.to_datetime('2018-03-20') <= data < pd.to_datetime('2018-06-21'):
        return 'Spring'
    elif pd.to_datetime('2018-06-21') <= data < pd.to_datetime('2018-09-23'):
        return 'Summer'
    elif pd.to_datetime('2018-09-23') <= data < pd.to_datetime('2018-12-22'):
        return 'Autumn'
    else:
        return 'Winter'


# Create final attribute
df_clean['Season'] = df_clean.Date.apply(get_season)
# Drop unnecessary column Date
df_clean.drop(columns='Date', inplace=True)

# CREATE WEEKEND ATTRIBUTE
df_clean['Weekend'] = df_clean.Weekday.apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)

# CREATE ATTRIBUTES RELATED TO WEATHER
# do web scrapping - https://www.wunderground.com/history/monthly/cy/τύμβου/LCEN/date/2018-7

# Delete useless variables
del holidays, holidays_date, qty_amount, weekDays, columns, df1, df, norm_cities, family_deliv

# Save df to csv file, to be explored in POWER BI
df_clean.to_csv(r'./data/data_cyprus.csv', index=False)

# ONE HOT ENCODING AND DATA PREPARATION -----------------------------------------------------------------------------------------------------
# save all column names
columns = df_clean.columns.to_list()

# Para usarmos quando quisermos usar apenas algumas variáveis para as association rules
product_cols = ['ProductDesignation_' + prod for prod in list(df_clean.ProductDesignation.unique())]
family_cols = ['ProductFamily_' + fam for fam in list(df_clean.ProductFamily.unique())]
# create list with dummy columns from Season
season_cols = ['Season_' + prod for prod in list(df_clean.Season.unique())]
# create list with dummy columns from Meal
meal_cols = ['Meal_' + prod for prod in list(df_clean.Meal.unique())]
# create list with dummy columns from Weekday
weekday_cols = ['Weekday_' + day for day in list(df_clean.Weekday.unique())]

# change type of the categorical new attributes
df_clean['Season'] = df_clean['Season'].astype('category')
df_clean['Meal'] = df_clean['Meal'].astype('category')
df_clean['Weekday'] = df_clean['Weekday'].astype('category')

# get all complementary products
# we will remove them from the analysis as people don't choose them, they are offered
complementary = list(df_clean.ProductDesignation[(df_clean.ProductDesignation.str.contains('COMPLIMENTARY')) | (df_clean.ProductDesignation.str.contains('COMPLEMENTARY'))].unique())
complementary = ['ProductDesignation_'+ elem for elem in complementary]

#check that santa only appears in delivery invoices (tsanta means bag)
df_clean.ProductDesignation.str.contains('TSANTA').sum() # 1749
df_clean[df_clean.IsDelivery==0].ProductDesignation.str.contains('TSANTA').sum() # 0
df_clean[df_clean.IsDelivery==1].ProductDesignation.str.contains('TSANTA').sum() # 1749
# we'll remove tsanta from the analysis as we already saw it means bags and is used for deliveries and we consider it doesn't add any valuable info
# we'll also drop Water as it is bought in almost all invoices from Dine Inn (it is not bough in Deliveries as we saw in PowerBI)
# we'll drop the delivery charge as it is an extra charged for deliveries

# Get dummies and dropping useless/ redudant columns
df_clean_final = pd.get_dummies(df_clean, columns=['Season', 'Meal', 'Weekday', 'ProductFamily', 'ProductDesignation']).\
    drop(["ProductDesignation_DELIVERY CHARGE", "ProductDesignation_MINERAL WATER 1.5LT", "ProductDesignation_TSANTA"]+complementary, axis=1)

for i in ["ProductDesignation_DELIVERY CHARGE", "ProductDesignation_MINERAL WATER 1.5LT", "ProductDesignation_TSANTA"]+complementary:
    product_cols.remove(i)

# create function to create a network graph
def network_rules(rulesdf, nrules=100, save_path=None):
    """Plot a basic network graph of the confidence rules. Arrows point to consequent.
    - rulesdf: association rules dataframe
    - nrules: how many rules to plot (int or "all" - not advisable for more than 500 rules)
    - save_path: path to save figure
    """
    # Create a copy of the rules and transform the frozensets to strings
    rulesToPlot = rulesdf.copy(deep=True)
    rulesToPlot['LHS'] = [','.join(list(x)) for x in rulesToPlot['antecedents']]
    rulesToPlot['RHS'] = [','.join(list(x)) for x in rulesToPlot['consequents']]
    # Remove duplicated itemsets i.e. A->C and C->A
    rulesToPlot['sortedRow'] = [sorted([a, b]) for a, b in zip(rulesToPlot.LHS, rulesToPlot.RHS)]
    rulesToPlot['sortedRow'] = rulesToPlot['sortedRow'].astype(str)
    #rulesToPlot.drop_duplicates(subset=['sortedRow'], inplace=True)
    # Plot
    if type(nrules) == int:
        rulesToPlot = rulesToPlot[:nrules]
    fig = plt.figure(figsize=(20, 20))
    rulesToPlot["LHS"] = rulesToPlot["LHS"].str.lower()
    rulesToPlot["RHS"] = rulesToPlot["RHS"].str.lower()
    G = nx.from_pandas_edgelist(rulesToPlot, 'LHS', 'RHS', create_using=nx.MultiDiGraph())
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, with_labels=False, node_size=30, node_color='r', alpha=0.9, seed=1234)
    nx.draw_networkx_edges(G, pos, arrows=True, arrow_style="fancy", edge_color="darkgrey", arrow_size=80,
                           alpha=0.8, seed=1234)
    label_pos = {k: v + [0.002, 0.004] for k, v in pos.items()}
    nx.draw_networkx_labels(G, label_pos, font_size=8)
    plt.axis('equal')
    if save_path:
        fig.savefig(save_path)
    else:
        fig.show()


def associations(invoice_df, min_support=0.05, min_confidence=0.5, high_lift=2, export_path=None):
    """
    Applies the apriori algorithm to find the most frequent itemsets and afterwards looks for the most relevant,
    low lift and high lift rules.
    - invoice_df: dataframe object with rows representing an invoice and binary columns.
    - min_support: minimum probability of an itemset to be considered frequent. Default 0.05.
    - min_confidence: minimum confidence of an association rule to be considered relevant. Default 0.5.
    - low_lift: maximum lift of an association rule to be considered as potential substitutes. Default 0.9.
    Should be <1.
    - high_lift: minimum lift of an association rule to be considered as potential complementary. Default 3.
    Should be >1.
    - export_path: path to export (if passed) most frequent itemsets, most relevant rules, low lift rules and
    high lift rules.
    """
    frequent_itemsets = apriori(invoice_df, min_support=min_support, use_colnames=True)
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

    relevant_rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    relevant_rules.sort_values(by='confidence', ascending=False, inplace=True)

    high_lift_rules = association_rules(frequent_itemsets, metric="lift", min_threshold=high_lift)
    high_lift_rules.sort_values(by='lift', ascending=False, inplace=True)

    if export_path:
        frequent_itemsets.to_excel(export_path + "frequent_itemsets.xlsx")
        relevant_rules.to_excel(export_path + "relevant_rules.xlsx")
        high_lift_rules.to_excel(export_path + "high_lift_rules.xlsx")

    return frequent_itemsets, relevant_rules, high_lift_rules

del df_clean,i,complementary,columns

# ASSOCIATION RULES - All Product Dummies ----------------------------------------------------------------------------------
# CREATE INVOICE DF
# Filter Product columns
df_clean_final_products = df_clean_final[["DocNumber"] + product_cols]
# Clean columns names
df_clean_final_products.columns = list(map(lambda x: x.split('_')[-1].strip(), df_clean_final_products.columns))

# Group by Invoice. We only have binary variables, so max() will do the job
df_clean_product_dummies = df_clean_final_products.groupby('DocNumber').max()
df_clean_product_dummies.describe(include="all").transpose()

# ASSOCIATIONS
frequent_itemsets_products, relevant_rules_products, \
    high_lift_rules_products = associations(df_clean_product_dummies,
                                            export_path="./Outputs/Products_Total/")

# FREQUENT ITEMSETS
frequent_itemsets_products.shape[0]  # number of rows
frequent_itemsets_products.sort_values("support")

# RELEVANT RULES
relevant_rules_products.shape[0]  # number of rows
relevant_rules_products["support"].plot.box()
relevant_rules_products["confidence"].plot.box()
network_rules(relevant_rules_products, "all")

# HIGH-LIFT RULES
high_lift_rules_products.shape[0]  # number of rows
network_rules(high_lift_rules_products, "all")

# SUBSTITUTE PRODUCTS
# To see substitute products we decided to put a minimum support of 0.001 and then check the rules with a lower lift
# We chose a very low support because if they are substitutes they should not be seen together.
#  We will not carry this analysis to the next attempts because this piece of code takes a long time to run and we don't have enough computer power
# The substitute products we found and refered in the report were found here
# Uncomment the next lines to get the substitutes analysis (takes a lot of time to run)

#frequent_itemsets = apriori(df_clean_product_dummies, min_support=0.001, use_colnames=True)
#low_lift_rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0)


del df_clean_product_dummies, frequent_itemsets_products, relevant_rules_products, high_lift_rules_products, df_clean_final_products

# ASSOCIATION RULES - Product Dummies Only Delivery --------------------------------------------------------------------
# CREATE INVOICE DF
# Filter Product columns
df_clean_final_products = df_clean_final.loc[df_clean_final.IsDelivery == 1, ["DocNumber"] + product_cols]
# Clean columns names
df_clean_final_products.columns = list(map(lambda x: x.split('_')[-1].strip(), df_clean_final_products.columns))
# Group by Invoice. We only have binary variables, so max() will do the job
df_clean_product_dummies = df_clean_final_products.groupby('DocNumber').max()
df_clean_product_dummies.describe(include="all").transpose()

# ASSOCIATIONS
frequent_itemsets_products_delivery, relevant_rules_products_delivery, \
    high_lift_rules_products_delivery = associations(df_clean_product_dummies,
                                                     export_path="./Outputs/Products_Delivery/")

# FREQUENT ITEMSETS
frequent_itemsets_products_delivery.shape[0]  # number of rows
frequent_itemsets_products_delivery.sort_values("support")

# RELEVANT RULES
relevant_rules_products_delivery.shape[0]  # number of rows
relevant_rules_products_delivery["support"].plot.box()
relevant_rules_products_delivery["confidence"].plot.box()
network_rules(relevant_rules_products_delivery, "all")

# HIGH-LIFT RULES
high_lift_rules_products_delivery.shape[0]  # number of rows
network_rules(high_lift_rules_products_delivery, "all")

del df_clean_product_dummies, frequent_itemsets_products_delivery, relevant_rules_products_delivery,\
 high_lift_rules_products_delivery, df_clean_final_products

# ASSOCIATION RULES - Product Dummies DineInn --------------------------------------------------------------------------
# CREATE INVOICE DF
# Filter Product columns
df_clean_final_products = df_clean_final.loc[df_clean_final.IsDelivery == 0, ["DocNumber"] + product_cols]
# Clean columns names
df_clean_final_products.columns = list(map(lambda x: x.split('_')[-1].strip(), df_clean_final_products.columns))
# Group by Invoice. We only have binary variables, so max() will do the job
df_clean_product_dummies = df_clean_final_products.groupby('DocNumber').max()
df_clean_product_dummies.describe(include="all").transpose()

# ASSOCIATIONS
frequent_itemsets_products_dineinn, relevant_rules_products_dineinn, \
    high_lift_rules_products_dineinn = associations(df_clean_product_dummies,
                                                    export_path="./Outputs/Products_DineInn/")

# FREQUENT ITEMSETS
frequent_itemsets_products_dineinn.shape[0]  # number of rows
frequent_itemsets_products_dineinn.sort_values("support")

# RELEVANT RULES
relevant_rules_products_dineinn.shape[0]  # number of rows
relevant_rules_products_dineinn["support"].plot.box()
relevant_rules_products_dineinn["confidence"].plot.box()
network_rules(relevant_rules_products_dineinn, "all")

# HIGH-LIFT RULES
high_lift_rules_products_dineinn.shape[0]  # number of rows
network_rules(high_lift_rules_products_dineinn, "all")

del df_clean_product_dummies, frequent_itemsets_products_dineinn, relevant_rules_products_dineinn, \
    high_lift_rules_products_dineinn, df_clean_final_products

# ASSOCIATION RULES - All Dummies --------------------------------------------------------------------------------------
# CREATE INVOICE DF
# Filter Product columns
df_clean_all_dummies = df_clean_final.drop(
    columns=['EmployeeID', 'IsDelivery', 'Pax', 'CustomerID', 'CustomerSince', 'Latitude', 'Longitude', 'CustomerCity',
             'Distance', 'Qty', 'TotalAmount', 'InvoiceDateHourTime', 'InvoiceHour'])
# Clean columns names
df_clean_all_dummies.columns = list(map(lambda x: x.split('_')[-1].strip(), df_clean_all_dummies.columns))

# Group by Invoice. We only have binary variables, so max() will do the job
df_clean_all_dummies = df_clean_all_dummies.groupby('DocNumber').max()
df_clean_all_dummies.describe(include="all").transpose()

# ASSOCIATIONS
frequent_itemsets_all, relevant_rules_all,  \
    high_lift_rules_all = associations(df_clean_all_dummies,
                                       min_support=0.15,
                                       high_lift=2,
                                       export_path="./Outputs/All_Total/")

# FREQUENT ITEMSETS
frequent_itemsets_all.shape[0]  # number of rows
frequent_itemsets_all.sort_values("support")

# RELEVANT RULES
relevant_rules_all.shape[0]  # number of rows
relevant_rules_all["support"].plot.box()
relevant_rules_all["confidence"].plot.box()
network_rules(relevant_rules_all, 200)

# HIGH-LIFT RULES
high_lift_rules_all.shape[0]  # number of rows
network_rules(high_lift_rules_all, 200)

del df_clean_all_dummies, frequent_itemsets_all, relevant_rules_all, high_lift_rules_all



# ASSOCIATION RULES - Product + Meal --------------------------------------------------------------------------------------
# CREATE INVOICE DF
# Filter Product and meal columns
df_clean_prod_meal = df_clean_final.drop(
    columns=['EmployeeID', 'IsDelivery', 'Pax', 'CustomerID', 'CustomerSince', 'Latitude', 'Longitude', 'CustomerCity',
             'Distance', 'Qty', 'TotalAmount', 'InvoiceDateHourTime', 'InvoiceHour','Weekend','Holiday']+season_cols+family_cols+weekday_cols)

# Clean columns names
df_clean_prod_meal.columns = list(map(lambda x: x.split('_')[-1].strip(), df_clean_prod_meal.columns))

# Group by Invoice. We only have binary variables, so max() will do the job
df_clean_prod_meal = df_clean_prod_meal.groupby('DocNumber').max()
df_clean_prod_meal.describe(include="all").transpose()

# as the 2 classes are so unbalanced we have to split the dataset and check the rules separately
df_clean_prod_meal.Lunch[df_clean_prod_meal.Lunch==1].sum() # 870
df_clean_prod_meal.Dinner[df_clean_prod_meal.Dinner==1].sum() # 10 277

# ASSOCIATIONS LUNCH
df_clean_prod_lunch = df_clean_prod_meal[df_clean_prod_meal.Lunch==1]
df_clean_prod_lunch.drop(columns=['Lunch','Dinner'], inplace=True)

frequent_itemsets_all, relevant_rules_lunch, \
    high_lift_rules_all = associations(df_clean_prod_lunch,
                                       export_path="./Outputs/All_Total/")

# FREQUENT ITEMSETS
frequent_itemsets_all.shape[0]  # number of rows
frequent_itemsets_all.sort_values("support")

# RELEVANT RULES
relevant_rules_lunch.shape[0]  # number of rows
relevant_rules_lunch["support"].plot.box()
relevant_rules_lunch["confidence"].plot.box()
network_rules(relevant_rules_lunch, 200)

# HIGH-LIFT RULES
high_lift_rules_all.shape[0]  # number of rows
network_rules(high_lift_rules_all, 200)


# ASSOCIATIONS DINNER

df_clean_prod_dinner = df_clean_prod_meal[df_clean_prod_meal.Dinner==1]
df_clean_prod_dinner.drop(columns=['Lunch','Dinner'], inplace=True)

frequent_itemsets_all, relevant_rules_dinner, \
    high_lift_rules_all = associations(df_clean_prod_dinner,
                                       export_path="./Outputs/All_Total/")

# FREQUENT ITEMSETS
frequent_itemsets_all.shape[0]  # number of rows
frequent_itemsets_all.sort_values("support")

# RELEVANT RULES
relevant_rules_dinner.shape[0]  # number of rows
relevant_rules_dinner["support"].plot.box()
relevant_rules_dinner["confidence"].plot.box()
network_rules(relevant_rules_dinner, 200)

# HIGH-LIFT RULES
high_lift_rules_all.shape[0]  # number of rows
network_rules(high_lift_rules_all, 200)

del df_clean_prod_dinner, df_clean_prod_lunch, df_clean_prod_meal, frequent_itemsets_all, relevant_rules_lunch, relevant_rules_dinner, high_lift_rules_all

# ASSOCIATION RULES - Product + Weekend --------------------------------------------------------------------------------------
# CREATE INVOICE DF
# Filter Product and meal columns
df_clean_prod_Weekend = df_clean_final.drop(
    columns=['EmployeeID', 'IsDelivery', 'Pax', 'CustomerID', 'CustomerSince', 'Latitude', 'Longitude', 'CustomerCity',
             'Distance', 'Qty', 'TotalAmount', 'InvoiceDateHourTime', 'InvoiceHour','Holiday']+season_cols+family_cols+weekday_cols+meal_cols)

# Clean columns names
df_clean_prod_Weekend.columns = list(map(lambda x: x.split('_')[-1].strip(), df_clean_prod_Weekend.columns))

# Group by Invoice. We only have binary variables, so max() will do the job
df_clean_prod_Weekend = df_clean_prod_Weekend.groupby('DocNumber').max()
df_clean_prod_Weekend.describe(include="all").transpose()

# as the 2 classes are so unbalanced we have to split the dataset and check the rules separately
df_clean_prod_Weekend.Weekend.value_counts() # 0 - 7465; 1 - 3682

# ASSOCIATIONS Week
df_clean_prod_week = df_clean_prod_Weekend[df_clean_prod_Weekend.Weekend==0]
df_clean_prod_week.drop(columns=['Weekend'], inplace=True)

frequent_itemsets_all, relevant_rules_week,  \
    high_lift_rules_all = associations(df_clean_prod_week,
                                       export_path="./Outputs/All_Total/")

# FREQUENT ITEMSETS
frequent_itemsets_all.shape[0]  # number of rows
frequent_itemsets_all.sort_values("support")

# RELEVANT RULES
relevant_rules_week.shape[0]  # number of rows
relevant_rules_week["support"].plot.box()
relevant_rules_week["confidence"].plot.box()
network_rules(relevant_rules_week, 200)

# HIGH-LIFT RULES
high_lift_rules_all.shape[0]  # number of rows
network_rules(high_lift_rules_all, 200)


# ASSOCIATIONS weekend

df_clean_prod_weekend = df_clean_prod_Weekend[df_clean_prod_Weekend.Weekend==1]
df_clean_prod_weekend.drop(columns=['Weekend'], inplace=True)

frequent_itemsets_all, relevant_rules_weekend, \
    high_lift_rules_all = associations(df_clean_prod_weekend,
                                       export_path="./Outputs/All_Total/")

# FREQUENT ITEMSETS
frequent_itemsets_all.shape[0]  # number of rows
frequent_itemsets_all.sort_values("support")

# RELEVANT RULES
relevant_rules_weekend.shape[0]  # number of rows
relevant_rules_weekend["support"].plot.box()
relevant_rules_weekend["confidence"].plot.box()
network_rules(relevant_rules_weekend, 200)

# HIGH-LIFT RULES
high_lift_rules_all.shape[0]  # number of rows
network_rules(high_lift_rules_all, 200)

del df_clean_prod_weekend, df_clean_prod_Weekend, df_clean_prod_week, frequent_itemsets_all, relevant_rules_weekend, relevant_rules_week, high_lift_rules_all

# ASSOCIATION RULES - Product + Holiday --------------------------------------------------------------------------------------
# CREATE INVOICE DF
# Filter Product and meal columns
df_clean_prod_Holiday = df_clean_final.drop(
    columns=['EmployeeID', 'IsDelivery', 'Pax', 'CustomerID', 'CustomerSince', 'Latitude', 'Longitude', 'CustomerCity',
             'Distance', 'Qty', 'TotalAmount', 'InvoiceDateHourTime', 'InvoiceHour','Weekend']+season_cols+family_cols+weekday_cols+meal_cols)

# Clean columns names
df_clean_prod_Holiday.columns = list(map(lambda x: x.split('_')[-1].strip(), df_clean_prod_Holiday.columns))

# Group by Invoice. We only have binary variables, so max() will do the job
df_clean_prod_Holiday = df_clean_prod_Holiday.groupby('DocNumber').max()
df_clean_prod_Holiday.describe(include="all").transpose()

# as the 2 classes are so unbalanced we have to split the dataset and check the rules separately
df_clean_prod_Holiday.Holiday.value_counts() # 0 - 10 754; 1 - 393

# ASSOCIATIONS holiday
df_clean_prod_holiday = df_clean_prod_Holiday[df_clean_prod_Holiday.Holiday==1]
df_clean_prod_holiday.drop(columns=['Holiday'], inplace=True)

frequent_itemsets_all, relevant_rules_holiday, \
    high_lift_rules_all = associations(df_clean_prod_holiday,
                                       export_path="./Outputs/All_Total/")

# FREQUENT ITEMSETS
frequent_itemsets_all.shape[0]  # number of rows
frequent_itemsets_all.sort_values("support")

# RELEVANT RULES
relevant_rules_holiday.shape[0]  # number of rows
relevant_rules_holiday["support"].plot.box()
relevant_rules_holiday["confidence"].plot.box()
network_rules(relevant_rules_holiday, 200)

# HIGH-LIFT RULES
high_lift_rules_all.shape[0]  # number of rows
network_rules(high_lift_rules_all, 200)


# ASSOCIATIONS not holiday

df_clean_prod_notholiday = df_clean_prod_Holiday[df_clean_prod_Holiday.Holiday==0]
df_clean_prod_notholiday.drop(columns=['Holiday'], inplace=True)

frequent_itemsets_all, relevant_rules_notholiday, \
    high_lift_rules_all = associations(df_clean_prod_notholiday,
                                       export_path="./Outputs/All_Total/")

# FREQUENT ITEMSETS
frequent_itemsets_all.shape[0]  # number of rows
frequent_itemsets_all.sort_values("support")

# RELEVANT RULES
relevant_rules_notholiday.shape[0]  # number of rows
relevant_rules_notholiday["support"].plot.box()
relevant_rules_notholiday["confidence"].plot.box()
network_rules(relevant_rules_notholiday, 200)

# HIGH-LIFT RULES
high_lift_rules_all.shape[0]  # number of rows
network_rules(high_lift_rules_all, 200)

del df_clean_prod_holiday, df_clean_prod_Holiday, df_clean_prod_notholiday, frequent_itemsets_all, high_lift_rules_all, relevant_rules_holiday, relevant_rules_notholiday


# ASSOCIATION RULES - Product + Season --------------------------------------------------------------------------------------
# CREATE INVOICE DF
# Filter Product and meal columns
df_clean_prod_Season = df_clean_final.drop(
    columns=['EmployeeID', 'IsDelivery', 'Pax', 'CustomerID', 'CustomerSince', 'Latitude', 'Longitude', 'CustomerCity',
             'Distance', 'Qty', 'TotalAmount', 'InvoiceDateHourTime', 'InvoiceHour','Weekend', 'Holiday']+family_cols+weekday_cols+meal_cols)

# Clean columns names
df_clean_prod_Season.columns = list(map(lambda x: x.split('_')[-1].strip(), df_clean_prod_Season.columns))

# Group by Invoice. We only have binary variables, so max() will do the job
df_clean_prod_Season = df_clean_prod_Season.groupby('DocNumber').max()
df_clean_prod_Season.describe(include="all").transpose()

# check the observations per class
df_clean_prod_Season.Autumn[df_clean_prod_Season.Autumn==1].sum() # 2904
df_clean_prod_Season.Winter[df_clean_prod_Season.Winter==1].sum() # 3079
df_clean_prod_Season.Spring[df_clean_prod_Season.Spring==1].sum() # 2636
df_clean_prod_Season.Summer[df_clean_prod_Season.Summer==1].sum() # 2528

# ASSOCIATIONS 

frequent_itemsets_all, relevant_rules_season, \
    high_lift_rules_all = associations(df_clean_prod_Season,
                                       export_path="./Outputs/All_Total/")

# FREQUENT ITEMSETS
frequent_itemsets_all.shape[0]  # number of rows
frequent_itemsets_all.sort_values("support")

# RELEVANT RULES
relevant_rules_season.shape[0]  # number of rows
relevant_rules_season["support"].plot.box()
relevant_rules_season["confidence"].plot.box()
network_rules(relevant_rules_season, 200)

# HIGH-LIFT RULES
high_lift_rules_all.shape[0]  # number of rows
network_rules(high_lift_rules_all, 200)

del df_clean_prod_Season, frequent_itemsets_all, high_lift_rules_all, relevant_rules_season


# ASSOCIATION RULES - Product + Weekdays --------------------------------------------------------------------------------------
# CREATE INVOICE DF
# Filter Product and meal columns
df_clean_prod_Weekdays = df_clean_final.drop(
    columns=['EmployeeID', 'IsDelivery', 'Pax', 'CustomerID', 'CustomerSince', 'Latitude', 'Longitude', 'CustomerCity',
             'Distance', 'Qty', 'TotalAmount', 'InvoiceDateHourTime', 'InvoiceHour','Weekend', 'Holiday']+family_cols+season_cols+meal_cols)

# Clean columns names
df_clean_prod_Weekdays.columns = list(map(lambda x: x.split('_')[-1].strip(), df_clean_prod_Weekdays.columns))

# Group by Invoice. We only have binary variables, so max() will do the job
df_clean_prod_Weekdays = df_clean_prod_Weekdays.groupby('DocNumber').max()
df_clean_prod_Weekdays.describe(include="all").transpose()

# check the observations per class
df_clean_prod_Weekdays.Monday[df_clean_prod_Weekdays.Monday==1].sum() # 978
df_clean_prod_Weekdays.Tuesday[df_clean_prod_Weekdays.Tuesday==1].sum() # 906
df_clean_prod_Weekdays.Wednesday[df_clean_prod_Weekdays.Wednesday==1].sum() # 2213
df_clean_prod_Weekdays.Thursday[df_clean_prod_Weekdays.Thursday==1].sum() # 1123
df_clean_prod_Weekdays.Friday[df_clean_prod_Weekdays.Friday==1].sum() # 2245
df_clean_prod_Weekdays.Saturday[df_clean_prod_Weekdays.Saturday==1].sum() # 2395
df_clean_prod_Weekdays.Sunday[df_clean_prod_Weekdays.Sunday==1].sum() # 1287

# ASSOCIATIONS  wednesday
df_clean_prod_wednesday = df_clean_prod_Weekdays[df_clean_prod_Weekdays.Wednesday==1]
weekday_cols_noprefix = [i.split('_')[-1] for i in weekday_cols]
df_clean_prod_wednesday.drop(columns=weekday_cols_noprefix, inplace=True)

frequent_itemsets_all, relevant_rules_wednesday, \
    high_lift_rules_all = associations(df_clean_prod_wednesday,
                                       export_path="./Outputs/All_Total/")

# FREQUENT ITEMSETS
frequent_itemsets_all.shape[0]  # number of rows
frequent_itemsets_all.sort_values("support")

# RELEVANT RULES
relevant_rules_wednesday.shape[0]  # number of rows
relevant_rules_wednesday["support"].plot.box()
relevant_rules_wednesday["confidence"].plot.box()
network_rules(relevant_rules_wednesday, 200)

# HIGH-LIFT RULES
high_lift_rules_all.shape[0]  # number of rows
network_rules(high_lift_rules_all, 200)

del df_clean_prod_wednesday, relevant_rules_wednesday, high_lift_rules_all, frequent_itemsets_all

# ASSOCIATIONS  friday
df_clean_prod_friday = df_clean_prod_Weekdays[df_clean_prod_Weekdays.Friday==1]
weekday_cols_noprefix = [i.split('_')[-1] for i in weekday_cols]
df_clean_prod_friday.drop(columns=weekday_cols_noprefix, inplace=True)

frequent_itemsets_all, relevant_rules_friday, \
    high_lift_rules_all = associations(df_clean_prod_friday,
                                       export_path="./Outputs/All_Total/")

# FREQUENT ITEMSETS
frequent_itemsets_all.shape[0]  # number of rows
frequent_itemsets_all.sort_values("support")

# RELEVANT RULES
relevant_rules_friday.shape[0]  # number of rows
relevant_rules_friday["support"].plot.box()
relevant_rules_friday["confidence"].plot.box()
network_rules(relevant_rules_friday, 200)

# HIGH-LIFT RULES
high_lift_rules_all.shape[0]  # number of rows
network_rules(high_lift_rules_all, 200)


del df_clean_prod_friday, relevant_rules_friday, high_lift_rules_all, frequent_itemsets_all, df_clean_prod_Weekdays