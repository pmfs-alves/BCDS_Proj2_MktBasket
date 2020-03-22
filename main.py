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

# If we want one single family for all Sushi
# SEE IF WE WANT THIS
df1.ProductFamily = df1.ProductFamily.apply(lambda x: 'SUSHI' if 'SUSHI' in x else x)

# If we want one single family for all indian food (maybe who eats indian, eats something else?)
# SEE IF WE WANT THIS
df1.ProductFamily = df1.ProductFamily.apply(lambda x: 'INDIAN' if 'IND' in x else x)


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
del family_deliv, holidays, holidays_date, qty_amount, weekDays, columns, df1, df, norm_cities

# Save df to csv file, to be explored in POWER BI
df_clean.to_csv(r'./data/data_cyprus.csv', index=False)

# ONE HOT ENCODING -----------------------------------------------------------------------------------------------------
columns = df_clean.columns.to_list()

# Para usarmos quando quisermos usar apenas algumas variáveis para as association rules
product_cols = ['ProductDesignation_' + prod for prod in list(df_clean.ProductDesignation.unique())]
#family_cols = list(df_clean.ProductFamily.unique())

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

# Get dummies and dropping useless/ redudant columns
df_clean_final = pd.get_dummies(df_clean, columns=['Season', 'Meal', 'Weekday', 'ProductFamily', 'ProductDesignation']).\
    drop(["ProductDesignation_DELIVERY CHARGE", "ProductDesignation_MINERAL WATER 1.5LT", "ProductDesignation_TSANTA"]+complementary, axis=1)

for i in ["ProductDesignation_DELIVERY CHARGE", "ProductDesignation_MINERAL WATER 1.5LT", "ProductDesignation_TSANTA"]+complementary:
    product_cols.remove(i)

# TODO: remove duplicated rules (e.g. A1={a,b,c,d} and A2={a,b,d} and C1=C2). Can be done by finding common subset.
#  Explore frozenset datatype https://docs.python.org/3.6/library/stdtypes.html#frozenset
# SE A1 NÃO FOR UMA BOA ASSOCIAÇÃO (LOW CONFIDENCE POR EXEMPLO), QUEREMOS REMOVÊ-LA E DEIXAR O A2 (SE O A2 FOR BOM)
# SE A1 FOR UMA BOA ASSOCIAÇÃO QUEREMOS REMOVER TUDO O QUE ESTÁ PARA BAIXO, CERT0? Temos que pensar nisto
#  TODO: plot the network with a sample of the rules but maintaining every single consequent
# TODO: see variables and think on what antecedents/ consequents would be interesting to explore
# TODO: color each node according to product family


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


def associations(invoice_df, min_support=0.05, min_confidence=0.5, low_lift=0.9, high_lift=3, export_path=None):
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

    low_lift_rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0)
    low_lift_rules = low_lift_rules[low_lift_rules.lift < low_lift]
    low_lift_rules.sort_values(by='lift', ascending=True, inplace=True)

    high_lift_rules = association_rules(frequent_itemsets, metric="lift", min_threshold=high_lift)
    high_lift_rules.sort_values(by='lift', ascending=False, inplace=True)

    if export_path:
        frequent_itemsets.to_excel(export_path + "frequent_itemsets.xlsx")
        relevant_rules.to_excel(export_path + "relevant_rules.xlsx")
        low_lift_rules.to_excel(export_path + "low_lift_rules.xlsx")
        high_lift_rules.to_excel(export_path + "high_lift_rules.xlsx")

    return frequent_itemsets, relevant_rules, low_lift_rules, high_lift_rules


# ASSOCIATION RULES - Product Dummies ----------------------------------------------------------------------------------
# CREATE INVOICE DF
# Filter Product columns
df_clean_final_products = df_clean_final[["DocNumber"] + product_cols]
# Clean columns names
df_clean_final_products.columns = list(map(lambda x: x.split('_')[-1].strip(), df_clean_final_products.columns))
# Group by Invoice. We only have binary variables, so max() will do the job
df_clean_product_dummies = df_clean_final_products.groupby('DocNumber').max()
df_clean_product_dummies.describe(include="all").transpose()

# ASSOCIATIONS
frequent_itemsets_products, relevant_rules_products, low_lift_rules_products, \
    high_lift_rules_products = associations(df_clean_product_dummies,
                                            low_lift=1,
                                            high_lift=2,
                                            export_path="./Outputs/Products_Total/")

# FREQUENT ITEMSETS
frequent_itemsets_products.shape[0]  # number of rows
frequent_itemsets_products.sort_values("support")

# RELEVANT RULES
relevant_rules_products.shape[0]  # number of rows
relevant_rules_products["support"].plot.box()
relevant_rules_products["confidence"].plot.box()
network_rules(relevant_rules_products, "all")

# LOW-LIFT RULES
low_lift_rules_products.shape[0]  # number of rows -> No rules with lift < 1
# network_rules(low_lift_rules_products, "all")

# HIGH-LIFT RULES
high_lift_rules_products.shape[0]  # number of rows
network_rules(high_lift_rules_products, "all")

del df_clean_product_dummies, frequent_itemsets_products, relevant_rules_products, low_lift_rules_products, \
    high_lift_rules_products, i, complementary

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
frequent_itemsets_products_delivery, relevant_rules_products_delivery, low_lift_rules_products_delivery, \
    high_lift_rules_products_delivery = associations(df_clean_product_dummies,
                                                     min_confidence=0.5,
                                                     low_lift=1,
                                                     high_lift=2,
                                                     export_path="./Outputs/Products_Delivery/")

# FREQUENT ITEMSETS
frequent_itemsets_products_delivery.shape[0]  # number of rows
frequent_itemsets_products_delivery.sort_values("support")

# RELEVANT RULES
relevant_rules_products_delivery.shape[0]  # number of rows
relevant_rules_products_delivery["support"].plot.box()
relevant_rules_products_delivery["confidence"].plot.box()
network_rules(relevant_rules_products_delivery, "all")

# LOW-LIFT RULES
low_lift_rules_products_delivery.shape[0]  # number of rows -> no rules with lift < 1
# network_rules(low_lift_rules_products_delivery, "all")

# HIGH-LIFT RULES
high_lift_rules_products_delivery.shape[0]  # number of rows
network_rules(high_lift_rules_products_delivery, "all")

del df_clean_product_dummies, frequent_itemsets_products_delivery, relevant_rules_products_delivery,\
    low_lift_rules_products_delivery, high_lift_rules_products_delivery

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
frequent_itemsets_products_dineinn, relevant_rules_products_dineinn, low_lift_rules_products_dineinn, \
    high_lift_rules_products_dineinn = associations(df_clean_product_dummies,
                                                    low_lift=1,
                                                    high_lift=2,
                                                    export_path="./Outputs/Products_DineInn/")

# FREQUENT ITEMSETS
frequent_itemsets_products_dineinn.shape[0]  # number of rows
frequent_itemsets_products_dineinn.sort_values("support")

# RELEVANT RULES
relevant_rules_products_dineinn.shape[0]  # number of rows
relevant_rules_products_dineinn["support"].plot.box()
relevant_rules_products_dineinn["confidence"].plot.box()
network_rules(relevant_rules_products_dineinn, "all")

# LOW-LIFT RULES
low_lift_rules_products_dineinn.shape[0]  # number of rows -> no rules with lift < 1
# network_rules(low_lift_rules_products_dineinn, "all")

# HIGH-LIFT RULES
high_lift_rules_products_dineinn.shape[0]  # number of rows
network_rules(high_lift_rules_products_dineinn, "all")

del df_clean_product_dummies, frequent_itemsets_products_dineinn, relevant_rules_products_dineinn, \
    low_lift_rules_products_dineinn, high_lift_rules_products_dineinn

# ASSOCIATION RULES - All Dummies --------------------------------------------------------------------------------------
# CREATE INVOICE DF
# Filter Product columns
df_clean_all_dummies = df_clean_final.drop(
    columns=['EmployeeID', 'IsDelivery', 'Pax', 'CustomerID', 'CustomerSince', 'Latitude', 'Longitude', 'CustomerCity',
             'Distance', 'Qty', 'TotalAmount', 'InvoiceDateHourTime', 'InvoiceHour'])
# Group by Invoice. We only have binary variables, so max() will do the job
df_clean_all_dummies = df_clean_all_dummies.groupby('DocNumber').max()
df_clean_all_dummies.describe(include="all").transpose()

# ASSOCIATIONS
frequent_itemsets_all, relevant_rules_all, low_lift_rules_all, \
    high_lift_rules_all = associations(df_clean_all_dummies,
                                       min_support=0.15,
                                       low_lift=0.95,
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

# LOW-LIFT RULES
low_lift_rules_all.shape[0]  # number of rows
network_rules(low_lift_rules_all, "all")

# HIGH-LIFT RULES
high_lift_rules_all.shape[0]  # number of rows
network_rules(high_lift_rules_all, 200)

del df_clean_all_dummies, frequent_itemsets_all, relevant_rules_all, low_lift_rules_all, high_lift_rules_all

# ASSOCIATION RULES - All Dummies Only Delivery ------------------------------------------------------------------------
# CREATE INVOICE DF
# Filter Product columns
df_clean_all_dummies = df_clean_final[df_clean_final.IsDelivery == 1].drop(
    columns=['EmployeeID', 'IsDelivery', 'Pax', 'CustomerID', 'CustomerSince', 'Latitude', 'Longitude', 'CustomerCity',
             'Distance', 'Qty', 'TotalAmount', 'InvoiceDateHourTime', 'InvoiceHour'])
# Group by Invoice. We only have binary variables, so max() will do the job
df_clean_all_dummies = df_clean_all_dummies.groupby('DocNumber').max()
df_clean_all_dummies.describe(include="all").transpose()

# ASSOCIATIONS
frequent_itemsets_all_delivery, relevant_rules_all_delivery, low_lift_rules_all_delivery, \
    high_lift_rules_all_delivery = associations(df_clean_all_dummies,
                                                min_support=0.15,
                                                low_lift=0.95,
                                                high_lift=2,
                                                export_path="./Outputs/All_Delivery/")

# FREQUENT ITEMSETS
frequent_itemsets_all_delivery.shape[0]  # number of rows
frequent_itemsets_all_delivery.sort_values("support")

# RELEVANT RULES
relevant_rules_all_delivery.shape[0]  # number of rows
relevant_rules_all_delivery["support"].plot.box()
relevant_rules_all_delivery["confidence"].plot.box()
network_rules(relevant_rules_all_delivery, 200)

# LOW-LIFT RULES
low_lift_rules_all_delivery.shape[0]  # number of rows -> 0 rows
# network_rules(low_lift_rules_all_delivery, "all")

# HIGH-LIFT RULES
high_lift_rules_all_delivery.shape[0]  # number of rows
network_rules(high_lift_rules_all_delivery, 200)

del df_clean_all_dummies, frequent_itemsets_all_delivery, relevant_rules_all_delivery, low_lift_rules_all_delivery, \
    high_lift_rules_all_delivery

# ASSOCIATION RULES - All Dummies Only Dine Inn ------------------------------------------------------------------------
# CREATE INVOICE DF
# Filter Product columns
df_clean_all_dummies = df_clean_final[df_clean_final.IsDelivery == 0].drop(
    columns=['EmployeeID', 'IsDelivery', 'Pax', 'CustomerID', 'CustomerSince', 'Latitude', 'Longitude', 'CustomerCity',
             'Distance', 'Qty', 'TotalAmount', 'InvoiceDateHourTime', 'InvoiceHour'])
# Group by Invoice. We only have binary variables, so max() will do the job
df_clean_all_dummies = df_clean_all_dummies.groupby('DocNumber').max()
df_clean_all_dummies.describe(include="all").transpose()

# ASSOCIATIONS
frequent_itemsets_all_dineinn, relevant_rules_all_dineinn, low_lift_rules_all_dineinn, \
    high_lift_rules_all_dineinn = associations(df_clean_all_dummies,
                                               min_support=0.15,
                                               low_lift=0.95,
                                               high_lift=2,
                                               export_path="./Outputs/All_DineInn/")

# FREQUENT ITEMSETS
frequent_itemsets_all_dineinn.shape[0]  # number of rows
frequent_itemsets_all_dineinn.sort_values("support")

# RELEVANT RULES
relevant_rules_all_dineinn.shape[0]  # number of rows
relevant_rules_all_dineinn["support"].plot.box()
relevant_rules_all_dineinn["confidence"].plot.box()
network_rules(relevant_rules_all_dineinn, 200)

# LOW-LIFT RULES
low_lift_rules_all_dineinn.shape[0]  # number of rows
network_rules(low_lift_rules_all_dineinn, "all")

# HIGH-LIFT RULES
high_lift_rules_all_dineinn.shape[0]  # number of rows
network_rules(high_lift_rules_all_dineinn, 200)

del df_clean_all_dummies, frequent_itemsets_all_dineinn, relevant_rules_all_dineinn, low_lift_rules_all_dineinn, \
    high_lift_rules_all_dineinn
