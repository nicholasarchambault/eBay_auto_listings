#!/usr/bin/env python
# coding: utf-8

# # Exploring Car Sales Data From eBay
# 
# by Nicholas Archambault
# 
# The following project analyzes 50,000 listings from 'eBay Klenanzeigen', a classifieds section from the German eBay website.  We'll attempt to explore and address a number of questions about the sales success of different types of cars in this database.

# The data are listed according to the following attributes:
# 
# * `dateCrawled` - When this ad was first crawled. All field-values are taken from this date.
# * `name` - Name of the car
# * `seller` - Whether the seller is private or a dealer.
# * `offerType` - The type of listing
# * `price` - The price on the ad to sell the car.
# * `abtest` - Whether the listing is included in an A/B test.
# * `vehicleType` - The vehicle Type.
# * `yearOfRegistration` - The year in which the car was first registered.
# * `gearbox` - The transmission type.
# * `powerPS` - The power of the car in PS.
# * `model` - The car model name.
# * `kilometer` - How many kilometers the car has driven.
# * `monthOfRegistration` - The month in which the car was first registered.
# * `fuelType` - What type of fuel the car uses.
# * `brand` - The brand of the car.
# * `notRepairedDamage` - If the car has a damage which is not yet repaired.
# * `dateCreated` - The date on which the eBay listing was created.
# * `nrOfPictures` - The number of pictures in the ad.
# * `postalCode` - The postal code for the location of the vehicle.
# * `lastSeenOnline` - When the crawler saw this ad last online.

# ## Initial Exploration

# In[1]:


# Import packages and explore data
import pandas as pd
import numpy as np
autos = pd.read_csv("autos.csv", encoding = "Latin-1")
autos.info()
autos.head()


# An initial glance at this dataset shows that most column entries are strings, as evidenced by their 'non-null object' designations.  The head of the data itself reveals that certain columns will definitely need to be cleaned prior to analysis.  Potential issues that could be addressed include language discrepancies, date and time formatting, model names, and the removal of special characters from the `price` and `odometer` columns.

# ## Data Cleaning
# 
# We'll effectuate the changes outlined above. Relabeling all multi-word column names and converting all column names to snake case matches with convention and makes the dataset easier to approach.

# In[2]:


# Clean column names
autos.columns = autos.columns.str.replace("monthOfRegistration", "registration_month").str.replace("notRepairedDamage", "unrepaired_damage").str.replace("yearOfRegistration", "registration_year").str.replace("dateCreated", "ad_created").str.lower()
autos.head()


# In[3]:


autos.describe(include = "all")


# In[4]:


autos["nrofpictures"].value_counts()


# Columns `seller` and `offertype` contain mostly the same value.  Since few new, interesting conclusions can likely be drawn from their analysis, these columns will be dropped from the dataset.  Additionally, the `nrofpictures` column contains 0 for every entry.  It will be dropped as well.

# In[5]:


# Drop columns
autos = autos.drop(["nrofpictures", "seller", "offertype"], axis = 1)


# The `price` and `odometer` columns are numeric values stored as text.  They must be cleaned and converted to 'numeric' dtype.

# In[6]:


# Price column
autos["price"] = autos["price"].str.replace("$", "").str.replace(",", "").astype(int)

autos["price"].head()


# In[7]:


# Odometer column
autos["odometer"] = (autos["odometer"].astype(str).str.replace("km", "").str.replace(",", "").astype(int))
autos.rename({"odometer":"odometer_km"}, axis = 1, inplace = True)
autos["odometer_km"].head()


# ## Exploring Price and Odometer
# 
# Values for the `odometer_km` column are rounded, indicating that sellers may have had to choose from preset options.  There are more high-mileage vehicles than low-mileage -- sensible for a classifieds website.

# In[8]:


autos["odometer_km"].value_counts()


# In[9]:


autos["price"].unique().shape


# In[10]:


autos["price"].describe()


# In[11]:


autos["price"].value_counts().head(20)


# From these summary statistics, we see that over 1,400 cars are listed with a price of $0.  These rows represent just 2\% of total cars, so we might consider removing them.  

# In[12]:


autos["price"].value_counts().sort_index(ascending = False)


# There are a number of price listings below \\$50, including nearly 1,500 at \\$0.  Since we are viewing data from eBay, there is a legitimate possibility that prices for certain cars could have started at \\$1.  

# We will eliminate rows with a price of \\$0, as well as those with a price greater than \\$350,000.  Prices increase steadily to that number before jumping to less realistic values beyond that threshold.

# These 1,591 eliminated rows represent just 3% of the total data.

# In[13]:


autos = autos[autos["price"].between(1, 351000)]
autos["price"].describe()


# Summary statistics for the remaining data are displayed above.  Just under 49,000 rows remain in the dataframe, with a mean price of \\$5,889.

# ## Exploring Date Columns
# 
# There are a number of columns with date information:
# 
# * `date_crawled`
# * `registration_month`
# * `registration_year`
# * `ad_created`
# * `last_seen`
# 
# Some of these columns need to be converted from string values to numeric representation: `datecrawled`, `ad_created`, `lastseen`.

# In[14]:


autos["datecrawled"].str[:10].value_counts(normalize = True, dropna = False).sort_index()


# From this output, it's evident that the site was crawled daily for just over a month from March to April, 2016.

# In[15]:


autos["lastseen"].str[:10].value_counts(normalize = True, dropna = False).sort_index()


# The crawler records the last date it encountered a listing, allowing us to see when certain listings were sold and removed from the site.  
# 
# The last three days account for proportions of sales six to ten times greater than the average daily proportion.  It's unlikely that this disparity is naturally occurring, given the uniformity of daily sales from the rest of the month-long period.  It seems that these last three days reflect the end of the crawling period rather than a spike in sales.

# In[16]:


autos["ad_created"].str[:10].value_counts(normalize = True, dropna = False).sort_index()


# Most ad creation dates fall within one or two months of the date the listing was crawled.  Some, however, are as much as nine months older.

# In[17]:


autos["registration_year"].describe()


# Upon examining the registration year of each listing, we find that the extreme values are nonsensical and will need to be explored further.

# ## Dealing With Incorrect Registration Year Data
# 
# It's impossible for a car to be registered in 9999 or anytime after 2016, so any rows with registration values greater than this will need to be removed.  We can't be certain about the lower limit for registration year.  It could be as early as the early part of the twentieth century, but certainly not as early as 1000.  We'll need to examine the data further in order to determine a reasonable lower limit.

# In[18]:


(~autos["registration_year"].between(1900, 2016)).sum()/autos.shape[0]


# We see that just under 4% of the data has registration years outside the range 1900 - 2016.  We can comfortably remove these rows without worrying about too drastically distorting our dataframe.

# In[19]:


# Drop data with registration years outside our range
autos = autos[autos['registration_year'].between(1900, 2016)]
print(autos["registration_year"].describe(include = "all"))
print(autos["registration_year"].value_counts(normalize = True))


# We now see that we've made a sound choice for the registration year boundaries by which we've pared the data.  The most recent registration is 2016, while the earliest is 1910.  The year in which the greatest proportion of cars were registered is 2000.  Most vehicles were registered in the past 20 years.

# ## Exploring Price by Brand

# In[20]:


autos["brand"].value_counts(normalize = True)


# German brands account for four of the top five most popular brands, and over 50% of all listed brands.  
# 
# Most brands do not represent a significant percentage of the data, so we'll limit our analysis of prices only to brands that account for over 3% of the data.

# In[21]:


# Drop brands that do not account for more than 3% of data
brand_counts = autos["brand"].value_counts(normalize = True)
common_brands = brand_counts[brand_counts > 0.03].index
print(common_brands)


# In[22]:


# Determine mean price by brand
brand_mean_prices = {}
for i in common_brands:
    brand = autos[autos["brand"] == i]
    mean = brand["price"].mean()
    brand_mean_prices[i] = int(mean)


# In[23]:


print(brand_mean_prices.items())


# We have calculated the mean prices for all listings of the seven most common brands on the site and stored these key-value pairs in a dictionary entitled `brand_mean_prices`.  We find that Audi listings are, on average, the  most expensive, followed closely by Mercedes-Benz and BMW.
# 
# There's a significant gap between the top 'tier' of brands--Audi, Mercedes, and BMW--and the others.  Interestingly, BMW is the third most expensive brand, as well as the second most frequently listed.  
# 
# Volkswagen, meanwhile, is the most common brand, accounting for 21% of listings, almost twice as many as BMW.  It's also the fourth most expensive brand, with an average price squarely between the two distinct brand tiers at \$5,402.  This moderate price may help explain its popularity. 

# ## Exploring Mileage

# We have already calculated the mean price for the top seven brands listed, and we can use the same technique to calculate the mean mileage for those brands.  After doing so, we can create a new dataframe containing this information and determine whether there is a connection between price and mileage, an indication of usage.

# In[24]:


# Determine mean mileage by brand
brand_mean_mileage = {}
for i in common_brands:
    brand = autos[autos["brand"] == i]
    mean = brand["odometer_km"].mean()
    brand_mean_mileage[i] = int(mean)

brand_mean_mileage


# In[25]:


# Convert mileage and price info to series objects and then to dataframe
mileage = pd.Series(brand_mean_mileage)
pricing = pd.Series(brand_mean_prices)

df = pd.DataFrame(mileage, columns = ["mean_odometer_km"])


# In[26]:


df["mean_price"] = pricing


# In[27]:


df


# The spread of mean mileage values is tighter than that of mean prices for each brand.  The highest mean mileage, belonging to BMW, is greater than the lowest by just 6.3%.  The highest mean price, belonging to Audi, is greater than the lowest by 73.5%.

# ## Evaluating Sales of Damaged Vehicles
# 
# We can evaluate price for damaged and undamaged vehicles. We hypothesize that damaged vehicles will, on average, be priced lower, since buyers aren't likely to pay as much for flawed products.

# In[28]:


# Partition original data into damaged and undamaged
undamaged = autos[autos["unrepaired_damage"] == "nein"]
damaged = autos[autos["unrepaired_damage"] == "ja"]


# In[30]:


# Calculate and show mean prices for each category
undamaged_price = undamaged["price"].mean()
damaged_price = damaged["price"].mean()

print("Undamaged mean price: ", undamaged_price)
print("Damaged mean price: ", damaged_price)

print(damaged["price"].describe())
print(undamaged["price"].describe())


# We see that, as expected, listings with previous damage sell for far lower prices than undamaged listings.  The mean price for `undamaged` is over three times that of `damaged`, while the maximum price is almost eight times greater.

# ## Examining the Most Common Models
# 
# So far, we have only examined the popularity and statistics of certain brands. It would be interesting to investigate these trends for individual car models. 

# In[31]:


brands_list = autos["brand"].value_counts().index


# In[32]:


# Create series objects of car brands and models
brands = pd.Series(autos["brand"])
models = pd.Series(autos["model"])
model_df = pd.DataFrame(brands, columns = ['brand'])
model_df["model"] = models

# Create empty list of models
models_list = {}

# Increment counts of individual model types within list
for index, row in model_df.iterrows():
    car = str(row["brand"]) + " " + str(row["model"])
    if car in models_list:
        models_list[car] += 1
    else:
        models_list[car] = 1
        
sorted_models = sorted(models_list.items(), key=lambda x: x[1], reverse = True)
sorted_models[:10]


# We find that all of the top ten listed models are German-made.  The Volkswagen Golf holds the top spot, outselling the second-place BMW 3er by nearly 1,100 cars.

# ## Conclusion
# 
# In this project, we explored a number of trends and attributes within auto sales data from the German eBay website, demonstrating data cleaning techniques and critical consideration of how and why to render particular features fit for analysis.
