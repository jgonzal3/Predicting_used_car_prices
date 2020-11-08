# Predicting_used_car_prices
Using machine learning to predict used car prices based on the data available on www.sahibinden.com

I'm planning to sell my car which is a 4-year-old wolkswagen polo. Used cars are usually sold on a website called "sahibinden" in Turkey. "Sahibinden" means "from the owner" although there are many dealers using this website to sell or buy used cars. The most critical part of selling a used car is to determine the optimal price. There are many websites that give you a price for used cars but you still want to search the market before setting the price. Moreover, there are other factors which affect the price such as location, how fast you want to sell the car, smoking in the car and so on. Before you post your ad on the website, it is best to look through the price of similar cars. However, this process might be exhausting because there are too many ads online. Therefore, I decided to take advantage of the convenience offered by machine learning to create a model that predicts used car prices based on the data available on "sahibinden".

**Predicting Used Car Prices with Machine Learning**

A complete data science project from data collection to model evaluation

I'm planning to sell my car which is a 4-year-old Volkswagen polo. Used
cars are usually sold on a website called
"[[sahibinden]{.underline}](https://www.sahibinden.com/kategori/vasita)"
in Turkey. "Sahibinden" means "from the owner" although there are many
dealers using this website to sell or buy used cars. The most critical
part of selling a used car is to determine the optimal price. There are
many websites that give you a price for used cars but you still want to
search the market before setting the price. Moreover, there are other
factors which affect the price such as location, how fast you want to
sell the car, smoking in the car and so on. Before you post your ad on
the website, it is best to look through the price of similar cars.
However, this process might be exhausting because there are too many ads
online. Therefore, I decided to take advantage of the convenience
offered by machine learning to create a model that predicts used car
prices based on the data available on
"[[sahibinden]{.underline}](https://www.sahibinden.com/kategori/vasita)".
It will not only help solve my problem of determining a price for my car
but also help me learn and practice many topics related to data science.

This project is divided into 5 subsections as follows:

-   Data collection

    Data cleaning

    Exploratory Data Analysis

    Regression Model and Evaluation

    Further improvement

All the data and codes are available on a [[github
repository]{.underline}](https://github.com/SonerYldrm/Predicting_used_car_prices).
Feel free to use or distribute.

1.  Data Collection

There are more than six thousand Volkswagen polo for sale on
"[[sahibinden]{.underline}](https://www.sahibinden.com/volkswagen-polo)"
website. I had to do web scraping to collect data from the website. I'm
not an expert on web scraping but I've learned enough to get what I
need. I think it is very important to learn web scraping to a certain
level if you want to work or are working in data science domain because
data is not usually served on a plate to us. We have to get what we
need.

I used [[beautiful
soup]{.underline}](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
which is a python library for pulling data out of HTML and XML files.
The syntax is pretty simple and easy to learn. There are a few important
details that you need to pay attention especially if the data is listed
on several pages.

Always import the dependencies first:

import pandas as pd

import numpy as np

import requests

from bs4 import BeautifulSoup as bs

I used the **get()** method of python's **requests** library to retrieve
data from the source and store it in a variable. Then I used beautiful
soup to extract and organize the content of this variable. Since the
data is on several pages, I had to create list to help parse through
different pages and also initiate empty lists to save the data.

\#initiate empty lists to save data

model\_info = \[\]

ad\_title = \[\]

year\_km\_color = \[\]

price = \[\]

ad\_date = \[\]

location = \[\]

\#create lists to parse through pages

page\_offset = list(np.arange(0,1000,50))

min\_km = \[0, 50000, 85000, 119000, 153000, 190000, 230000\]

max\_km = \[50000, 85000, 119000, 153000, 190000, 230000, 500000\]

The maximum number of ads displayed on a page is 50. In order to scrape
data for about six thousand cars, I needed to iterate over 120 pages.
First, I organized the code in a for loop to extract data from 120
pages. However, after the process was done, I found out that data was
repeated after first 1000 entries. Then, I decided to group data into
smaller sections which would not exceed 1000 entries per group so I used
'km' criteria to differentiate groups. I created nested for loops to
extract data for about six thousands cars as below:

for i, j in zip(min\_km, max\_km):

for page in page\_offset:

r = requests.get(f\'https://www.sahibinden.com/volkswagen-
polo?pagingOffset=**{page}**&pagingSize=50&a4\_**max={j}**&sorting=date\_asc&a4\_**min={i}**\',
headers=headers)

soup = bs(r.content,\'lxml\')

model\_info +=
soup.find\_all(\"td\",{\"class\":\"searchResultsTagAttributeValue\"})

ad\_title +=
soup.find\_all(\"td\",{\"class\":\"searchResultsTitleValue\"})

year\_km\_color +=
soup.find\_all(\"td\",{\"class\":\"searchResultsAttributeValue\"})

price += soup.find\_all(\"td\",{\"class\":\"searchResultsPriceValue\"})

ad\_date +=
soup.find\_all(\"td\",{\"class\":\"searchResultsDateValue\"})

location +=
soup.find\_all(\"td\",{\"class\":\"searchResultsLocationValue\"})

At each iteration, the base url is modified using the values in
page\_offset, max\_km and min\_km lists to go to next page. Then the
content of website is decomposed into pre-defined lists based on the tag
and class. The classes and tags in html can be displayed by inspecting
the website on the browser.

![](media/image1.png){width="6.041666666666667in"
height="2.3472222222222223in"}

HTML of
"[[sahibinden]{.underline}](https://www.sahibinden.com/volkswagen-polo)"
website

After getting the content of html, I extracted the text part:

model\_info\_text = \[\]

for i in range(0,6731):

model\_info\_text.append(model\_info\[i\].text)

This process was done for each list and then I combined the lists to
build a pandas DataFrame:

df = pd.DataFrame({\"model\":model\_info\_text,
\"ad\_title\":ad\_title\_text,\"year\":year\_text, \"km\":km\_text,
\"color\":color\_text,\"price\":price\_text,
\"ad\_date\":ad\_date\_text, \"location\":location\_text})

print(df.shape)

print(df\[\'ad\_title\'\].nunique())

(6731, 8)

6293

Dataframe includes 6731 entries but 6293 of them seem to be unique
according to the title of the ad which I think is the best option to
distinguish ads. Some users might re-post the same ad or titles of some
ads might be exactly the same.

2.  Data Cleaning

I saved the data scraped from the website as a csv file.

df = pd.read\_csv(\'polo\_data.csv\')

df.head()

![](media/image2.png){width="6.546027996500437in"
height="1.834750656167979in"}

New line indicators (\\n) had to be removed. I used pandas **remove()**
function with **regex** parameter set True. Similarly TL representing
Turkish currency in price cell had to be removed to make numerical
analysis.

df = df.replace(\'\\n\',\'\',regex=True)

df.price = df.price.replace(\'TL\',\'\',regex=True)

We always need to look for missing values and check data types before
trying to do any analysis:

df.isna().any()

model False

ad\_title False

year False

km False

color False

price False

ad\_date False

location False

dtype: bool

df.dtypes

model object

ad\_title object

year int64

km float64

color object

price object

ad\_date object

location object

dtype: object

The data type of date was object. To be able to use the dates properly,
I converted data dype to **datetime**. The data is in Turkish so I
changed the name of months to English before using **astpye()**
function. I used a dictionary to change the names of the months.

months = {\"Ocak\":\"January\", \"Şubat\":\"February\",
\"Mart\":\"March\",
\"Nisan\":\"April\",\"Mayıs\":\"May\",\"Haziran\":\"June\",\"Temmuz\":\"July\",\"Ağustos\":\"August\",\"Eylül\":\"September\",
\"Ekim\":\"October\", \"Kasım\":\"November\", \"Aralık\":\"December\"}

df.ad\_date = df.ad\_date.replace(months, regex=True)

\#change the datatype

df.ad\_date = pd.to\_datetime(df.ad\_date)

The "km" colums which shows how many kilometres the car has made so for
was truncated while reading the csv file. It is because of 'dot' used in
thousands. For example, 25.000 which is twenty five thousands detected
as 25.0. To fix this issue, I multiplied 'km' column with 1000. To be
able to change the datatype of "km" column to numeric (int or float), I
also removed "." and "," characters.

df.km = df.km \* 1000

df.iloc\[:,5\] = df.iloc\[:,5\].str.replace(r\'.\',\'\')

df.iloc\[:,5\] = df.iloc\[:,5\].str.replace(r\',\',\'\')

\#change the datatype

df.price = df.price.astype(\'float64\')

In Turkey, location might be a factor in determining the price of a used
car due to uneven population distribution. Location data in our
dataframe includes city and district. I don't think price changes in
different districts of the same city. Therefore, I modified location
data to include only the name of the city.

![](media/image3.png){width="1.7916666666666667in"
height="2.3333333333333335in"}

Location information is formatted as CityDistrict (no space in between).
The name of the district starts with a capital letter which can be used
to separate city and district. I used the **sub()** function of **re**
module of python.

import re

s = df\[\'location\'\]

city\_district = \[\]

for i in range(0,6731):

city\_district.append(re.sub( r\"(\[A-Z, \'Ç\', \'İ\', \'Ö\', \'Ş\',
\'Ü\'\])\", r\" \\1\", s\[i\]).split())

city\_district\[:5\]

\[\[\'Ağrı\', \'Merkez\'\],

\[\'İstanbul\', \'Kağıthane\'\],

\[\'Ankara\', \'Altındağ\'\],

\[\'Ankara\', \'Çankaya\'\],

\[\'Samsun\', \'Atakum\'\]\]

This for loop splits the strings in each cell of location column at
capital letters. Turkish alphabet has letters that are not in the
\[A-Z\] range of English alphabet. I added these letters in sub function
as well. The output is a list of two-item lists. I created another
column named "city" using the first items of this list.

city = \[\]

for i in range(0,6731):

city.append(city\_district\[i\]\[0\])

city\[:5\]

\[\'Ağrı\', \'İstanbul\', \'Ankara\', \'Ankara\', \'Samsun\'\]

df\[\'city\'\] = city

**nunique()** function counts the unique values which can be useful for
both exploratory data analysis and confirming the results.

df.city.nunique()

81

There are 81 cities in Turkey so the dataset includes at least one car
in each city.

3.  Exploratory Data Analysis

**Price**

It's always good to get some insight about the target variable. The
target or dependent variable is price in our case.

print(df.price.mean())

print(df.price.median())

83153.7379289853

64250.0

Mean is much higher than median which indicates there are outliers or
extreme values. Let's also check maximum and minimum values:

print(df.price.max())

print(df.price.min())

111111111.0

24.0

This values are obviously wrong. There is no Volkswagen polo for over
100 million unless it is gold coated. Similarly, the value of 24 Turkish
Liras is not possible. After sorting values in price column by using
**sort\_values()** function, I detected a few more outliers and dropped
them using pandas **drop()** function by passing indexes of the values
to be dropped. Let's check new mean and median values:

print(df.price.mean())

print(df.price.median())

print(df.price.median())

66694.66636931311

64275.0

25000.0

Mean is still higher than median but the difference is not extreme. I
also checked mode which is the value that occurs most often. Mean being
higher than median indicates that the data is right or positive skewed
which means we have more of lower prices and some outliers with higher
values. Measures of central tendency being sorted as mean \> median \>
mode is an indication of positive (right) skewness. We can double check
with distribution plot:

x = df.price

plt.figure(figsize=(10,6))

sns.distplot(x).set\_title(\'Frequency Distribution Plot of Prices\')

![](media/image4.png){width="5.999305555555556in"
height="3.671499343832021in"}

It can be seen from the graph that the data is right skewed and the peak
around 25000 shows us the mode. Another way of checking the distribution
and outliers is **boxplot**:

plt.figure(figsize=(8,5))

sns.boxplot(y=\'price\', data=df, width=0.5)

![](media/image5.png){width="5.013194444444444in"
height="2.834785651793526in"}

The bottom and top of the blue box represent first quartile (25%) and
third quartile (75%), respectively. First quartile means 25% of data
points are below this point. The line in the middle is the median (50%).
The outliers are shown with dots above the maximum line.

**Date**

I don't think date by itself has an effect on the price but waiting
period of the ad on website is a factor to be considered. Longer waiting
time might motivate owner to reduce the price. If an ad stays on the
website for a long time, it might be because the price is not set
properly. So I will add a column indicating the number of days ad has
been on the website. Data was scraped on 18.01.2020.

df\[\'ad\_duration\'\] = pd.to\_datetime(\'2020-01-18\') -
df\[\'ad\_date\'\]

![](media/image6.png){width="1.05625in" height="2.0305555555555554in"}

Ad\_duration must be a numerical data so 'days' next to numbers need to
be removed. I used pandas **replace()** function to remove 'days'.

Let's check the distribution of ad duration data:

print(df.ad\_duration.mean())

print(df.ad\_duration.median())

12.641540291406482

10.0

![](media/image7.png){width="6.027083333333334in"
height="3.8399146981627297in"}

Mean is higher than the median and there are many outliers. Data is
right skewed. To get a better understanding, I also plotted data points
less than 50:

![](media/image8.png){width="6.027083333333334in"
height="3.8399146981627297in"}

**Location**

There are 81 different cities but 62% of all ads are listed in top 10
cities with Istanbul having 23% of all ads.

a = df.city.value\_counts()\[:10\]

df\_location = pd.DataFrame({\"city\": a , \"share\": a/6726})

df\_location.share.sum()

0.6216176033303599

![](media/image9.png){width="1.7255063429571305in"
height="2.6327318460192477in"}

**Color**

It seems like the optimal choice of color is white for Volkswagen polo.
More than half of the cars are white followed by red and black. Top 3
colors cover 72% of all cars.

![](media/image10.png){width="1.9705916447944007in"
height="2.8583333333333334in"}

**Year**

The age of the car definitely effects the prices. However, instead of
the model year of the car, it makes more sense to use is as age. So I
substituted 'year' column from current year.

df\[\'age\'\] = 2020 - df\[\'year\'\]

![](media/image11.png){width="5.527083333333334in"
height="3.521360454943132in"}

According to the distribution, most of the cars are less than 10 years
old. There is a huge drop at 10 followed by an increasing trend.

**Km**

Km value shows how much the car has ben driven so it is definitely an
important factor determining the price. Km data has approximately a
normal distribution.

print(df.km.mean())

print(df.km.median())

141011.5676479334

137000.0

![](media/image12.png){width="5.405883639545057in"
height="3.308333333333333in"}

**Ad title**

Ad title is kind of a caption of the ad. Sellers try to attract possible
buyers with a limited number of characters. Once an ad is clicked on,
another page with pictures and more detailed information opens up.
However, the first step is to get people to click on your ad so ad title
plays a critical role in selling process.

Let's check what people usually write in the title. I used **wordcloud**
for this task.

\#import dependencies

from wordcloud import WordCloud, STOPWORDS

The only required parameter for WordCloud is a text. You can check the
docstring by typing "?WordCloud" for other optional parameters. We
cannot input a list to wordcloud so I created a text by concatenating
all the titles in ad\_title column:

text\_list = list(df.ad\_title)

text = \'-\'.join(text\_list)

Then used this text to generate a wordcloud:

\#generate wordcloud

wordcloud = WordCloud(background\_color=\'white\').generate(text)

\#plot wordcloud

plt.figure(figsize=(10,6))

plt.imshow(wordcloud, interpolation=\'bilinear\')

plt.axis(\"off\")

plt.show()

![](media/image13.png){width="4.55625in" height="2.3996259842519687in"}

The idea of a wordcloud is pretty simple. The more frequent words are
shown bigger. It is an informative and easy-to-understand tool for text
analysis. However, the wordcloud above does not tell us much because the
words "vw", "Volkswagen" and "polo" are not what we are looking for.
They show the brand we are analyzing. In this case, we should use
**stopwords** parameter of wordcloud to list the words that need to be
excluded.

stopwords = \[\'VW\', \'VOLKSWAGEN\', \'POLO\', \'MODEL\', \'KM\'\]

wordcloud = WordCloud(stopwords=stopwords).generate(text)

plt.figure(figsize=(10,6))

plt.imshow(wordcloud, interpolation=\'bilinear\')

plt.axis(\"off\")

plt.show()

![](media/image14.png){width="5.18125in" height="2.7287915573053367in"}

I did not use **background\_color** parameter this time just to show the
difference. The words are in Turkish so I will give a brief explanation:

-   "Hatasız" : Without any problem/issue

    "Sahibinden": From the owner (this is important because people tend
    to buy from the owner rather than a dealer).

    "Otomatik": Automatic transmission

    "Boyasız": No paint (no part of the car painted due to a crack,
    scratch or a repair)

The other words are mainly about being clean, not having any previous
repairs.

**Model**

Model column includes three different kinds of information: engine size,
fuel type and variant. After checking the values, I found out that only
engine size information is complete for all cells. Fuel type and variant
are missing for most of the cells so I created a separate column for
engine size.

![](media/image15.png){width="1.30625in" height="2.1277777777777778in"}

The first three characters after spaces represent engine size. I first
removed spaces and extracted the first three characters from model
column:

\#remove spaces

df.model = df.model.replace(\' \',\'\',regex=True)

engine = \[x\[:3\] for x in df.model\]

df\[\'engine\'\] = engine

Let's check how price changes with different engine sizes:

df.engine.value\_counts()

1.4 3172

1.6 1916

1.2 1205

1.0 409

1.9 20

1.3 4

Name: engine, dtype: int64

df\[\[\'engine\',\'price\'\]\].groupby(\[\'engine\'\]).mean().sort\_values(by=\'price\',
ascending=False)

![](media/image16.png){width="1.6292060367454069in"
height="2.0388888888888888in"}

It seems like average price decreases with increasing engine size. 1.3
can be ignored since there are only 4 cars with 1.3 engine. There is a
big gap with 1.0 and the other engine sizes because 1.0 is a newer
model. As you can see on the chart below, cars with 1.0 engine size have
both lowest age and km on average which shows that they are newer
models.

![](media/image17.png){width="2.2215277777777778in"
height="2.052088801399825in"}

4.  Regression Model

Linear regression is a widely-used supervised learning algorithm to
predict a continuous dependent (or target) variable. Depending on the
number of independent variables, it could be in the form of simple or
multiple linear regression. I created a multiple linear regression model
because I used many independent variables to predict the dependent
variable which is the price of a used car.

We should not just use all of the independent variables without any
pre-processing or prior judgment. **Feature selection** is the process
to decide which features (independent variables) to use in the model.
Feature selection is a very critical step because using unnecessary
features has a negative effect on the performance and eliminating
important features prevent us from getting a high accuracy.

We can use **regression plots** to check the relation between dependent
variable and independent variables. I checked the relationship between
km and price which I think is highly correlated.

plt.figure(figsize=(10,6))

sns.regplot(x=\'km\', y=\'price\', data=df).set\_title(\'Km vs Price\')

![](media/image18.png){width="6.027083333333334in"
height="3.6255938320209973in"}

It is clearly seen that as the km goes up, price goes down. However,
there are outliers. According to the regression plot above, cars with km
higher than 400000 can be marked as outliers. I removed these outliers
in order to increase the accuracy of the model. Outliers tend to make
the model over fitting.

df = df\[df.km \< 400000\]

plt.figure(figsize=(10,6))

sns.regplot(x=\'km\', y=\'price\', data=df).set\_title(\'Km vs Price\')

![](media/image19.png){width="6.184507874015748in"
height="3.761111111111111in"}

Much better now!

After applying same steps with age and price, a similar relationship was
observed:

![](media/image20.png){width="5.929861111111111in"
height="3.6062478127734034in"}

I also checked the relationship between ad duration and engine size with
price. Average price decreases as the engine size gets bigger. However,
ad duration seems to have little to no effect on price.

![](media/image21.png){width="5.965261373578302in"
height="3.6277777777777778in"}

![](media/image22.png){width="5.929861111111111in"
height="3.6062478127734034in"}

Another way to check relationship between variables is **correlation
matrix**. Pandas **corr()** function calculates correlation between
numerical variables.

![](media/image23.png){width="6.054861111111111in"
height="1.630906605424322in"}

The closer the value is to 1, the higher the correlation. '-' sign
indicates negative correlation. These values are inline with the
regression plots above.

We can also visualize the correlation matrix using seaborn **heatmap**:

corr = df.corr()

plt.figure(figsize=(10,6))

sns.heatmap(corr, vmax=1, square=True)

![](media/image24.png){width="4.540972222222222in"
height="3.433875765529309in"}

The color of box at the intersection of two variables shows the
correlation value according to the color chart at the right.

**Linear Regression Model**

After checking the correlation and distribution of variables, I decided
to use age, km, engine size and ad duration to predict the price of a
used car.

I used [[scikit-learn]{.underline}](https://scikit-learn.org/stable/)
which provides simple and effective machine learning tools.

from sklearn.model\_selection import train\_test\_split

from sklearn.linear\_model import LinearRegression

from sklearn.metrics import r2\_score

I extracted the features (columns) to be used:

X = df\[\[\'age\',\'km\',\'engine\',\'ad\_duration\'\]\] \#independent
variables

y = df\[\'price\'\] \#dependent (target) variable

Then using **train\_test\_split** function of scikit-learn, I divided
the data into train and test subsets. To separate train and test set is
a very important step for every machine learning algorithm. Otherwise,
if we both train and test on the same dataset, we would be asking the
model to predict something it already knows.

X\_train, X\_test, y\_train, y\_test = train\_test\_split(X, y,
random\_state=42)

Then I created a LinearRegression() object, trained it with train
dataset.

linreg = LinearRegression()

linreg.fit(X\_train, y\_train)

It's time to measure the accuracy of the model. I measured the accuracy
of model on both train and test dataset. If accuracy on train dataset is
much higher than the accuracy on test dataset, we have a serious
problem: **overfitting**. I will not go in detail about overfitting. It
might be a topic of another post but I just want to give a brief
explanation. Overfitting means the model is too specific and not
generalized well. An overfit model tries to capture noise and extreme
values on training dataset.

linreg.score(X\_train, y\_train)

0.8351901442035045

linreg.score(X\_test, y\_test)

0.8394139260643358

The scores are very close which is good. The score here is
[**[R-squared]{.underline}**](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html)
score which is a measure to determine how close the actual data points
are to the fitted regression line. The closer R-squared score is to 1,
the more accurate our model is. R-squared measures how much of the
variation of target variable is explained by our model.

**Residual plots** are used to check the error between actual values and
predicted values. If a linear model is appropriate, we expect to see the
errors to be randomly spread and have a zero mean.

plt.figure(figsize=(10,6))

sns.residplot(x=y\_pred, y=y\_test)

The mean of the points might be close to zero but obviously they are not
randomly spread. The spread is close to a U-shape which indicates a
linear model might not be the best option for this task.

In this case, I wanted to try another model using
**RandomForestRegressor()** of scikit-learn. Random Forest is an
ensemble method built on **decision trees**. [[This
post]{.underline}](https://towardsdatascience.com/an-implementation-and-explanation-of-the-random-forest-in-python-77bf308a9b76)
by Will Koehrsen gives a comprehensive explanation about decision trees
and random forests. Random forest are generally used for classification
task but work well on regression too.

from sklearn.ensemble import RandomForestRegressor

regr = RandomForestRegressor(max\_depth=5, random\_state=0,
n\_estimators=10)

![](media/image25.png){width="6.06875in" height="3.475997375328084in"}

Residual Plots figure

Unlike linear regression, there are critical hyperparametes to optimize
for random forests.
**[[max\_depth]{.underline}](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)**
is maximum depth of a tree (quite self-explanatory) which controls how
deep a tree is or how many splits you want.
[**[n\_estimator]{.underline}**](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
is the number of trees in a forest. Decision trees are prone to
overfitting which means you can easily make it too specific. If
max\_depth is too high, you will likely end up with an overfit model. I
manually changed max\_depth in order to check accuracy and overfitting.
However, scikit-learn provides very good tools for hyperparameter
tuning:
**[[RandomizedSearchCV]{.underline}](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
and
[[GridSearchCV]{.underline}](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).**
For more complex tasks and model, I highly recommend to use it.

print(\'R-squared score (training): {:.3f}\'

.format(regr.score(X\_train, y\_train)))

R-squared score (training): 0.902

print(\'R-squared score (training): {:.3f}\'

.format(regr.score(X\_test, y\_test)))

R-squared score (training): 0.899

R-squared score on test set is 0.899 which indicates a significant
improvement compared to linear regression. I also tried with max\_depth
parameter set to 20 and the result is on overfit model as below. Model
is very accurate on training set but accuracy on test set becomes lower.

regr = RandomForestRegressor(max\_depth=20, random\_state=0,
n\_estimators=10)

regr.fit(X\_train, y\_train)

print(\'R-squared score (training): {:.3f}\'

.format(regr.score(X\_train, y\_train)))

R-squared score (training): 0.979

print(\'R-squared score (training): {:.3f}\'

.format(regr.score(X\_test, y\_test)))

R-squared score (training): 0.884

Finally, I checked the price of my car according to the model:

regr.predict(\[\[4,75000,1.2,1\]\])

array(\[99743.84587199\])

The model suggested me to sell my car for almost 100 thousand which is
higher than the price in my mind. However, my car has been in an
accident and repaired which lowers the price. This information was not
taken into account as an independent variable which brings us to the
last section of this post: Further improvement.

5.  Further improvement

There are many ways to improve a machine learning model. I think the
most fundamental and effective one is to gather more data. In our case,
we can (1) collect data for more cars or (2) more information of the
cars in the current dataset or both. For the first one, there are other
websites to sell used cars so we can increase the size of our dataset by
adding new cars. For the second one, we can scrape more data about the
cars from
"[[sahibinden]{.underline}](https://www.sahibinden.com/volkswagen-polo?pagingSize=50)"
website. If we click on an ad, another page with detailed information
and pictures opens up. In this page, people write about the problems of
the car, any previous accident or repairs and so on. This kind
information is definitely valuable.

Another way to improve is to adjust model hyper-parameters. We can use
[[RandomizedSearchCV]{.underline}](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
to find optimum hyperparameter values.

6.  Conclusion

I tried to give you an overview of how a machine learning project builds
up. Although this is not a very complicated task, most of the data
science projects follow a similar pattern.

-   Define the problem or question

    Collect and clean the data

    Do exploratory data analysis to get some insight about data

    Build a model

    Evaluate the model

Go back to any of the previous steps unless the result is sufficient.
