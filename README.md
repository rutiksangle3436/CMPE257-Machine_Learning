# CMPE257-Machine_Learning

## Team Info and usernames
1. Nhat Trinh [011227645] - nhattrinh
2. Suhas Byrapuneni [016118596] - suhas-byrapuneni
3. Venkata Sai Sri Batchu [016118557] - chaitanya1818
4. Rutik Sanjay Sangle [016007589] - rutiksangle3436


## Data
Our group will build the project using the COVID-19 dataset provided by WHO. Specifically, the dataset is named “Daily cases and deaths by date reported to WHO”. The dataset is provided as a CSV file. The dataset includes almost every country in the world that has reported COVID deaths since the start of the 2020 calendar year. Each row in the dataset has the date, country code, country name, assigned WHO region, new deaths, new cases, cumulative deaths, and cumulative cases.

## Problem
The problem our group will be trying to solve is finding where and when the next COVID outbreak will happen by looking at historic daily data from the beginning of the pandemic and learning from it. A COVID outbreak can be characterized as an abnormal change of upwards slope in the daily COVID cases/deaths graph. Even though the cause of COVID-19 transmission can be multi-faceted, such as transmission through touch, proximity, city planning, region, etc. Our group believes it can be largely tied to seasonal and temperature changes that cause the uptick in COVID cases and/or deaths.

## Potential Methods
Since our dataset is labeled, Our team will potentially try to use a supervised learning method; namely, the regression method can understand the relationship between dependent and independent variables. Specifically, our group will try to use an autoregression model utilizing Poisson distribution, called Poisson Autoregression (PAR).

## Preprocessing
The initial data analysis was carried out by checking for the types of data values and checking if there are any missing values. The type of data values is suitable for our solution but there were a few missing values in the column ‘Country_code’. After some more investigation, we found out that the country code for Namibia was missing. By using the fillna() method of pandas, the missing values were replaced. 

## Initial Findings
For initial findings, we made some plots using Plotly.express library. The plots describe some basic information like the Top 10 countries with the highest number of cases. Along with that, we used the choropleth plot for plotting the world map which shows the number of total cases in every country.
