# Outage Duration Prediction and Cause Analysis


<div style="text-align: right; font-size: 14px; color: #555;">
  Created by <a href="https://github.com/anvitag" target="_blank">Anvita Gollu</a> |
  <a href="https://github.com/anvita-g/Power-Outage-Prediction" target="_blank">GitHub Repo</a> | EECS 398
</div>


# Introduction

### Research Motivation: 
The central research question is: **"What characteristics are associated with each category of cause?"**. Identifying trends within the data could lead to predictive models for future outages, helping utility companies anticipate outages to prioritize maintaenance and design mitigantion strategies. Overall, this would reduce downtime and improve the reliabilty for customers.


### Dataset Overview
The dataset contains **1534 rows** with each row representing a specific outage event witnessed by different states in the continental U.S. during January 2000 – July 2016.

| Column Name             | Description                                                   |
|-------------------------|----------------------------------------------------------------|
| `CAUSE.CATEGORY`        | Category of the event that caused the power outage |
| `CAUSE.CATEGORY.DETAIL` | More specific description of the outage cause within the category|
| `OUTAGE.DURATION`       | Duration of the power outage in minutes    |
| `U.S._STATE`            | US state where the outage occurred                   |  
| `CLIMATE.REGION`        | US climate region classification based on National Centers for Environmental Information|
| `YEAR`                  | The year the outage took place |
| `POPULATION`            | Population size, which may provide urbanization details or affect outage complexity and response time|
| `MONTH`                 | The numeric month of the outage, which contributes to seasonal features
| `OUTAGE.START.DATE`         | Date the outage started                       |
| `OUTAGE.START.TIME`         | Time of day the outage started                                           |
| `OUTAGE.RESTORATION.DATE`   | Date when power was fully restored                                        |
| `OUTAGE.RESTORATION.TIME`   | Time of day when power was restored    |




# Exploratory Data Analysis


***
<div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
  <iframe
    src="assets/cause1-bar-plot.html"
    width="620"
    height="450"
    frameborder="0">
  </iframe>

  <iframe
    src="assets/cause2-bar-plot.html"
    width="620"
    height="450"
    frameborder="0">
  </iframe>
</div>


The Distribution of Outage Categories bar chart shows that "Severe Weather" is the most common cause category, accounting for 763 outages. This is more than half of all the recorded events. "Intentional Attack" is the second most common cause with 418 outages, highlighting that human-driven disruptions such as vandalism (cause of 335 outages) is a significant concern. Other categories such as equipment failure, fuel supply issues, and public appeal are relatively rare.

The second plot Top 5 Outage Causes reinforces that human factors and natural hazards contribute to power outages. The top weather related causes such as "Thunderstorm" (178), "Winter Storm" (101), "Hurricanes" (74), and "Heavy Wind" (61) occur during different seasons. This suggests that weather-related power disruptions are diverse and seasonally distributed.

***

<div style="display: flex; justify-content: center; gap: 30px; flex-wrap: wrap; align-items: flex-start;">
  <iframe
    src="assets/cause-duration.html"
    width="650"
    height="450"
    frameborder="0">
  </iframe>

  <div>
    <h5>Average Outage Duration by Cause (in Hours)</h5>
    <table>
      <thead>
        <tr>
          <th>Cause Category</th>
          <th>Outage Duration (Hours)</th>
        </tr>
      </thead>
      <tbody>
        <tr><td>fuel supply emergency</td><td>224.73</td></tr>
        <tr><td>severe weather</td><td>64.73</td></tr>
        <tr><td>equipment failure</td><td>30.28</td></tr>
        <tr><td>public appeal</td><td>24.47</td></tr>
        <tr><td>system operability disruption</td><td>12.15</td></tr>
        <tr><td>intentional attack</td><td>7.17</td></tr>
        <tr><td>islanding</td><td>3.34</td></tr>
      </tbody>
    </table>
  </div>
</div>


To explore the relationship between causes and outage duration, I created a box plot and pivot table. This reveals that severe weather outages are not only the most common but also have a wide range of durations where most of them fall within a reasonable window but some extend beyond 500 hours, indicating occasional extreme events. On the other hand, although fuel supply outages is rare, they have a long median duration and a few significant outliers exending 1000+ hours. Intentional attack and islanding have short and clustured durations, suggesting these are often resolved quickly. The pivot table complements the box plot by summarizing the mean outage duration per cause; fuel supply emergency is the highest with an average of 224.73 hours, followed by severe weather at 64.73 hours, and equipment failure at 30.28 hours.

***

##### Severe Weather Event Count by Climate Region

| Climate Region        | Severe Weather Event Count |
|------------------------|----------------------------|
| Northeast              | 175                        |
| Central                | 133                        |
| Southeast              | 116                        |
| South                  | 106                        |
| East North Central     | 104                        |
| West                   | 67                         |
| Northwest              | 25                         |
| Southwest              | 10                         |
| West North Central     | 4                          |


The pivot table shows how frequently severe weather is reported by the ause of power outages across different U.S. climate regions. The top 3 are Northeast (175), Central (133), and Southeast (116). This aligns with the knows weather patterns in these areas as being particularly prone to storms, hurricanes, and winter weather. This distribution enforce the idea that geography plays a major role in the likelihood of outages due to weather.

***

## Data Cleaning


##### Combining Start Date and Start Time

I combined the OUTAGE.START.DATE and OUTAGE.START.TIME columns into a single OUTAGE.START timestamp using pd.to_datetime(). Similarly, I also created a OUTAGE.RESTORATION timestamp from OUTAGE.RESTORATION.DATE and OUTAGE.RESTORATION.TIME columns. Before doing this, I dropped any rows that were missing in these columns to avoid generating invalid timestamps.

Significance: This transformation is important because it allows for more precise time-based calculations, such as calculating outage durations, identifying trends by hour or seasons, and incorpating time-based features into predictive models. Combining these featires makes time-based operations more consistent and readable.


##### Imputation 

To ensure integrity of the model, I removed all rows that contained missing values in the featues used for the prediction model (CAUSE.CATEGORY, CLIMATE.REGION, YEAR, OUTAGE.START, POPULATION, MONTH, and U.S._STATE), as well as the response variable OUTAGE.DURATION.

For the remaining columns not used in modeling, I used a more standard imputation technique. I only chose to use this on the remaining columns becuase using it on the features for the predictive model might introduce bias or not reflect real patterns in the data, especially since the missingness isn't random. 
- Numerical columns were imputed using the median value of each column. Compared to using mean, median is more robust to outliers and maintains central tendency, making it suitable for skewed and distrubutions with noise.
- Categorical columns were imputed using mode. This maintains logical consistency, ensuring the imputed value reflect the most common real-world obseved value.


***

# Framing a Prediction Problem
The prediction problem is to forecast the **duration of a power outage**, specifically the OUTAGE.DURATION column, which represents the length of time a power outage lasts(measured in minutes). This is a **regression problem** because we are predicting a **continuous numerical** outcome that represents the difference in minutes.

### Why Outage Duration?
The **OUTAGE.DURATION** variable is chosen because understanding the length of outages can help utilities better plan for anticipating resource needs, improving maintainance schedules, and estimating potential impact on customers. Overall, it will help improve service reliability.

### Evaluation Metric
I will use **Mean Absolute Error (MAE)** to evaluate performance, because it’s easy to interpret in real-world terms (minutes) and is robust to outliers. This metric treats all errors equally, regardless of whether they are above or below the predict. 

I am chosing Mean Absolute Error over Mean Squared error because it is less sensitive to exteme outliers, which is important since a few unusually long outages could skew MSE. We care more about average accuracy over punishing rare outliers here. Additionally, the units for MSE would be minutes sqaured, making it harder to interpret. 


### Justification for Features at Prediction Time

All features used in training are ones that would be available at the time the outage is reported. 
- Known **before** the outage occurs: 'YEAR', 'MONTH', 'POPULATION', 'U.S._STATE', 'CLIMATE.REGION', 'CLIMATE.CATEGORY'
- Known **immediately** at the start of the outage: 'CAUSE.CATEGORY', 'OUTAGE.START'

I intentionally excluded any features that would only be known after the outage is resolved (such as restoration time, total demand lost, or post-outage metrics) because including them would lead to data leakage and artificially inflate performance of the model. This ensure the model reflects a realistic prediction scenario and could be used to estimate outage duration in real-time as soon as an outage begins.

***

# Baseline Model

For the baseline model, I used **linear regression** on two features: **'YEAR'** and **'U.S._State'**. These features were chosen because they are both readily available at the time of predication and they serve as basic glimpses into the temporal and geographic context.

- Quantitative feature (1):
    - YEAR: A numerical variable indicating the year the outage occurred.
- Nominal feature (1):
    - U.S._STATE: A categorical variable representing the U.S. state where the outage took place. This was encoded using OneHotEncoder within a ColumnTransformer to make it usable in the linear model.
 
A pipeline was used to combine preprocessing and regression in a single step, ensuring a clean and repeatable transformation.

#### Performance Evaluation
The baseline model achieved a **Mean Absolute Error (MAE) of 2943.65 minutes** on the test set. This translates to an average error of about 49 hours, indicating the model provides a rough estimate of the outage duration. While the model is not highly accurate, it establishes a reasonable baseline. The purpose of this model is to provide a benchmark for evaluating more sophisticated models. I would not consider this model “good,” but it is appropriate as a starting point.

***

# Final Model

### New Engineered Features
- **'OUTAGE.HOUR'** – Extracted from 'OUTAGE.START', capturing the hour an outage began. Outages at night may face slower restoration times due to lower staffing or delayed detection.
- **'LOG.POPULATION'** – A log-transformed version of 'POPULATION'. This transformation reduces skew caused by large population values and helps the model interpret the relationship more linearly.
- **'SEASON'** – Derived from the 'MONTH' column. Seasonal patterns (like hurricanes in summer or ice storms in winter) can influence the likelihood and severity of outages.

All three features were added using a FunctionTransformer in a Pipeline to ensure clean integration and reproducibility.



### Features used for the model

| Feature Name        | Type          | Reason for Inclusion                                       |
|---------------------|---------------|-----------------------------------------------------------------|
| `CAUSE.CATEGORY`    | Nominal       | Represents primary reason for the outage which directly impacts duration             |
| `CLIMATE.REGION`    | Nominal       | Captures geographic weather trends across regions                                    |
| `SEASON`            | Nominal       | Derived from month; helps identify seasonal patterns                                |
| `U.S._STATE`        | Nominal       | Accounts for geographic and infrastructure differences between states               |
| `CLIMATE.CATEGORY`  | Nominal       | Broader climate patterns (warm/normal/cold) that affect risk                        |
| `YEAR`              | Quantitative  | Captures long-term trends or changes in infrastructure and policy             |
| `OUTAGE.HOUR`       | Quantitative  | Reflects time of day patterns                                                     |
| `LOG.POPULATION`    | Quantitative  | Log-transformed to reduce skew; population reflects urbanization which may affect duration |



### Model Comparison and Hyperparameter Tuning
I tested three regression models:
- Linear Regression
- Ridge Regression, with a grid search over alpha values to control regularization
- Random Forest Regressor, with a grid search over n_estimators, max_depth, and min_samples_leaf

**GridSearchCV with a 5-fold cross-validation** was used to find the best hyperparameters for Ridge and Random Forest. All models were trained and evaluated on the same X_train and X_test split to ensure a fair comparison with the baseline model.

<iframe 
  src="assets/model-comparison.html" 
  width="1000" 
  height="200" 
  frameborder="0">
</iframe>


The **Random Forest Regressor achieved the lowest Mean Absolute Error of 2469.26 minutes**. This tree-based ensemble model can capture non-linear patterns and interactions that linear models may miss. I tuned three key hyperparameters:
- n_estimators: number of trees in the forest ([50, 100])
- max_depth: maximum depth of each tree ([10, 15])
- min_samples_leaf: minimum samples in each leaf ([1, 5])

Using GridSearchCV with 5-fold cross-validation, the optimal configuration was:
- n_estimators = 100
- max_depth = 15-
- min_samples_leaf = 5

### Performance Comparison
- Baseline MAE: 2943.65 minutes
- Random Forest MAE: 2469.26 minutes


The **Random Forest model** outperformed the baseline linear regression model with an approximate **16.12% reduction** in prediction error(nearly 7.9 hours of improvement on average). This improvement is meaningful because it demonstrates that leveraging more informative and thoughtfully engineered features—such as seasonality, population size, and time-of-day—can better capture the complexity of power outage durations. The model's lower MAE indicates more reliable and actionable predictions, which are critical for utilities seeking to optimize response strategies and improve service reliability.

***




<hr>

<div style="text-align: center; font-size: 14px; color: #666; padding: 20px;">
  Created by <a href="https://github.com/anvitag" target="_blank">Anvita Gollu</a> |
  <a href="https://github.com/anvita-g/Power-Outage-Prediction" target="_blank">Project Repository</a>
</div>


<link rel="stylesheet" href="{{ '/assets/custom.css' | relative_url }}">