## Problem Statement

#### Introduction:
Financial institutions, especially banks, rely on accurate assessments of credit risk to make informed decisions regarding loan approvals, risk management, and customer relationship management. In this project, we address the challenge faced by a bank that has collected two distinct datasets: one sourced from CIBIL (Credit Information Bureau India Limited), an external credit bureau, and another from its internal records. The CIBIL dataset comprises credit-related information about customers sourced from external credit bureaus, while the internal dataset contains various financial and demographic attributes of customers maintained internally by the bank.

#### Objective:
The primary objective of this project is to leverage the information from both datasets to develop predictive models that assess credit risk and predict customer creditworthiness accurately. By doing so, the bank aims to enhance its decision-making process, streamline loan approvals, improve risk management strategies, and enhance customer relationship management.

#### Challenges:
1. **Data Integration**: Combining and reconciling information from two distinct datasets with different data formats, naming conventions, and data quality present challenges.

2. **Feature Engineering**: Identifying relevant features and engineering new ones from the combined dataset is essential for improving model performance and accuracy.

3. **Imbalanced Data**: Imbalanced data, where the number of good credit customers significantly outweighs the number of bad credit customers, can lead to biased models. Addressing this imbalance is crucial for building robust credit risk assessment models.

4. **Model Interpretability**: Ensuring that the developed predictive models are interpretable and provide insights into the factors influencing credit risk assessment is essential for decision-making and regulatory compliance.

5. **Regulatory Compliance**: Regulatory guidelines mandate that credit risk assessment models do not discriminate against protected groups. Ensuring compliance with these regulations is paramount.

#### Proposed Solution:
To address these challenges, we propose the following approach:

- **Data Preprocessing**: Cleanse, preprocess, and integrate the CIBIL and internal datasets to create a unified dataset suitable for model training.

- **Feature Selection and Engineering**: Identify relevant features, handle missing values, encode categorical variables, and engineer new features to improve model performance.

- **Model Development**: Develop and train predictive models using machine learning algorithms such as logistic regression, random forests, or gradient boosting machines.

- **Model Evaluation**: Evaluate the performance of the models using appropriate evaluation metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.

- **Model Interpretation**: Interpret the trained models to understand the factors influencing credit risk assessment and customer creditworthiness.

- **Deployment**: Deploy the developed models into production for real-time credit risk assessment and decision-making.

#### Data Dictionaries:
#### CIBIL Dataset

| Column Name                | Description                                                   |
|----------------------------|---------------------------------------------------------------|
| PROSPECTID                 | Unique identifier for each customer                           |
| Total_TL                   | Total number of credit lines                                  |
| Tot_Closed_TL              | Total number of closed credit lines                           |
| Tot_Active_TL              | Total number of active credit lines                           |
| Total_TL_opened_L6M        | Total number of credit lines opened in the last 6 months      |
| Tot_TL_closed_L6M          | Total number of credit lines closed in the last 6 months      |
| pct_tl_open_L6M            | Percentage of credit lines opened in the last 6 months        |
| pct_tl_closed_L6M          | Percentage of credit lines closed in the last 6 months        |
| pct_active_tl              | Percentage of active credit lines                             |
| pct_closed_tl              | Percentage of closed credit lines                             |
| Total_TL_opened_L12M       | Total number of credit lines opened in the last 12 months     |
| Tot_TL_closed_L12M         | Total number of credit lines closed in the last 12 months     |
| pct_tl_open_L12M           | Percentage of credit lines opened in the last 12 months       |
| pct_tl_closed_L12M         | Percentage of credit lines closed in the last 12 months       |
| Tot_Missed_Pmnt            | Total number of missed payments                               |
| Auto_TL                    | Total number of auto loans                                    |
| CC_TL                      | Total number of credit card loans                             |
| Consumer_TL                | Total number of consumer loans                                |
| Gold_TL                    | Total number of gold loans                                    |
| Home_TL                    | Total number of home loans                                    |
| PL_TL                      | Total number of personal loans                                |
| Secured_TL                 | Total number of secured loans                                 |
| Unsecured_TL               | Total number of unsecured loans                               |
| Other_TL                   | Total number of other types of loans                          |
| Age_Oldest_TL              | Age of the oldest credit line                                 |
| Age_Newest_TL              | Age of the newest credit line                                 |

#### Bank Internal Dataset

| Column Name                        | Description                                                |
|------------------------------------|------------------------------------------------------------|
| PROSPECTID                         | Unique identifier for each customer                        |
| time_since_recent_payment          | Time since the most recent payment                         |
| time_since_first_deliquency        | Time since the first delinquency                           |
| time_since_recent_deliquency       | Time since the most recent delinquency                     |
| num_times_delinquent               | Number of times the customer has been delinquent           |
| max_delinquency_level              | Maximum level of delinquency                                |
| max_recent_level_of_deliq          | Maximum recent level of delinquency                        |
| num_deliq_6mts                     | Number of delinquencies in the last 6 months               |
| num_deliq_12mts                    | Number of delinquencies in the last 12 months              |
| num_deliq_6_12mts                  | Number of delinquencies in the last 6-12 months            |
| max_deliq_6mts                    | Maximum delinquency in the last 6 months                   |
| max_deliq_12mts                   | Maximum delinquency in the last 12 months                  |
| num_times_30p_dpd                 | Number of times 30+ days past due                          |
| num_times_60p_dpd                 | Number of times 60+ days past due                          |
| num_std                           | Number of standard loans                                    |
| num_std_6mts                      | Number of standard loans in the last 6 months              |
| num_std_12mts                     | Number of standard loans in the last 12 months             |
| num_sub                           | Number of substandard loans                                 |
| num_sub_6mts                      | Number of substandard loans in the last 6 months           |
| num_sub_12mts                     | Number of substandard loans in the last 12 months          |
| num_dbt                           | Number of doubtful loans                                    |
| num_dbt_6mts                      | Number of doubtful loans in the last 6 months              |
| num_dbt_12mts                     | Number of doubtful loans in the last 12 months             |
| num_lss                           | Number of loss loans                                        |
| num_lss_6mts                      | Number of loss loans in the last 6 months                  |
| num_lss_12mts                     | Number of loss loans in the last 12 months                 |
| recent_level_of_deliq             | Recent level of delinquency                                |
| tot_enq                           | Total number of inquiries                                  |
| CC_enq                            | Number of credit card inquiries                            |
| CC_enq_L6m                        | Number of credit card inquiries in the last 6 months       |
| CC_enq_L12m                       | Number of credit card inquiries in the last 12 months      |
| PL_enq                            | Number of personal loan inquiries                          |
| PL_enq_L6m                        | Number of personal loan inquiries in the last 6 months     |
| PL_enq_L12m                       | Number of personal loan inquiries in the last 12 months    |
| time_since_recent_enq             | Time since the most recent inquiry                         |
| enq_L12m                          | Number of inquiries in the last 12 months                  |
| enq_L6m                           | Number of inquiries in the last 6 months                   |
| enq_L3m                           | Number of inquiries in the last 3 months                   |
| MARITALSTATUS                     | Marital status of the customer                             |
| EDUCATION                         | Education level of the customer                             |
| AGE                               | Age of the customer                                         |
| GENDER                            | Gender of the customer                                      |
| NETMONTHLYINCOME                  | Net monthly income of the customer                         |
| Time_With_Curr_Empr               | Time with the current employer                              |
| pct_of_active_TLs_ever            | Percentage of active credit lines ever                     |
| pct_opened_TLs_L6m_of_L12m        | Percentage of credit lines opened in the last 6 months of total credit lines opened in the last 12 months |
| pct_currentBal_all_TL             | Percentage of current balance of all credit lines          |
| CC_utilization                    | Credit card utilization                                    |
| CC_Flag                           | Flag indicating if the customer has a credit card           |
| PL_utilization                    | Personal loan utilization                                  |
| PL_Flag                           | Flag indicating if the customer has a personal loan         |
| pct_PL_enq_L6m_of_L12m           | Percentage of personal loan inquiries in the last 6 months of total personal loan inquiries in the last 12 months |
| pct_CC_enq_L6m_of_L12m           | Percentage of credit card inquiries in the last 6 months of total credit card inquiries in the last 12 months |
| pct_PL_enq_L6m_of_ever           | Percentage of personal loan inquiries in the last 6 months of total personal loan inquiries ever |
| pct_CC_enq_L6m_of_ever           | Percentage of credit card inquiries in the last 6 months of total credit card inquiries ever |
| max_unsec_exposure_inPct         | Maximum unsecured exposure as a percentage of net monthly income |
| HL_Flag                           | Flag indicating if the customer has a home loan             |
| GL_Flag                           | Flag indicating if the customer has a gold loan             |
| last_prod_enq2                   | Last product inquiry type                                  |
| first_prod_enq2                  | First product inquiry type                                 |
| Credit_Score                     | Credit score of the customer                               |
| Approved_Flag                    | Flag indicating if the customer's application was approved |


