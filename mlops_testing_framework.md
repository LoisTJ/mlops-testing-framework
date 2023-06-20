# MLOps Testing Framework (Draft)

- [MLOps Testing Framework (Draft)](#mlops-testing-framework-draft)
	- [1. Introduction to MLOps Testing](#1-introduction-to-mlops-testing)
		- [1.1 What is MLOps Testing](#11-what-is-mlops-testing)
		- [1.2 Why is MLOps Testing Important](#12-why-is-mlops-testing-important)
	- [2. Types of MLOps Testing](#2-types-of-mlops-testing)
		- [2.1 Code Testing](#21-code-testing)
			- [2.1.1 Unit Testing](#211-unit-testing)
			- [2.1.2 Integration Testing](#212-integration-testing)
			- [2.1.3 System Testing](#213-system-testing)
		- [2.2 Data Testing](#22-data-testing)
			- [2.2.1 Data Quality Testing](#221-data-quality-testing)
			- [2.2.2 Data Bias Testing](#222-data-bias-testing)
			- [2.2.3 Data Drift Testing](#223-data-drift-testing)
		- [2.3 Model Testing](#23-model-testing)
			- [2.3.1 Model Accuracy Testing](#231-model-accuracy-testing)
			- [2.3.2 Model Bias Testing](#232-model-bias-testing)
			- [2.3.3 Model Robustness Testing](#233-model-robustness-testing)
	- [3. Stages of MLOps Testing](#3-stages-of-mlops-testing)
		- [3.1 Model Development](#31-model-development)
		- [3.2 Model Training](#32-model-training)
		- [3.3 Model Deployment](#33-model-deployment)
		- [3.4 Production Environment](#34-production-environment)
	- [4. Tools and Techniques](#4-tools-and-techniques)
		- [4.1 Frameworks](#41-frameworks)
	- [5. Roles and Responsibilities](#5-roles-and-responsibilities)
		- [5.1 Data Scientist](#51-data-scientist)
		- [5.2 Data Engineer](#52-data-engineer)
		- [5.3 DevOps Engineer/ML Engineer](#53-devops-engineerml-engineer)
	- [6. Challenges](#6-challenges)
		- [6.1 Data Quality](#61-data-quality)
		- [6.2 Data Bias](#62-data-bias)
		- [6.3 Model Drift](#63-model-drift)
	- [7. Best Practices](#7-best-practices)
		- [7.1 Test-driven Development](#71-test-driven-development)
		- [7.2 CI/CD Pipeline](#72-cicd-pipeline)
		- [7.3 Define Testing Strategy](#73-define-testing-strategy)
	- [Conclusion](#conclusion)
	- [References](#references)

## 1. Introduction to MLOps Testing
### 1.1 What is MLOps Testing
MLOps Testing is a set of processes and activities that ensure the quality of machine learning (ML) models and systems. It is a crucial process that needs to be integrated into multiple stages of ML model development and post-deployment.

As ML model generated insights are increasing allowed to support key business decisions and public sector policies, the adoption of MLOps Testing becomes more important than ever.

Through the incorporation of MLOps Testing, data professionals stand to benefit from **increased model and code quality**; **reduced risk of having unethical or unexpected ML model behaviours** in production environment; and **improved efficiency of the ML model lifecycle**.

### 1.2 Why is MLOps Testing Important
MLOps Testing is a critical part of the MLOps lifecycle, and helps to ensure that ML models are reliable, accurate, and performant in production. 

Unknown to many, the MLOps Testing process is an ongoing process that lives with the ML model. As the model grows in both size and complexity over time, MLOps Testing helps to ensure that these growths are tested repeatedly and continue to meet required quality standards. In additino to quality checks, new changes in ML models are also tested for their compatibility and impact on the existing model's output.

There are 3 main categories of MLOps Testing:
- Code Testing
- Data Testing
- Model Testing

In the following section, we will go into greater details on each testing category, and how they work hand-in-hand to ensure a robust and performant ML model development process.

## 2. Types of MLOps Testing
### 2.1 Code Testing
Code testing ensures that the codes written to 1) Translate business logics; 2) Engineer data features; 3) Prepare and transform model training data; 4) Deploy and serve the ML model inference, are reliable and performant.

#### 2.1.1 Unit Testing
This is the lowest level of testing, as it tests only individual units of code, such as user defined functions and classes. Unit testing helps to ensure that a single building block of code functions as expected, both when handling data inputs observed in data sample, and **error handling**, when unexpected data inputs are passed into the function.

#### 2.1.2 Integration Testing
As the name suggest, integration testing is about ensuring different units of code interact with each other. For example, we often have functions that are nested and called within other functions, which implies that the nested function is expected to take inputs defined by the parent function, as well as produce outputs that can be used by the parent function. 

Some key testing areas to cater for under integration testing include: **data type errors**; **dependency errors**; **exception handling and recovery**. The list here is not exhaustive, as integration testing can expand into many more areas, such as data corruption and loss errors, when data are accidentally overwritten when passed to the next function.

#### 2.1.3 System Testing
System testing looks at the entire end-to-end ML system setup. This includes the data, the ETL code, the ML model, and the infrastructure the model runs on. System testing is usually outside of the scope for data professionals and fall under the care of QA engineers, who specialise in checking and ensuring the underlying infrastructure meets all the requirements to support the ML modelling processes.

### 2.2 Data Testing
Data testing ensures the data used for ML model training and evaluation is accurate and well-represented. ML models are as good as the data it gets, by checking for **quality**, **bias**, and **drifts** in the training and evaluation data set, we can eradicate potential downstream problems caused by a poorly explored dirty data set.

#### 2.2.1 Data Quality Testing
Data quality testing is a sizable topic on its own. There are many metrics that can be defined to analyse and grade the quality of a data set. However, it is important to note that all data sets' quality measures should be **defined based on the specific project's needs**. In other words, data quality metrics should be defined differently across different projects. 

**Note:** Here we draw attention to the fact that, the context of data quality testing discussed under MLOps Testing is different from managing data quality in a data lake, where standardised quality metrics can be defined and monitored across the lake.

Some examples of data quality checks include:
- **Data missing rate by data field**
  There are various data imputation techniques that can handle different scenarios of missing data, which can be classified into 3 main types:

	1. Missing completely at random (MCAR)
	This is when the missingness of data is non-systemtic and has no relationships with both the target and independent variables. Since there is no specific pattern or reason for the missing data, it is therefore considered less of a problem, as MCAR is then assumed not to introduce any bias to the analysis.

	MCAR data fields can be imputed using suitable techniques, depending on the variable. However, it is not that straight forward when determining whether a data field falls under the MCAR type, as this would require both statistical tests to see if the missingness of the variable is truly random, as well as a good understanding of the nature and relation, of the variable with missing data, to the rest of the other variables.

	2. Missing at random (MAR)
	This is when the missingness of data is not random, but can be explained by variables other than itself. MAR issues on a single variable (variable X) can be resolved by accounting for the other variable (variable Z) that's related to the missingness of the variable X.

	For example, if we know that in online job postings, the hiring company's name for jobs publicised by recruitment agencies are usually missing, then the missingness of the hiring company name can be considered MAR, if the type of job posting owner is observed and differentiated (Recruitment Agency vs Hiring Company). 

	3. Not missing at random (NMAR)
	This is when the missingness of the data is not random, and is only dependent on the unobserved data of the varible itself. This is a tricky situation, as NMAR cannot be resolved by including other observed independent variables. Hence, if a data set has multiple fields suffering from NMAR issues, it should be flagged out as an unsuitable/unqualified data set for ML model training and evaluation.

	For example, in a survey that asks pedestrians on how well they observe road safety when crossing, pedestrians who are used to jaywalking are less likely to report their preferences to jaywalk, so the missingness of jaywalking data would be considered NMAR.

	On data imputation techniques, it deserves a separate discussion on its own, but here are a list of techniques to consider. Let's assume you have already split the data set into the train, test, validation sets, we are interested in imputing the training data now.

     - Impute using Mean / Median values
		```python
		from sklearn.impute import SimpleImputer

		impute_mean = SimpleImputer(strategy='mean') # or you can use 'median'
		impute_mean.fit(train)
		imputed_train_df = impute_mean.transform(train)
		```

     - Impute using Mode (most common) or constant values
		```python
		from sklearn.impute import SimpleImputer

		impute_mode = SimpleImputer(strategy='most_frequent)
		impute_mode.fit(train)
		imputed_train_df = imputed_mode.transform(train)
		```

     - Impute using k-NN
		```python
		import sys
		from impyute.imputation.cs import fast_knn
		sys.setrecursionlimit(10000)

		imputed_training = fast_knn(train.values, k=30)
		```

     - Impute using multivariate imputation by chained equation (MICE) 
		```python
		from impyute.imputation.cs import mice

		imputed_training = mice(train.values)
		```

     - Impute using interpolation and extrapolation

- **Data reasonableness by data type**
  For **numerical data fields**, a reasonableness check would entail setting an expected range for the values. For example, for monthly salary of job postings online, the range of value should be approximately between 600 to 30000, anything below the range will likely be a poorly parsed hourly-wage, anything above the range is likely a poorly parsed annual salary amount.  

  For text data fields, a reasonableness check would require more knowledge of the specific data fields. For example, while all data fields related to name of people should not contain digits, data fields for email address can contain digits. Quick **assertion tests** can be written in **python** to flag out data reasonableness issues mentioned above. 

  Assertion tests such as asserting if a variable value contains alphanumerics only `.isalnum()` **Note:** " " (space) is not an alphanumeric, so if you have a variable like "Peter John", it will return "FALSE" for the assertion. For more string method assertion tests, consider reading up on `.isalpha()`,`.isascii()`,`.isdigit()`, `.isnumeric()`, and `.isspace()` as a start. **Note:** it's important to understand the difference between similar method, such as `.isdigit()` vs `.isnumeric()`. The use of `.isnumeric()` will only make a difference when the variable to be checked is a vular fraction like '⅓' (input via unicode `"U+2153"`).

- **Data validity**
  Data validity is concerned with data type, format, and size both when it's in storage at databases, and after it's pumped through the ETL pipelines. Therefore, this is a shared responsibility between the Data Engineer and Data Scientist.

  Data type can make or break a ML model, as data streams in via the ETL pipelines, an unexpected data type can raise a fatal error at the ML model feature engineering stage, or worse, stay hidden until the modelling stage and affect the model performance adversely.

- **Data consistency**
  Data sets rarely exist on their own, without any reference to existing master data. Hence, it is important to check that for data fields involving categories, types, and indices, they do not contradict or violate existing master data or reference data sets.

  One example of how data inconsistency can affect a ML model adversely, is with the HDB resale data set. For executive condominium, they are tagged as "EC" under property type. But once they pass 10 years of maturity, they now become "Private Property" and can be purchased by foreign buyers in Singapore. Let's assume 2 data sets containing records of executive condominiums, one generated when the EC is in its 10th year, another generated when the EC is in its 11th year. We will have inconsistent records in the property type column for the same exact property.

- **Data integrity**
  Data integrity checks are necessary when multiple long tables are joined to form a wide table for analysis. During data set joins, many things can go wrong as a result of misunderstanding a business rule, or the definition of a data field, or the merge method applied.

  A mistake in data set joining can easily be surfaced with a comparison of the number of null records, number of columns, number of rows, before and after the join. Common python methods include `dataframe.info()` for number of null values per variable, `dataframe.shape` for number of rows and columns of a data set, or `len(dataframe.index)` for number of rows. If you have a dataset with too many variables, you can also consider `dataframe.columns` for a print out of all the variable names.

#### 2.2.2 Data Bias Testing
**#TODO**
#### 2.2.3 Data Drift Testing
### 2.3 Model Testing
#### 2.3.1 Model Accuracy Testing
#### 2.3.2 Model Bias Testing
#### 2.3.3 Model Robustness Testing

## 3. Stages of MLOps Testing
### 3.1 Model Development
### 3.2 Model Training
### 3.3 Model Deployment
### 3.4 Production Environment

## 4. Tools and Techniques
### 4.1 Frameworks

## 5. Roles and Responsibilities
### 5.1 Data Scientist
### 5.2 Data Engineer
### 5.3 DevOps Engineer/ML Engineer

## 6. Challenges
### 6.1 Data Quality
### 6.2 Data Bias
### 6.3 Model Drift

## 7. Best Practices
### 7.1 Test-driven Development
### 7.2 CI/CD Pipeline
### 7.3 Define Testing Strategy

## Conclusion

## References
https://towardsdatascience.com/6-different-ways-to-compensate-for-missing-values-data-imputation-with-examples-6022d9ca0779