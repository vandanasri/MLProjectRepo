# Findings — Project 1: EDA on the Titanic Dataset
  Full code and plots for every finding below are in `Titanic.ipynb`.

## Dataset Overview
**1.** The dataset has 891 rows and 12 columns.\
**2.** These are columnc present in titanic dataset ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']\
**3.** 'PassengerId' is fully unique. It is just a row counter - **Action:** exclude `PassengerId` from any feature set.\
**4.** All column names and all columns are labelled and unique.\
**5.** In Total 12 columns there is are => dtypes: float64(2), int64(5), object(5) .\
**6.** List of numerical columns [`PassengerId`, `Survived`, `Pclass`, `Age`, `SibSp`, `Parch`, `Fare`]\
**7.** List of categorical columns [`Name`, `Sex`, `Ticket`, `Cabin`, `Embarked`]

## Data License
One row represents one passenger with a *known, labelled* survival outcome —
this is the 891-row Kaggle competition training extract, not the full ~2,224-person
Titanic manifest (passengers + crew) — so any survival-rate figure quoted from this
file (e.g. 38.4%) describes this specific labelled subset, not "the" historical
Titanic survival rate as a whole. **Action:** caveat all summary statistics as
describing this dataset, not the full historical event, in any write-up.

## Data Quality
- Missing Values\
**1.** There are total 866 missing values in Titanic dataset.\
**2.** Only 3 columns [`Cabin`, `Age`, `Embarked`] which has missing values.\
**3.** `Cabin` has the most 687 missing data with almost 77.1% of missing data.\
**4.** `Age` has 177 missing data with 19.87% in entire 891 rows.\
**5.** `Embarked` has least 2 missing values with 0.22%.\
**6.** If we compare Cabin data according to Pclass, i.e which passenger `pclass` has most `cabin` missing data. Then it is found that  Pclass has 3 classes [1, 2, 3]. Now, getting the missing values of cabin according to passenger class. According to the observation Pclass-3 has the most missing cabin values almost 97.5, then pclass-2 has 91.3 missing cabins and pclass-1 has the least missing cabin values 18.5.\
**7.** After comparing Age according to the `Pclass`, the findings are - 3 class passengers has most missing values 27.7, 2 class pessengers has least missing values 5.98 and 1 class has 13.89 missing values. Titanic has highest numbers of Young adult passengers.\
  
- Duplicated rows and duplicated ids\
  No duplicate rows and no duplicate ids — this dataset does not need de-duplication.\\
- As per the findings, there is no impossible values in the dataset.\
- Sentinel data ==> Fifteen passengers have `Fare == 0.0` which is pretending to be Sentinel "UNKNOWN" data. Everyone is male and 4 of them hold non-numeric ticket 'LINE'.\
- Wrong datatype ==> Pclass [1,2,3] and Survived [0,1] (died/survived) are stored as int64, but both are categorical, not continuous quantities — Pclass is ordinal, Survived is binary.\
- In the Titanic data, the highest was Parch at 76.1% — high, but that's just normal class imbalance for a count variable (most people didn't travel with parents/children), not a "useless column" problem. So the finding there was: no constant or near-constant columns exist — which itself is worth stating explicitly rather than assuming.\

## Target Analysis
- `Survived` is a sensible classification target.\
- It's the value the Kaggle competition itself asks you to predict.\
- There are total 891 passengers travelling in Titanic. There is no missing value in Survived data. After finding - 342 passengers survived and rest 549 not survived.\
- 61.6% died and 38.4% survived, so a model that always predicts "died" already.
