# Findings — Project 1: EDA on the Titanic Dataset
  Full code and plots for every finding below are in `Titanic.ipynb`.

## Dataset
**1.** The dataset has 891 rows and 12 columns.\
**2.** These are columnc present in titanic dataset ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']\
**3.** 'PassengerId' is fully unique. It is just a row counter - **Action:** exclude `PassengerId` from any feature set.\
**4.** All column names and all columns are labelled and unique.\
**5.** In Total 12 columns there is are => dtypes: float64(2), int64(5), object(5) .\
**6.** List of numerical columns ['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']\
**7.** List of categorical columns ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']

## Data License
**1.** One row represents one passenger with a *known, labelled* survival outcome —
this is the 891-row Kaggle competition training extract, not the full ~2,224-person
Titanic manifest (passengers + crew) — so any survival-rate figure quoted from this
file (e.g. 38.4%) describes this specific labelled subset, not "the" historical
Titanic survival rate as a whole. **Action:** caveat all summary statistics as
describing this dataset, not the full historical event, in any write-up.

