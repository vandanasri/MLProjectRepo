# Findings — Project 1: EDA on the Titanic Dataset
  Full code and plots for every finding below are in `project1_eda.ipynb`.

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
**8** The Embarked column with almost no missing data — just 2 rows out of 891.
  
- Duplicated rows and duplicated id
  No duplicate rows and no duplicate ids — this dataset does not need de-duplication.
- As per the findings, there is no impossible values in the dataset.
- Sentinel data ==> Fifteen passengers have `Fare == 0.0` which is pretending to be Sentinel "UNKNOWN" data. Everyone is male and 4 of them hold non-numeric ticket 'LINE'.
- Wrong datatype ==> Pclass [1,2,3] and Survived [0,1] (died/survived) are stored as int64, but both are categorical, not continuous quantities — Pclass is ordinal, Survived is binary.
- In the Titanic data, the highest was Parch at 76.1% — high, but that's just normal class imbalance for a count variable (most people didn't travel with parents/children), not a "useless column" problem. So the finding there was: no constant or near-constant columns exist — which itself is worth stating explicitly rather than assuming.

## Target Analysis
- `Survived` is a sensible classification target.
- It's the value the Kaggle competition itself asks you to predict.
- There are total 891 passengers travelling in Titanic. There is no missing value in Survived data. After finding - 342 passengers survived and rest 549 not survived.
- 61.6% died and 38.4% survived, so a model that always predicts "died" already.
  
**Skew:** Survived has more 0s than 1s. Therefore its mean is 0.384 while its median is 0. But don't treat this as the usual continuous-variable skewness that requires a log transformation. This is mainly telling us that the target classes are imbalanced: 61.6% died and 38.4% survived. When building the model, preserve this ratio using a stratified split and don't judge the model using accuracy alone.

**Outliers:** outlier analysis in the usual sense does not apply to a binary target.
- there is no "extreme" value of 0 or 1. The honest thing to do is say so rather than force the check. 
- The nearest useful analogue is outliers in `Fare`, the most skewed numeric feature associated with the target, checked next.
- Solo travellers (family size 1) survived at 30.4%, family sizes 2–4 survived at 55–72%, and family sizes ≥5 dropped back to 0–33%. And large family rate is worse.
- Embarked: records which port a passenger boarded the Titanic from. Code Port and their servival rate:- S Southampton, England — 55.4% C Cherbourg, France - 39% Q Queenstown, Ireland - 33.7%.

## Relationships

- `Sex` is the single strongest univariate split in the dataset — 74.2%
survival for female passengers vs. 18.9% for male.
**Action:** do not present a `Sex`-driven model as a generalizable "who survives a shipwreck" rule outside this
historical context.

- `Fare` correlates with `Survived` at r = 0.26, but `Fare` also correlates far more strongly with `Pclass` at r = −0.55, and `Pclass` alone. the class-based survival gap (63% → 47% → 24% across classes 1→2→3). So `Fare` and `Pclass` are largely capturing the same underlying deck/lifeboat access effect, and crediting `Fare` in isolation risks double-counting a class effect.
**Action:** interpret `Fare` and `Pclass` together in any model, or check for multicollinearity before assuming both contribute independent signal.
  
- `FamilySize` (`SibSp + Parch + self`) has an almost-zero, non-significant linear correlation with `Survived` (Pearson r = 0.017, p = 0.62), which would wrongly suggest family size doesn't matter. But the true relationship is a strong inverted: solo travellers survived at 30.4%, families of 2–4 at 55–72%, and families of 5+ back down to 0–33%. This is the misleading-correlation case the brief specifically asks for: a linear coefficient hides a real, non-linear pattern. **Action:** enter `FamilySize` into any model as a binned categorical term (alone / small / large), not as a raw linear numeric feature.

## Risks
-*(Risk)* 134 tickets are shared by 2 or more passengers travelling together (families, groups), so rows in this dataset are not independent — a plain random train/test split can place some members of a travel group in train and others in test, leaking group-level survival patterns (families tended to survive or die together) into evaluation and inflating apparent model performance.

-*(Risk)* Several categories are effectively singletons — title values such as "Countess", "Jonkheer", "Capt", "Don", "Sir", "Lady", "Mme" each appear exactly once, and `Embarked == 'Q'` is only 8.6% of rows — so a random split, especially a small one, can strand a category entirely in the test fold with zero training examples, breaking a one-hot encoder at inference time with an unseen-category error. **Action:** bucket rare titles into an "Other" category before encoding, and use a stratified split on at least `Pclass`/`Embarked`/`Sex`.

-*(Risk)* Survival on this voyage was mechanistically produced by the evacuation policy actually followed (women/children first, class-based deck and lifeboat access) — `Sex`, `Pclass`, and `Age` were not just correlated with survival, they were causally upstream of who reached a lifeboat. A model trained here is closer to reconstructing this one incident's known, explicit prioritization policy than to learning a transferable "who survives a maritime disaster" rule. **Action:** do not present this model's feature importances as general disaster-survival factors outside the historical context of this specific ship and policy.

-*(Risk)* `Cabin`/`HasCabin` missingness is not at random and is highly redundant with `Pclass` (77% missing overall, but 97.6% missing in 3rd class vs. 18.5% in 1st) — an engineered `HasCabin` flag mostly re-encodes ticket class rather than adding independent information, so a model that appears to gain accuracy from cabin-derived features may just be double-counting the same `Pclass` signal through two columns, inflating its apparent feature importance. **Action:** if `HasCabin` is used, check its marginal contribution over `Pclass` alone before trusting its importance score.

## Next Steps

- Cast `Pclass` and `Survived` to categorical dtype; drop `PassengerId` before any
  modelling step.
- Engineer `FamilySize`, `Title`, and treat `HasCabin` cautiously given its overlap with `Pclass`.
- Use `log1p(Fare)` rather than raw `Fare`.
- Use `GroupKFold` on `Ticket` (or family surname) combined with stratification on `Survived`, rather than a plain random split, for any cross-validation.
- If `Age` is imputed for modelling, impute conditional on `Pclass`/`Title`, not with a single global value.
- Treat any resulting model's feature importances as specific to this ship and this voyage's evacuation policy, not a general theory of disaster survival (Risk 15).
- A baseline logistic regression on `Sex`, `Pclass`, and binned `FamilySize` alone should be tried first — given Finding 10 and 12, these three features likely carry most of the learnable signal, and a simple baseline should be beaten before trusting a more complex model's added value.


