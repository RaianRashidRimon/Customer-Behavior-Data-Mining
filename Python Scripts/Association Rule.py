import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Load CSV (hardcoded path)
file_path = r"D:\Academics\Semesters\4-2\Lab\Data Mining and Big Data Analysis Laboratory\Experiment Reports\Support Dataset\Customer Data (Categorical).csv"
df = pd.read_csv(file_path)

# Drop target column if already included in rules
# or keep it to allow mining rules like: conditions => Will_purchase=No
# In WEKA it was included
# So we keep it here too

# Ensure all columns are treated as strings
df = df.astype(str)

# One-hot encode all categorical features (like in WEKA)
df_encoded = pd.get_dummies(df)

# Apply Apriori algorithm (min_support = 0.1 to match WEKA)
frequent_itemsets = apriori(df_encoded, min_support=0.1, use_colnames=True)

# Generate association rules (min_confidence = 0.9)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.9)

# Filter to only rules with at least 2 items in antecedent (like WEKA's deeper levels)
rules = rules[rules['antecedents'].apply(lambda x: len(x) >= 2)]

# Sort by confidence descending
rules = rules.sort_values(by='confidence', ascending=False)

# Show top rules
print("Top Association Rules (min_support=0.1, min_conf=0.9):\n")
for idx, row in rules.head(10).iterrows():
    ant = ', '.join(row['antecedents'])
    con = ', '.join(row['consequents'])
    print(f"{ant} ==> {con}  [conf: {row['confidence']:.2f}, lift: {row['lift']:.2f}]")
