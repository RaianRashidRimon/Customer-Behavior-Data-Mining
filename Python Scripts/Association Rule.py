import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def generate_association_rules(csv_path="data.csv", min_support=0.1, min_confidence=0.7):
    try:
        df = pd.read_csv('Customer Data (Categorical).csv')
        df_encoded = pd.get_dummies(df)
        frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
        rules = rules.sort_values('confidence', ascending=False)
        return rules
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None
def main():
    csv_path = "data.csv"
    rules = generate_association_rules(csv_path)
    if rules is not None:
        print("\nAssociation Rules found:")
        print(f"Number of rules: {len(rules)}")
        print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
        rules.to_csv('association_rules_output.csv', index=False)
        print("\nRules saved to 'association_rules_output.csv'")
if __name__ == "__main__":
    main()
