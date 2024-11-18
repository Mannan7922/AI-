import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
data = pd.read_csv('ED.csv', low_memory=False)

# Display the first few rows of the dataset
print("Sample data:")
print(data.head())

# Data cleaning and preprocessing
# Drop duplicate or irrelevant columns
columns_to_drop = ['Country', 'Week number', 'Gender', 'Year', 'Variable']
data = data.drop(columns=columns_to_drop)

# Rename columns for consistency
data.rename(columns={'AGE': 'Age_Group'}, inplace=True)

# Target column and preprocessing
target_column = 'Value'
if target_column not in data.columns:
    print(f"Error: '{target_column}' column not found.")
else:
    # Defining features and target
    X = data.drop(target_column, axis=1)
    y = pd.cut(data[target_column], bins=[-float('inf'), 0, 50, 100, float('inf')],
               labels=['negative', 'low', 'medium', 'high'])

    # One-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # First Rule-Based Method
    def first_rule_based_method(row, target_value):
        if row.get('Age_Group_Y_GE65', 0) == 1 and target_value > 100:
            return 'high'
        elif row.get('Age_Group_Y0T44', 0) == 1 and target_value < 50:
            return 'low'
        else:
            return 'medium'

    # Apply first rule-based method
    y_pred_rule_based_1 = X_test.apply(lambda row: first_rule_based_method(row, data.loc[row.name, target_column]), axis=1)
    y_pred_rule_based_1 = pd.Categorical(y_pred_rule_based_1, categories=['negative', 'low', 'medium', 'high'])

    # Evaluate first rule-based method
    accuracy_1 = accuracy_score(y_test, y_pred_rule_based_1)
    precision_1 = precision_score(y_test, y_pred_rule_based_1, average='weighted', zero_division=1)
    recall_1 = recall_score(y_test, y_pred_rule_based_1, average='weighted', zero_division=1)
    f1_1 = f1_score(y_test, y_pred_rule_based_1, average='weighted', zero_division=1)

    # Second Rule-Based Method
    def second_rule_based_method(row, target_value):
        if row.get('Age_Group_Y_GE65', 0) == 1 and target_value < 0:
            return 'negative'
        elif row.get('Age_Group_Y0T44', 0) == 1 and target_value >= 50:
            return 'high'
        elif row.get('Age_Group_Y_GE65', 0) == 1 and target_value <= 50:
            return 'medium'
        else:
            return 'low'

    # Apply second rule-based method
    y_pred_rule_based_2 = X_test.apply(lambda row: second_rule_based_method(row, data.loc[row.name, target_column]), axis=1)
    y_pred_rule_based_2 = pd.Categorical(y_pred_rule_based_2, categories=['negative', 'low', 'medium', 'high'])

    # Evaluate second rule-based method
    accuracy_2 = accuracy_score(y_test, y_pred_rule_based_2)
    precision_2 = precision_score(y_test, y_pred_rule_based_2, average='weighted', zero_division=1)
    recall_2 = recall_score(y_test, y_pred_rule_based_2, average='weighted', zero_division=1)
    f1_2 = f1_score(y_test, y_pred_rule_based_2, average='weighted', zero_division=1)

    # Print performance metrics
    print("\nFirst Rule-Based Method Performance:")
    print(f"Accuracy: {accuracy_1:.2f}")
    print(f"Precision: {precision_1:.2f}")
    print(f"Recall: {recall_1:.2f}")
    print(f"F1 Score: {f1_1:.2f}")

    print("\nSecond Rule-Based Method Performance:")
    print(f"Accuracy: {accuracy_2:.2f}")
    print(f"Precision: {precision_2:.2f}")
    print(f"Recall: {recall_2:.2f}")
    print(f"F1 Score: {f1_2:.2f}")

    # Plotting the performance metrics
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    scores_1 = [accuracy_1, precision_1, recall_1, f1_1]
    scores_2 = [accuracy_2, precision_2, recall_2, f1_2]

    x = range(len(metrics))

    plt.figure(figsize=(10, 6))
    plt.bar(x, scores_1, width=0.4, label='Rule-Based Method 1', color='lightblue')
    plt.bar([p + 0.4 for p in x], scores_2, width=0.4, label='Rule-Based Method 2', color='lightgreen')

    # Adding labels to the bars for clarity
    for i, v in enumerate(scores_1):
        plt.text(i - 0.1, v + 0.02, f"{v:.2f}", color='blue', ha='center')
    for i, v in enumerate(scores_2):
        plt.text(i + 0.3, v + 0.02, f"{v:.2f}", color='green', ha='center')

    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.title('Performance Comparison of Rule-Based Methods')
    plt.xticks([p + 0.2 for p in x], metrics)
    plt.ylim(0, 1)
    plt.legend()

    plt.tight_layout()
    plt.show()
