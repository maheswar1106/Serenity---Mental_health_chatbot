import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, ttest_ind
import numpy as np


import os
import pandas as pd

def load_source_documents(folder_name="ghji"):
    documents = []
    categories = []
    for file_name in os.listdir(folder_name):
        # Filter only .txt files
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_name, file_name)
            print(f"Processing: {file_path}")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    category_line = [line for line in content.split("\n") if "Category:" in line]
                    if category_line:
                        category = category_line[0].split("Category:")[-1].strip()
                        categories.append(category)
                        documents.append(len(content))
            except UnicodeDecodeError as e:
                print(f"Error reading {file_name}: {e}")
                
    return pd.DataFrame({"Category": categories, "Text Length": documents})


def perform_hypothesis_testing(data):
    unique_categories = data['Category'].unique()
    for target_category in unique_categories:
        print(f"\nPerforming hypothesis testing for category: {target_category}")
        
        target_data = data[data['Category'] == target_category]
        other_categories_data = data[data['Category'] != target_category]

        if len(target_data) > 0 and len(other_categories_data) > 0:
            grouped_data = [target_data['Text Length'], other_categories_data['Text Length']]
            anova_stat, anova_p = f_oneway(*grouped_data)
            print("One-Way ANOVA Results:")
            print(f"F-statistic: {anova_stat:.4f}, P-value: {anova_p:.4f}")

            if anova_p < 0.05:
                print("Reject H0: Significant differences in text lengths between the '{}' category and others.".format(target_category))
            else:
                print("Fail to reject H0: No significant differences in text lengths between the '{}' category and others.".format(target_category))
            
            plt.figure(figsize=(10, 6))
            sns.boxplot(x="Category", y="Text Length", data=data)
            plt.title(f"Text Length Distribution for '{target_category}' Category vs Others")
            plt.xlabel("Category")
            plt.ylabel("Text Length")
            plt.legend(title="Category", labels=data['Category'].unique())
            plt.show()

            plt.figure(figsize=(10, 6))
            sns.histplot(data=data, x="Text Length", hue="Category", kde=True, bins=20)
            plt.title(f"Histogram of Text Lengths by Category ({target_category})")
            plt.xlabel("Text Length")
            plt.ylabel("Frequency")
            plt.legend(title="Category")
            plt.show()

            plt.figure(figsize=(10, 6))
            sns.violinplot(x="Category", y="Text Length", data=data)
            plt.title(f"Violin Plot of Text Length Distribution Across Categories ({target_category})")
            plt.xlabel("Category")
            plt.ylabel("Text Length")
            plt.show()

            print("\nPairwise T-Test Results:")
            p_values = np.zeros((len(unique_categories), len(unique_categories)))
            for i, cat1 in enumerate(unique_categories):
                for j, cat2 in enumerate(unique_categories):
                    if i < j:
                        group1 = data[data['Category'] == cat1]['Text Length']
                        group2 = data[data['Category'] == cat2]['Text Length']
                        t_stat, p_val = ttest_ind(group1, group2)
                        p_values[i, j] = p_val
                        print(f"{cat1} vs {cat2}: T-statistic: {t_stat:.4f}, P-value: {p_val:.4f}")

            plt.figure(figsize=(8, 6))
            sns.heatmap(p_values, annot=True, xticklabels=unique_categories, yticklabels=unique_categories, cmap="coolwarm", cbar_kws={'label': 'P-value'})
            plt.title("Pairwise T-Test P-values Heatmap")
            plt.show()

        else:
            print(f"Not enough data for the {target_category} category or other categories.")

if __name__ == "__main__":
    folder_name = "ghji"
    if os.path.exists(folder_name):
        data = load_source_documents(folder_name)
        perform_hypothesis_testing(data)
    else:
        print(f"Folder '{folder_name}' does not exist. Please ensure source documents are saved.")
