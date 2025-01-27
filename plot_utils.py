import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_time(ccds):
    # Plot the 'Time' feature
    plt.figure(figsize=(10, 6))
    sns.histplot(ccds['Time'], kde=True, color='#1f77b4')
    plt.title('Distribution of Time Feature')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()

def plot_anon_feats(ccds):
    # Features V1 to V28
    anonymized_features = [f"V{i}" for i in range(1, 29)]

    # Create subplots
    fig, axes = plt.subplots(nrows=4, ncols=7, figsize=(8, 21), tight_layout=True)
    axes = axes.flatten()

    # Plot distribution for each feature
    for i, feature in enumerate(anonymized_features):
        sns.histplot(ccds[feature], kde=True, ax=axes[i], color='#1f77b4')
        axes[i].set_title(f'Distribution of {feature}')
        axes[i].set_xlim([ccds[feature].min(), ccds[feature].max()])

    # Remove any empty subplots
    for j in range(len(anonymized_features), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def pie_chart(ccds):
    # Count occurrences of each class (0: Non-fraudulent, 1: Fraudulent)
    class_counts = ccds['Class'].value_counts()

    # Define labels and colors
    labels = ['Non-fraudulent', 'Fraudulent']
    colors = ['#1f77b4', '#ff7f0e']

    # Plot the pie chart with adjusted start angle
    plt.figure(figsize=(10, 6))
    plt.pie(class_counts, labels=labels, colors=colors, autopct='%1.2f%%', startangle=180)
    plt.title('Proportion of Fraudulent vs. Non-Fraudulent Transactions')
    plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
    plt.show()
