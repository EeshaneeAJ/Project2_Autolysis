import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print("✅ Data loaded successfully.")
        return df
    except Exception as e:
        print(f"❗ Error loading data: {e}")
        sys.exit(1)

def analyze_data(df, output_dir):
    print("🔎 Performing data analysis...")

    # Data Overview
    with open(f"{output_dir}/README.md", "w") as f:
        f.write("## Data Overview\n")
        f.write("### Data Info\n")
        f.write(str(df.info()) + "\n\n")
        f.write("### Summary Statistics\n")
        f.write(str(df.describe()) + "\n\n")
        f.write("### Missing Values\n")
        f.write(str(df.isnull().sum()) + "\n\n")
    print("📊 Analysis results saved to README.md")

def plot_correlation(df, output_dir):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig(f"{output_dir}/correlation_heatmap.png")
    print("🖼️ Correlation heatmap saved.")

def plot_distributions(df, output_dir):
    for column in df.select_dtypes(include=np.number).columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[column], kde=True)
        plt.title(f'Distribution of {column}')
        plt.savefig(f"{output_dir}/{column}_distribution.png")
        print(f"🖼️ Distribution plot for {column} saved.")

def main():
    if len(sys.argv) != 2:
        print("❗ Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if "goodreads" in file_path.lower():
        output_dir = "goodreads"
    elif "happiness" in file_path.lower():
        output_dir = "happiness"
    elif "media" in file_path.lower():
        output_dir = "media"
    else:
        print("❗ Invalid dataset. Please use goodreads.csv, happiness.csv, or media.csv.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    
    df = load_data(file_path)
    analyze_data(df, output_dir)
    plot_correlation(df, output_dir)
    plot_distributions(df, output_dir)

if __name__ == "__main__":
    main()
