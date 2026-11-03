"""
YUVA TASK 3: REAL-WORLD STUDENT PERFORMANCE DATA ANALYSIS
Run: python student_performance_analysis.py
"""

from pathlib import Path
import zipfile
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Set up folders and download the real dataset
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ZIP_PATH = DATA_DIR / "student-performance.zip"
DATA_PATH = DATA_DIR / "student-por.csv"
UCI_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"

if not DATA_PATH.exists():
    print("Downloading the official UCI Student Performance dataset...")
    urllib.request.urlretrieve(UCI_URL, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)

# 2. Load the data
# The official UCI CSV files use a semicolon as the separator.
df = pd.read_csv(DATA_PATH, sep=";")
print("\\nFIRST FIVE ROWS")
print(df.head())
print("\\nDATASET SHAPE (rows, columns):", df.shape)
print("\\nCOLUMN INFORMATION")
df.info()

# 3. Understand and clean the data
print("\\nMISSING VALUES BEFORE CLEANING")
print(df.isnull().sum())
print("\\nDUPLICATE ROWS BEFORE CLEANING:", df.duplicated().sum())

# Remove exact duplicate rows if any are present.
df = df.drop_duplicates().copy()

# Convert grade columns to numeric values safely.
for col in ["G1", "G2", "G3"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill any missing numeric values with their column median.
numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill any missing categorical values with the most frequent value.
categorical_cols = df.select_dtypes(exclude=np.number).columns
for col in categorical_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

print("\\nMISSING VALUES AFTER CLEANING")
print(df.isnull().sum())
print("\\nDATASET SHAPE AFTER CLEANING:", df.shape)

# 4. Descriptive/statistical analysis
print("\\nSTATISTICAL SUMMARY")
print(df[["age", "studytime", "absences", "G1", "G2", "G3"]].describe().round(2))
print("\\nAVERAGE FINAL GRADE:", round(df["G3"].mean(), 2))
print("MEDIAN FINAL GRADE:", round(df["G3"].median(), 2))
print("MODE FINAL GRADE:", df["G3"].mode().iloc[0])
print("STANDARD DEVIATION OF FINAL GRADE:", round(df["G3"].std(), 2))
print("LOWEST FINAL GRADE:", df["G3"].min())
print("HIGHEST FINAL GRADE:", df["G3"].max())

# Create an easy-to-read performance category.
# This is an analysis label created for this project, not an official UCI label.
df["performance_level"] = pd.cut(
    df["G3"],
    bins=[-1, 9, 14, 20],
    labels=["Needs support (0-9)", "Moderate (10-14)", "High (15-20)"]
)
print("\\nSTUDENT COUNT BY PERFORMANCE LEVEL")
print(df["performance_level"].value_counts().sort_index())

# 5. Exploratory data analysis (EDA)
print("\\nAVERAGE FINAL GRADE BY STUDY TIME")
study_grade = df.groupby("studytime", observed=False)["G3"].mean().round(2)
print(study_grade)

print("\\nAVERAGE FINAL GRADE BY NUMBER OF PAST FAILURES")
failure_grade = df.groupby("failures")["G3"].mean().round(2)
print(failure_grade)

print("\\nAVERAGE FINAL GRADE BY GENDER")
gender_grade = df.groupby("sex")["G3"].mean().round(2)
print(gender_grade)

print("\\nCORRELATION WITH FINAL GRADE")
print(df[["studytime", "absences", "failures", "G1", "G2", "G3"]]
      .corr(numeric_only=True)["G3"].sort_values(ascending=False).round(3))

# 6. Visualizations - each graph is saved in the outputs folder.
sns.set_theme(style="whitegrid")

plt.figure(figsize=(8, 5))
sns.histplot(df["G3"], bins=15, kde=True)
plt.title("Distribution of Students' Final Grades")
plt.xlabel("Final grade (G3), out of 20")
plt.ylabel("Number of students")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_final_grade_distribution.png", dpi=150)
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(x=study_grade.index.astype(str), y=study_grade.values)
plt.title("Average Final Grade by Weekly Study Time")
plt.xlabel("Study time category (1 = less than 2 hours; 4 = over 10 hours)")
plt.ylabel("Average final grade (G3)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_studytime_vs_grade.png", dpi=150)
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="failures", y="G3")
plt.title("Final Grades by Number of Past Class Failures")
plt.xlabel("Number of past failures (dataset coding)")
plt.ylabel("Final grade (G3)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_failures_vs_grade.png", dpi=150)
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="absences", y="G3", alpha=0.65)
plt.title("School Absences and Final Grade")
plt.xlabel("Number of absences")
plt.ylabel("Final grade (G3)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_absences_vs_grade.png", dpi=150)
plt.show()

plt.figure(figsize=(9, 6))
corr = df[["studytime", "absences", "failures", "G1", "G2", "G3"]].corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Correlation Heatmap of Selected Academic Variables")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_correlation_heatmap.png", dpi=150)
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="performance_level",
              order=["Needs support (0-9)", "Moderate (10-14)", "High (15-20)"])
plt.title("Number of Students in Each Performance Category")
plt.xlabel("Performance category (project-defined)")
plt.ylabel("Number of students")
plt.xticks(rotation=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_performance_categories.png", dpi=150)
plt.show()

# 7. Save cleaned data and key summary tables
df.to_csv(OUTPUT_DIR / "cleaned_student_performance.csv", index=False)
summary = pd.DataFrame({
    "measure": ["student_records", "average_final_grade", "median_final_grade",
                "minimum_final_grade", "maximum_final_grade",
                "average_absences", "average_studytime_code"],
    "value": [len(df), round(df["G3"].mean(), 2), round(df["G3"].median(), 2),
              df["G3"].min(), df["G3"].max(), round(df["absences"].mean(), 2),
              round(df["studytime"].mean(), 2)]
})
summary.to_csv(OUTPUT_DIR / "project_summary.csv", index=False)

print("\\nPROJECT COMPLETE")
print("Graphs and result files are saved in:", OUTPUT_DIR)
print("IMPORTANT: Read each graph and write its actual observation in your report.")
