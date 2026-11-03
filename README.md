# Yuva Task 3 - Student Performance Data Analysis

## Dataset
This project uses the UCI Machine Learning Repository's Student Performance dataset, specifically `student-por.csv` (Portuguese language course). The original UCI collection contains 649 records and 33 attributes across the two course files. Source: https://archive.ics.uci.edu/dataset/320/student%2Bperformance

## Run in VS Code
1. Install Python and the VS Code Python extension.
2. Open this project folder in VS Code.
3. Open Terminal > New Terminal.
4. Run: `python -m pip install -r requirements.txt`
5. Open `Student_Performance_Analysis.ipynb` and choose your Python kernel, then run cells from top to bottom. Alternatively run `python student_performance_analysis.py`.
6. Internet access is required on the first run so the script can download the UCI ZIP file.
7. Check the `outputs` folder for six graphs, the cleaned CSV and summary CSV.

## Important
- The dataset uses semicolon-separated values; the code reads it with `sep=';'`.
- The project-defined performance bands are descriptive labels, not official UCI labels.
- Correlation does not prove causation.
- Write your own observations from the actual outputs before submitting.
