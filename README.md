# LabFlow

# Lab Randomizer & Statistical Analysis Tool

A web-based, assumption-aware statistical analysis tool designed for wet-lab researchers.
This project provides an end-to-end workflow for experimental data analysis, including
randomization, distribution checks, outlier handling, appropriate test selection,
and post-hoc comparisons — without requiring advanced statistical or programming knowledge.
-----------------------------------------------------------------------------------------------------------------------------------
Disclaimer - 
This tool is intended to assist in statistical analysis and decision-making.
It does not replace consultation with a statistician for complex experimental designs.
-----------------------------------------------------------------------------------------------------------------------------------
NOTE- This project is licensed under the Apache License 2.0.
Please cite LabFlow Github repository if you use this work in academic work. 
-----------------------------------------------------------------------------------------------------------------------------------
## Installation & Usage

[![DOI](https://zenodo.org/badge/1115710912.svg)](https://doi.org/10.5281/zenodo.18046274)

```bash
Live link
v1.1.0
https://labflow25.streamlit.app/
```
-----------------------------------------------------------------------------------------------------------------------------------
##Features

- CSV-based data input
- Automatic distribution analysis (normal vs non-normal) and distribution plots download in vector form. 
- Outlier detection and transparent removal reporting
- Assumption-aware test selection:
  - ANOVA / Kruskal–Wallis
- Automatic post-hoc selection:
  - Tukey HSD
  - Games–Howell
  - Dunn (Bonferroni-adjusted)
  - Bonferroni pairwise comparisons
- Clear p-value formatting (scientific notation for very small values)
- Interactive visualizations
- Web-based interface (Streamlit)

---

## Typical Workflow

1. Upload experimental data as a CSV file  
2. Inspect distribution and detect outliers  
3. Optionally remove outliers (with full audit trail)  
4. Run global statistical test  
5. Automatically select and run the best post-hoc analysis  
6. Interpret results with clear statistical output  

---

## Input Data Format

The input CSV file must contain **two columns**:

| Column | Description |
|------|------------|
| Group | Experimental group or condition |
| Value | Measured numeric value |

NOTE - 
Information on the "Groups" column and how to use it. - There should be two columns in your csv file, one labelled as "group" and the other labelled as "value". In the group column, you can mention the types that you are processing, for eg. - sample, treatment, control. All the types that you mention will be treated as different categories. If you want to process different samples, then label all of them as "sample" only because mentioning them as sample1, sample2, etc will lead to them being mentioned as different categories. 
I have provided the below example. 
Example:
```csv
Group,Value
Control,5.2
Control,5.4
Treatment,8.1
Treatment,8.3
sample, 1.2
sample, 2.1
sample, 1.1
sample, 1.4
