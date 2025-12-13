# LabFlow

#Lab Randomizer & Statistical Analysis Tool

A web-based, assumption-aware statistical analysis tool designed for wet-lab researchers.

This project provides an end-to-end workflow for experimental data analysis, including
randomization, distribution checks, outlier handling, appropriate test selection,
and post-hoc comparisons — without requiring advanced statistical or programming knowledge.

---

##Features

- CSV-based data input
- Automatic distribution analysis (normal vs non-normal)
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

##Typical Workflow

1. Upload experimental data as a CSV file  
2. Inspect distribution and detect outliers  
3. Optionally remove outliers (with full audit trail)  
4. Run global statistical test  
5. Automatically select and run the best post-hoc analysis  
6. Interpret results with clear statistical output  

---

##Input Data Format

The input CSV file must contain **two columns**:

| Column | Description |
|------|------------|
| Group | Experimental group or condition |
| Value | Measured numeric value |

Example:
```csv
Group,Value
Control,5.2
Control,5.4
Treatment,8.1
Treatment,8.3
