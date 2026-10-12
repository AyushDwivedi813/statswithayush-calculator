
import streamlit as st
import numpy as np
from scipy.stats import chi2_contingency

st.set_page_config(page_title="StatsWithAyush Calculator", page_icon="📊")

st.title("📊 StatsWithAyush")
st.header("Chi-Square Test Calculator")
st.write("Enter observed frequencies separated by commas. Use a new line for each row.")

data = st.text_area(
    "Observed frequency table",
    value="20,30\n30,20",
    height=120
)

alpha = st.selectbox("Significance level", [0.01, 0.05, 0.10], index=1)

if st.button("Calculate Chi-Square"):
    try:
        rows = [
            [float(x.strip()) for x in line.split(",")]
            for line in data.strip().splitlines()
            if line.strip()
        ]
        table = np.array(rows, dtype=float)

        if table.ndim != 2 or len(rows) < 2 or len(set(map(len, rows))) != 1 or len(rows[0]) < 2:
            st.error("Enter a rectangular table with at least 2 rows and 2 columns.")
        elif not np.isfinite(table).all() or (table < 0).any() or not np.equal(table, np.floor(table)).all():
            st.error("Enter non-negative whole-number frequencies.")
        elif (table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any():
            st.error("Every row and column must have a positive total.")
        else:
            chi2, p, dof, expected = chi2_contingency(table)

            st.subheader("Results")
            st.metric("Chi-Square Statistic", f"{chi2:.4f}")
            st.metric("Degrees of Freedom", dof)
            st.metric("P-value", f"{p:.6f}")
            st.write(f"**Significance level (α):** {alpha}")

            if p < alpha:
                st.success("Reject the null hypothesis.")
            else:
                st.info("Do not reject the null hypothesis.")

            st.subheader("Expected Frequencies")
            st.dataframe(np.round(expected, 3))

            st.caption("Interpret results in the context of your study. Expected frequencies should be sufficiently large for the chi-square approximation to be reliable.")

    except Exception:
        st.error("Please check your table. Enter numbers separated by commas, with one row per line.")

