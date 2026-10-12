
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

st.set_page_config(page_title="StatsWithAyush Calculator", page_icon="📊")

st.title("📊 StatsWithAyush")
st.header("Chi-Square Test Calculator")
st.write("Enter observed frequencies. Use commas between columns and a new line for each row.")

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

        if (
            table.ndim != 2
            or len(rows) < 2
            or len(set(map(len, rows))) != 1
            or len(rows[0]) < 2
        ):
            st.error("Enter a rectangular table with at least 2 rows and 2 columns.")
        elif (
            not np.isfinite(table).all()
            or (table < 0).any()
            or not np.equal(table, np.floor(table)).all()
        ):
            st.error("Enter non-negative whole-number frequencies.")
        elif (table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any():
            st.error("Every row and column must have a positive total.")
        else:
            chi2, p, dof, expected = chi2_contingency(table)

            st.subheader("Test Results")
            st.metric("Chi-Square Statistic", f"{chi2:.4f}")
            st.metric("Degrees of Freedom", dof)
            st.metric("P-value", f"{p:.6f}")

            if p < alpha:
                st.success("Reject the null hypothesis.")
            else:
                st.info("Do not reject the null hypothesis.")

            st.subheader("Observed vs Expected Frequencies")
            st.caption(
                "Bars compare observed and expected counts for each cell "
                "of your contingency table."
            )

            labels = [
                f"R{r+1}, C{c+1}"
                for r in range(table.shape[0])
                for c in range(table.shape[1])
            ]
            observed_values = table.flatten()
            expected_values = expected.flatten()
            x = np.arange(len(labels))
            width = 0.38

            fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 4.5))
            ax.bar(x - width / 2, observed_values, width, label="Observed")
            ax.bar(x + width / 2, expected_values, width, label="Expected")
            ax.set_xlabel("Table cells")
            ax.set_ylabel("Frequency")
            ax.set_title("Chi-Square: Observed vs Expected")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Expected Frequency Table")
            st.dataframe(
                pd.DataFrame(
                    np.round(expected, 3),
                    index=[f"Row {i+1}" for i in range(table.shape[0])],
                    columns=[f"Column {i+1}" for i in range(table.shape[1])]
                )
            )

            st.caption(
                "The chi-square approximation may be unreliable when expected "
                "frequencies are too small. Interpret results in context."
            )

    except Exception:
        st.error("Please check your data and enter a valid numeric table.")
