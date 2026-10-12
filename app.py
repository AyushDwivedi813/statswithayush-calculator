
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(
    page_title="StatsWithAyush | Statistics & SQC",
    page_icon="📊",
    layout="wide"
)

st.title("📊 StatsWithAyush")
st.caption("Interactive statistical tests, graphs and Statistical Quality Control tools.")

ALPHA_OPTIONS = [0.01, 0.05, 0.10]


def decision(p, alpha):
    if p < alpha:
        st.success("Reject the null hypothesis at the selected significance level.")
    else:
        st.info("Do not reject the null hypothesis at the selected significance level.")


def numbers(text):
    values = np.array(
        [float(x.strip()) for x in text.replace("\n", ",").split(",") if x.strip()],
        dtype=float
    )
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Enter valid finite numbers.")
    return values


def plot_lines(values, center, lcl, ucl, title, ylabel="Value"):
    values = np.asarray(values, dtype=float)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(1, len(values) + 1), values, marker="o", label="Sample statistic")
    ax.axhline(center, linestyle="-", label=f"Center line = {center:.4g}")
    ax.axhline(ucl, linestyle="--", label=f"UCL = {ucl:.4g}")
    ax.axhline(lcl, linestyle="--", label=f"LCL = {lcl:.4g}")
    ax.set_title(title)
    ax.set_xlabel("Sample number")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    flagged = np.where((values > ucl) | (values < lcl))[0] + 1
    if len(flagged):
        st.warning("Points outside control limits: " + ", ".join(map(str, flagged)))
    else:
        st.success("No points are outside the calculated control limits. Other non-random patterns may still exist.")


def subgroup_data(text):
    rows = [
        [float(v.strip()) for v in line.split(",") if v.strip()]
        for line in text.strip().splitlines() if line.strip()
    ]
    if len(rows) < 2 or len(set(map(len, rows))) != 1:
        raise ValueError("Enter at least two rows with the same number of values.")
    data = np.array(rows, dtype=float)
    if not np.isfinite(data).all() or data.shape[1] < 2:
        raise ValueError("Enter finite numeric values with at least two observations per subgroup.")
    return data


def capability(values, lsl, usl, within_sigma=None):
    if not lsl < usl:
        st.error("LSL must be less than USL.")
        return
    overall = np.std(values, ddof=1)
    if overall <= 0:
        st.error("Capability cannot be calculated when all values are identical.")
        return
    cp = cpk = None
    if within_sigma is not None and within_sigma > 0:
        cp = (usl - lsl) / (6 * within_sigma)
        cpk = min(usl - np.mean(values), np.mean(values) - lsl) / (3 * within_sigma)
    pp = (usl - lsl) / (6 * overall)
    ppk = min(usl - np.mean(values), np.mean(values) - lsl) / (3 * overall)
    if cp is not None:
        c1, c2 = st.columns(2)
        c1.metric("Cp", f"{cp:.3f}")
        c2.metric("Cpk", f"{cpk:.3f}")
    c1, c2 = st.columns(2)
    c1.metric("Pp", f"{pp:.3f}")
    c2.metric("Ppk", f"{ppk:.3f}")
    st.caption("Cp/Cpk use the supplied within-process sigma. Pp/Ppk use overall sample standard deviation. Interpret only when process and data assumptions are appropriate.")


test = st.sidebar.selectbox(
    "Choose a tool",
    [
        "Chi-Square Test",
        "One-Sample T-Test",
        "Two-Sample T-Test",
        "One-Sample Z-Test",
        "X-bar and R Charts",
        "X-bar and S Charts",
        "Individuals and Moving Range",
        "p Chart",
        "np Chart",
        "c Chart",
        "u Chart",
        "Process Capability",
        "Pareto Chart",
        "Histogram"
    ]
)

alpha = st.sidebar.selectbox("Significance level", ALPHA_OPTIONS, index=1)


# ---------------- HYPOTHESIS TESTS ----------------

if test == "Chi-Square Test":
    st.header("Chi-Square Test of Independence")
    st.write("Enter observed counts: comma-separated columns, one row per line.")
    raw = st.text_area("Observed frequency table", "20,30\n30,20")

    if st.button("Calculate Chi-Square"):
        try:
            table = subgroup_data(raw)
            if (table < 0).any() or not np.equal(table, np.floor(table)).all():
                raise ValueError("Counts must be non-negative whole numbers.")
            if (table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any():
                raise ValueError("Each row and column must have a positive total.")
            chi2, p, dof, expected = stats.chi2_contingency(table)
            st.metric("Chi-Square statistic", f"{chi2:.4f}")
            st.metric("Degrees of freedom", dof)
            st.metric("P-value", f"{p:.6f}")
            decision(p, alpha)
            labels = [f"R{i+1},C{j+1}" for i in range(table.shape[0]) for j in range(table.shape[1])]
            x = np.arange(len(labels))
            width = 0.38
            fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.65), 4))
            ax.bar(x-width/2, table.flatten(), width, label="Observed")
            ax.bar(x+width/2, expected.flatten(), width, label="Expected")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_ylabel("Frequency")
            ax.set_title("Observed vs Expected Frequencies")
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.subheader("Expected frequencies")
            st.dataframe(pd.DataFrame(np.round(expected, 3)))
            st.caption("Small expected frequencies can make the chi-square approximation unreliable.")
        except Exception as e:
            st.error(str(e))


elif test == "One-Sample T-Test":
    st.header("One-Sample T-Test")
    raw = st.text_area("Sample values, comma-separated", "12,14,15,13,16,14")
    mu = st.number_input("Hypothesized population mean", value=10.0)
    alt_label = st.selectbox("Alternative hypothesis", ["Two-sided", "Greater than", "Less than"])
    alt = {"Two-sided": "two-sided", "Greater than": "greater", "Less than": "less"}[alt_label]

    if st.button("Calculate T-Test"):
        try:
            x = numbers(raw)
            if len(x) < 2 or np.std(x, ddof=1) == 0:
                raise ValueError("Enter at least two values with non-zero sample variation.")
            result = stats.ttest_1samp(x, mu, alternative=alt)
            st.metric("T statistic", f"{result.statistic:.4f}")
            st.metric("Degrees of freedom", len(x)-1)
            st.metric("P-value", f"{result.pvalue:.6f}")
            decision(result.pvalue, alpha)
            fig, ax = plt.subplots()
            ax.hist(x, bins="auto", edgecolor="black")
            ax.axvline(np.mean(x), linestyle="--", label=f"Sample mean {np.mean(x):.2f}")
            ax.axvline(mu, linestyle=":", label=f"Hypothesized mean {mu:.2f}")
            ax.legend()
            ax.set_title("Sample distribution")
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(str(e))


elif test == "Two-Sample T-Test":
    st.header("Independent Two-Sample T-Test (Welch's)")
    a_raw = st.text_area("Group 1 values", "12,14,15,13,16")
    b_raw = st.text_area("Group 2 values", "10,11,12,9,13")
    alt_label = st.selectbox("Alternative hypothesis", ["Two-sided", "Group 1 greater", "Group 1 less"])
    alt = {"Two-sided": "two-sided", "Group 1 greater": "greater", "Group 1 less": "less"}[alt_label]

    if st.button("Calculate Two-Sample T-Test"):
        try:
            a, b = numbers(a_raw), numbers(b_raw)
            if len(a) < 2 or len(b) < 2:
                raise ValueError("Each group needs at least two observations.")
            result = stats.ttest_ind(a, b, equal_var=False, alternative=alt)
            st.metric("T statistic", f"{result.statistic:.4f}")
            st.metric("P-value", f"{result.pvalue:.6f}")
            decision(result.pvalue, alpha)
            fig, ax = plt.subplots()
            ax.boxplot([a, b], labels=["Group 1", "Group 2"])
            ax.set_title("Comparison of independent groups")
            ax.set_ylabel("Value")
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(str(e))


elif test == "One-Sample Z-Test":
    st.header("One-Sample Z-Test")
    st.write("Use when population standard deviation is known and the sampling assumptions are met.")
    mean = st.number_input("Sample mean", value=52.0)
    mu = st.number_input("Hypothesized population mean", value=50.0)
    sigma = st.number_input("Known population standard deviation", min_value=0.000001, value=5.0)
    n = st.number_input("Sample size", min_value=1, value=30, step=1)
    alt_label = st.selectbox("Alternative hypothesis", ["Two-sided", "Greater than", "Less than"])

    if st.button("Calculate Z-Test"):
        z = (mean-mu)/(sigma/np.sqrt(n))
        p = (2*stats.norm.sf(abs(z)) if alt_label == "Two-sided"
             else stats.norm.sf(z) if alt_label == "Greater than"
             else stats.norm.cdf(z))
        st.metric("Z statistic", f"{z:.4f}")
        st.metric("P-value", f"{p:.6f}")
        decision(p, alpha)
        grid = np.linspace(-4, 4, 500)
        fig, ax = plt.subplots()
        ax.plot(grid, stats.norm.pdf(grid))
        ax.axvline(z, linestyle="--", label=f"Observed Z = {z:.2f}")
        if alt_label == "Two-sided":
            mask = (grid <= -abs(z)) | (grid >= abs(z))
        elif alt_label == "Greater than":
            mask = grid >= z
        else:
            mask = grid <= z
        ax.fill_between(grid, stats.norm.pdf(grid), where=mask, alpha=0.3, label="Tail area")
        ax.legend()
        ax.set_title("Standard normal distribution")
        st.pyplot(fig)
        plt.close(fig)


# ---------------- SQC CONTROL CHARTS ----------------

elif test == "X-bar and R Charts":
    st.header("X-bar and R Control Charts")
    st.write("Enter each subgroup on a separate line. Use equal subgroup sizes, typically 2–10.")
    raw = st.text_area("Subgroup measurements", "10,11,9,10\n12,11,10,11\n9,10,11,10\n11,12,10,11")
    if st.button("Create X-bar and R Charts"):
        try:
            d = subgroup_data(raw)
            n = d.shape[1]
            constants = {
                2:(1.880,0,3.267), 3:(1.023,0,2.574), 4:(0.729,0,2.282),
                5:(0.577,0,2.114), 6:(0.483,0,2.004), 7:(0.419,0.076,1.924),
                8:(0.373,0.136,1.864), 9:(0.337,0.184,1.816), 10:(0.308,0.223,1.777)
            }
            if n not in constants:
                raise ValueError("This X-bar/R calculator supports subgroup sizes 2–10.")
            A2, D3, D4 = constants[n]
            means, ranges = d.mean(axis=1), np.ptp(d, axis=1)
            xbarbar, rbar = means.mean(), ranges.mean()
            plot_lines(means, xbarbar, xbarbar-A2*rbar, xbarbar+A2*rbar, "X-bar Chart")
            plot_lines(ranges, rbar, D3*rbar, D4*rbar, "R Chart", "Range")
        except Exception as e:
            st.error(str(e))


elif test == "X-bar and S Charts":
    st.header("X-bar and S Control Charts")
    raw = st.text_area("Subgroup measurements", "10,11,9,10\n12,11,10,11\n9,10,11,10\n11,12,10,11")
    if st.button("Create X-bar and S Charts"):
        try:
            d = subgroup_data(raw)
            n = d.shape[1]
            # B3/B4/c4 constants for subgroup sizes 2-10
            constants = {
                2:(0,3.267,0.7979), 3:(0,2.568,0.8862), 4:(0,2.266,0.9213),
                5:(0,2.089,0.9400), 6:(0.030,1.970,0.9515), 7:(0.118,1.882,0.9594),
                8:(0.185,1.815,0.9650), 9:(0.239,1.761,0.9693), 10:(0.284,1.716,0.9727)
            }
            if n not in constants:
                raise ValueError("This X-bar/S calculator supports subgroup sizes 2–10.")
            B3, B4, c4 = constants[n]
            means, s = d.mean(axis=1), d.std(axis=1, ddof=1)
            xb, sb = means.mean(), s.mean()
            plot_lines(means, xb, xb-3*sb/(c4*np.sqrt(n)), xb+3*sb/(c4*np.sqrt(n)), "X-bar Chart")
            plot_lines(s, sb, B3*sb, B4*sb, "S Chart", "Standard deviation")
        except Exception as e:
            st.error(str(e))


elif test == "Individuals and Moving Range":
    st.header("Individuals and Moving Range (I-MR)")
    raw = st.text_area("Measurements in time order", "10,11,9,12,10,11,13,10,9,12")
    if st.button("Create I-MR Charts"):
        try:
            x = numbers(raw)
            if len(x) < 3:
                raise ValueError("Enter at least three observations.")
            mr = np.abs(np.diff(x))
            mrbar = mr.mean()
            center = x.mean()
            sigma = mrbar/1.128
            plot_lines(x, center, center-3*sigma, center+3*sigma, "Individuals Chart")
            plot_lines(mr, mrbar, 0, 3.267*mrbar, "Moving Range Chart", "Moving range")
        except Exception as e:
            st.error(str(e))


elif test == "p Chart":
    st.header("p Chart — Fraction Defective")
    st.write("Enter defectives and sample size for each sample.")
    defects = st.text_area("Defective counts", "4,5,3,6,4,5")
    sizes = st.text_area("Sample sizes", "100,100,100,100,100,100")
    if st.button("Create p Chart"):
        try:
            d, n = numbers(defects), numbers(sizes)
            if len(d) != len(n) or (n <= 0).any() or (d < 0).any() or (d > n).any():
                raise ValueError("Counts and sample sizes must match; each defective count must be between 0 and sample size.")
            pbar = d.sum()/n.sum()
            se = np.sqrt(pbar*(1-pbar)/n)
            plot_lines(d/n, pbar, np.maximum(0, pbar-3*se), np.minimum(1, pbar+3*se), "p Chart", "Fraction defective")
        except Exception as e:
            st.error(str(e))


elif test == "np Chart":
    st.header("np Chart — Number Defective")
    defects = st.text_area("Defective counts", "4,5,3,6,4,5")
    size = st.number_input("Constant sample size", min_value=1, value=100)
    if st.button("Create np Chart"):
        try:
            d = numbers(defects)
            if (d < 0).any() or (d > size).any() or not np.equal(d, np.floor(d)).all():
                raise ValueError("Defective counts must be whole numbers from 0 to sample size.")
            pbar = d.sum()/(len(d)*size)
            center = size*pbar
            sd = np.sqrt(size*pbar*(1-pbar))
            plot_lines(d, center, max(0, center-3*sd), min(size, center+3*sd), "np Chart", "Number defective")
        except Exception as e:
            st.error(str(e))


elif test == "c Chart":
    st.header("c Chart — Defects per Constant Inspection Unit")
    raw = st.text_area("Defect counts per unit", "4,5,3,6,4,5,2,7")
    if st.button("Create c Chart"):
        try:
            d = numbers(raw)
            if (d < 0).any() or not np.equal(d, np.floor(d)).all():
                raise ValueError("Defect counts must be non-negative whole numbers.")
            cbar = d.mean()
            plot_lines(d, cbar, max(0, cbar-3*np.sqrt(cbar)), cbar+3*np.sqrt(cbar), "c Chart", "Defects")
        except Exception as e:
            st.error(str(e))


elif test == "u Chart":
    st.header("u Chart — Defects per Unit")
    defects = st.text_area("Total defect counts", "4,5,3,6,4,5")
    units = st.text_area("Units inspected in each sample", "100,100,100,100,100,100")
    if st.button("Create u Chart"):
        try:
            d, n = numbers(defects), numbers(units)
            if len(d) != len(n) or (n <= 0).any() or (d < 0).any():
                raise ValueError("Enter matching non-negative defect counts and positive unit counts.")
            ubar = d.sum()/n.sum()
            se = np.sqrt(ubar/n)
            plot_lines(d/n, ubar, np.maximum(0, ubar-3*se), ubar+3*se, "u Chart", "Defects per unit")
        except Exception as e:
            st.error(str(e))


# ---------------- PROCESS CAPABILITY ----------------

elif test == "Process Capability":
    st.header("Process Capability: Cp, Cpk, Pp, Ppk")
    raw = st.text_area("Process measurements", "9.8,10.1,10.2,9.9,10.0,10.3,9.7,10.1,10.0,9.9")
    lsl = st.number_input("Lower specification limit (LSL)", value=9.0)
    usl = st.number_input("Upper specification limit (USL)", value=11.0)
    sigma_within = st.number_input("Within-process standard deviation (optional; enter 0 to omit)", min_value=0.0, value=0.0)
    if st.button("Calculate Capability"):
        try:
            x = numbers(raw)
            if len(x) < 2:
                raise ValueError("Enter at least two measurements.")
            capability(x, lsl, usl, sigma_within if sigma_within > 0 else None)
            fig, ax = plt.subplots()
            ax.hist(x, bins="auto", edgecolor="black")
            ax.axvline(lsl, linestyle="--", label="LSL")
            ax.axvline(usl, linestyle="--", label="USL")
            ax.axvline(np.mean(x), linestyle=":", label="Mean")
            ax.legend()
            ax.set_title("Process measurements and specification limits")
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(str(e))


# ---------------- PARETO AND HISTOGRAM ----------------

elif test == "Pareto Chart":
    st.header("Pareto Chart")
    st.write("Enter defect categories and their counts in matching order.")
    labels_raw = st.text_area("Categories, comma-separated", "Scratches,Wrong size,Missing part,Stains,Other")
    counts_raw = st.text_area("Counts, comma-separated", "35,25,18,12,10")
    if st.button("Create Pareto Chart"):
        try:
            labels = [x.strip() for x in labels_raw.split(",") if x.strip()]
            counts = numbers(counts_raw)
            if len(labels) != len(counts) or (counts < 0).any():
                raise ValueError("Provide one non-negative count for each category.")
            order = np.argsort(counts)[::-1]
            counts, labels = counts[order], [labels[i] for i in order]
            total = counts.sum()
            if total <= 0:
                raise ValueError("Total count must be greater than zero.")
            cumulative = np.cumsum(counts)/total*100
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(labels, counts, edgecolor="black")
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=35)
            ax2 = ax.twinx()
            ax2.plot(labels, cumulative, marker="o", linestyle="-")
            ax2.set_ylabel("Cumulative percentage")
            ax2.set_ylim(0, 110)
            ax.set_title("Pareto Chart")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.dataframe(pd.DataFrame({"Category": labels, "Count": counts, "Cumulative %": np.round(cumulative, 2)}))
        except Exception as e:
            st.error(str(e))


elif test == "Histogram":
    st.header("Histogram")
    raw = st.text_area("Numerical data, comma-separated", "12,14,15,13,16,14,12,17,15,13")
    if st.button("Create Histogram"):
        try:
            x = numbers(raw)
            if len(x) < 2:
                raise ValueError("Enter at least two values.")
            fig, ax = plt.subplots()
            ax.hist(x, bins="auto", edgecolor="black")
            ax.axvline(np.mean(x), linestyle="--", label=f"Mean = {np.mean(x):.3g}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            ax.set_title("Histogram")
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(str(e))


st.divider()
st.caption(
    "Educational tool by StatsWithAyush. Control limits estimate process behavior; "
    "they are not specification limits. Verify assumptions and constants before using "
    "results for operational quality decisions."
)
