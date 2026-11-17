#!/usr/bin/env python3
"""
PH1050 Problem 2 - Vapour Pressure: Data and Statistics

Reads data/vapour_pressure_60_values.csv (relative to the repository root).
Install: python -m pip install numpy pandas matplotlib reportlab
Run:     python src/vapour_pressure_data_statistics.py
Outputs: PDF report in results/reports and three PNG plots in results/figures.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image, PageBreak

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CSV = ROOT / "data" / "vapour_pressure_60_values.csv"
PDF = ROOT / "results" / "reports" / "vapour_pressure_data_statistics_report.pdf"
FIGS = ROOT / "results" / "figures"
R = 8.314462618  # J/(mol K)

# Read by column position because the original CSV has a legacy-encoded
# degree symbol in one header. The values themselves are numeric.
source = pd.read_csv(CSV, encoding="latin1")
data = source.iloc[:, :7].copy()
data.columns = [
    "observation", "temperature_c", "temperature_k", "reference_kpa",
    "simulated_kpa", "uncertainty_kpa", "split"
]
for column in data.columns[:6]:
    data[column] = pd.to_numeric(data[column], errors="coerce")
data = data.dropna(subset=[
    "temperature_c", "temperature_k", "reference_kpa",
    "simulated_kpa", "uncertainty_kpa"
]).copy()
data["split"] = data["split"].astype(str).str.strip().str.lower()
training = data[data["split"] == "train"].copy()
testing = data[data["split"] == "test"].copy()

if len(data) != 60 or len(training) != 36 or len(testing) != 24:
    raise ValueError("Expected 60 rows: 36 train and 24 test. Check the CSV file.")
if not np.allclose(data["temperature_k"], data["temperature_c"] + 273.15, atol=1e-6):
    raise ValueError("Temperature conversion check failed.")

# Weighted Clausius-Clapeyron fit, using training rows only:
# ln(P) = ln(P0) - (L/R) * (1/T - 1/T0)
T0 = float(data["temperature_k"].iloc[0])
x = 1.0 / training["temperature_k"].to_numpy() - 1.0 / T0
y = np.log(training["simulated_kpa"].to_numpy())
sigma_log_p = training["uncertainty_kpa"].to_numpy() / training["simulated_kpa"].to_numpy()
weights = 1.0 / sigma_log_p**2
design = np.column_stack([np.ones(len(x)), x])
covariance = np.linalg.inv(design.T @ (weights[:, None] * design))
parameters = covariance @ (design.T @ (weights * y))
ln_p0, slope = parameters
p0 = float(np.exp(ln_p0))
p0_se = float(p0 * np.sqrt(covariance[0, 0]))
latent_heat = float(-slope * R)
latent_heat_se = float(R * np.sqrt(covariance[1, 1]))
fit_residuals = y - design @ parameters
chi_square = float(np.sum((fit_residuals / sigma_log_p)**2))
degrees_freedom = len(training) - 2

# Evaluate the fitted model on both subsets; test data were not used in fitting.
temperature_k = data["temperature_k"].to_numpy()
data["fitted_kpa"] = np.exp(ln_p0 + slope * (1.0 / temperature_k - 1.0 / T0))

def metrics(part):
    observed = part["simulated_kpa"].to_numpy()
    predicted = part["fitted_kpa"].to_numpy()
    rmse = np.sqrt(np.mean((predicted - observed)**2))
    rms_relative = 100 * np.sqrt(np.mean(((predicted - observed) / observed)**2))
    mean_abs_relative = 100 * np.mean(np.abs(predicted / observed - 1.0))
    return rmse, rms_relative, mean_abs_relative

training_metrics = metrics(data.loc[training.index])
testing_metrics = metrics(data.loc[testing.index])
all_metrics = metrics(data)
relative_noise = data["simulated_kpa"].to_numpy() / data["reference_kpa"].to_numpy() - 1.0
noise_signed = 100 * np.mean(relative_noise)
noise_mae = 100 * np.mean(np.abs(relative_noise))
noise_rms = 100 * np.sqrt(np.mean(relative_noise**2))

# Create clean plots using ASCII-only labels to avoid missing-glyph squares.
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25})
temperature_grid = np.linspace(data["temperature_k"].min(), data["temperature_k"].max(), 400)
pressure_grid = np.exp(ln_p0 + slope * (1.0 / temperature_grid - 1.0 / T0))

fig, ax = plt.subplots(figsize=(7.1, 3.65))
ax.errorbar(data["temperature_c"], data["simulated_kpa"],
            yerr=data["uncertainty_kpa"], fmt="o", ms=3, capsize=2,
            alpha=0.72, label="Simulated data (+/- uncertainty)")
ax.plot(data["temperature_c"], data["reference_kpa"], ".", ms=3, label="Reference data")
ax.plot(temperature_grid - 273.15, pressure_grid, lw=2,
        label="Weighted fit (training data)")
ax.set(xlabel="Temperature (deg C)", ylabel="Vapour pressure (kPa)",
       title="Vapour pressure and fitted model")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(FIGS / "vapour_fit.png", bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.1, 3.65))
ax.errorbar(1.0 / training["temperature_k"], np.log(training["simulated_kpa"]),
            yerr=training["uncertainty_kpa"] / training["simulated_kpa"],
            fmt="o", ms=4, label="Training data")
ax.scatter(1.0 / testing["temperature_k"], np.log(testing["simulated_kpa"]),
           marker="s", s=20, label="Test data")
inverse_temperature = np.linspace(1.0 / data["temperature_k"].max(),
                                  1.0 / data["temperature_k"].min(), 300)
ax.plot(inverse_temperature,
        ln_p0 + slope * (inverse_temperature - 1.0 / T0),
        lw=2, label="Weighted linear fit")
ax.set(xlabel="1 / T (1/K)", ylabel="ln(P in kPa)",
       title="Linearised Clausius-Clapeyron fit")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(FIGS / "vapour_linearised_fit.png", bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.1, 3.65))
ax.axhline(0, color="black", lw=1)
ax.scatter(data["temperature_c"],
           data["simulated_kpa"] - data["fitted_kpa"], s=22)
ax.set(xlabel="Temperature (deg C)",
       ylabel="Observed minus fitted pressure (kPa)",
       title="Residual analysis")
fig.tight_layout()
fig.savefig(FIGS / "vapour_residuals.png", bbox_inches="tight")
plt.close(fig)

# Build a compact PDF with standard Helvetica fonts and ASCII-safe report text.
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="MainTitle", parent=styles["Title"],
    fontName="Helvetica-Bold", fontSize=16, leading=19, alignment=TA_CENTER,
    textColor=colors.HexColor("#742638"), spaceAfter=3))
styles.add(ParagraphStyle(name="Sub", parent=styles["Normal"], fontName="Helvetica",
    fontSize=8.5, leading=11, alignment=TA_CENTER, textColor=colors.HexColor("#444444"),
    spaceAfter=8))
styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"],
    fontName="Helvetica-Bold", fontSize=10.5, leading=13,
    textColor=colors.HexColor("#742638"), spaceBefore=6, spaceAfter=3))
styles.add(ParagraphStyle(name="Body9", parent=styles["BodyText"],
    fontName="Helvetica", fontSize=8.2, leading=11, spaceAfter=3))
styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"],
    fontName="Helvetica", fontSize=7.4, leading=9, spaceAfter=2))

def make_table(rows, widths, font_size=7.2):
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#742638")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c7c7c7")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f5f2f3")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table

story = [
    Paragraph("PH1050 - FOUNDATION OF COMPUTATIONAL PHYSICS", styles["MainTitle"]),
    Paragraph("FINAL GROUP PROJECT | PROBLEM 2: FIND THE LAW<br/>"
              "PART 2 - DATA AND STATISTICS: VAPOUR PRESSURE", styles["Sub"]),
    Paragraph("1. Dataset overview", styles["Section"]),
    Paragraph(
        f"The dataset contains {len(data)} measurements from "
        f"{data.temperature_c.min():.1f} to {data.temperature_c.max():.1f} deg C. "
        f"It includes reference pressure, simulated pressure, estimated uncertainty, "
        f"and a pre-assigned split: {len(training)} training points and "
        f"{len(testing)} test points. The temperature conversion was checked using "
        "T(K) = T(deg C) + 273.15. Parameters were fitted using training data only.",
        styles["Body9"])
]
summary_rows = [
    ["Quantity (all 60 points)", "Mean", "Sample SD", "Min", "Max"],
    ["Temperature (deg C)", f"{data.temperature_c.mean():.3f}",
     f"{data.temperature_c.std():.3f}", f"{data.temperature_c.min():.3f}",
     f"{data.temperature_c.max():.3f}"],
    ["Temperature (K)", f"{data.temperature_k.mean():.3f}",
     f"{data.temperature_k.std():.3f}", f"{data.temperature_k.min():.3f}",
     f"{data.temperature_k.max():.3f}"],
    ["Reference pressure (kPa)", f"{data.reference_kpa.mean():.3f}",
     f"{data.reference_kpa.std():.3f}", f"{data.reference_kpa.min():.3f}",
     f"{data.reference_kpa.max():.3f}"],
    ["Simulated pressure (kPa)", f"{data.simulated_kpa.mean():.3f}",
     f"{data.simulated_kpa.std():.3f}", f"{data.simulated_kpa.min():.3f}",
     f"{data.simulated_kpa.max():.3f}"],
    ["Uncertainty (kPa)", f"{data.uncertainty_kpa.mean():.3f}",
     f"{data.uncertainty_kpa.std():.3f}", f"{data.uncertainty_kpa.min():.3f}",
     f"{data.uncertainty_kpa.max():.3f}"],
]
story += [
    make_table(summary_rows, [47*mm, 25*mm, 25*mm, 25*mm, 25*mm]),
    Paragraph("2. Physical model and fitting method", styles["Section"]),
    Paragraph(
        "For constant molar latent heat L, the Clausius-Clapeyron equation is "
        "dP/dT = L P / (R T^2). Its integrated form is "
        "P(T) = P0 exp[-(L/R)(1/T - 1/T0)]. Here T is in kelvin and "
        "R = 8.314462618 J/(mol K). Taking logarithms gives "
        "ln(P) = ln(P0) - (L/R)(1/T - 1/T0). A weighted straight-line fit was "
        "applied to ln(P_sim) versus (1/T - 1/T0), using training data only. "
        "The propagated log-pressure uncertainty was approximated by "
        "sigma_lnP = sigma_P/P, so weights were 1/sigma_lnP^2.",
        styles["Body9"]),
    Paragraph("3. Fitted parameters", styles["Section"]),
    make_table([
        ["Parameter", "Estimate", "Standard uncertainty"],
        ["Reference temperature T0", f"{T0:.2f} K", "-"],
        ["Reference pressure P0", f"{p0:.4f} kPa", f"{p0_se:.4f} kPa"],
        ["Latent heat L", f"{latent_heat/1000:.3f} kJ/mol",
         f"{latent_heat_se/1000:.3f} kJ/mol"],
        ["Approx. 95% interval for L",
         f"{(latent_heat-1.96*latent_heat_se)/1000:.3f} to "
         f"{(latent_heat+1.96*latent_heat_se)/1000:.3f} kJ/mol",
         "Normal approximation"],
        ["Reduced chi-square", f"{chi_square/degrees_freedom:.3f}",
         f"{degrees_freedom} degrees of freedom"],
    ], [48*mm, 52*mm, 52*mm]),
    Paragraph("4. Noise and test performance", styles["Section"]),
    Paragraph(
        f"Relative deviation of simulated pressure from reference pressure: "
        f"mean signed = {noise_signed:.2f}%, mean absolute = {noise_mae:.2f}%, "
        f"and RMS = {noise_rms:.2f}%.", styles["Body9"]),
    make_table([
        ["Subset", "N", "RMSE (kPa)", "RMS relative error", "Mean abs. relative error"],
        ["Training", str(len(training)), f"{training_metrics[0]:.3f}",
         f"{training_metrics[1]:.3f}%", f"{training_metrics[2]:.3f}%"],
        ["Test", str(len(testing)), f"{testing_metrics[0]:.3f}",
         f"{testing_metrics[1]:.3f}%", f"{testing_metrics[2]:.3f}%"],
        ["All data", str(len(data)), f"{all_metrics[0]:.3f}",
         f"{all_metrics[1]:.3f}%", f"{all_metrics[2]:.3f}%"],
    ], [29*mm, 13*mm, 30*mm, 37*mm, 43*mm]),
    Paragraph(
        "The held-out test error is close to the training error, indicating similar "
        "relative predictive performance on both subsets. The reduced chi-square is "
        "greater than one, suggesting that the scatter is larger than the stated "
        "log-pressure uncertainties alone would predict.", styles["Body9"]),
    PageBreak(),
    Paragraph("5. Plots and interpretation", styles["Section"]),
    Image(str(FIGS / "vapour_fit.png"), width=164*mm, height=84*mm),
    Paragraph("Figure 1. Simulated measurements with stated uncertainties, "
              "reference values, and fitted model.", styles["Small"]),
    Image(str(FIGS / "vapour_linearised_fit.png"), width=164*mm, height=84*mm),
    Paragraph("Figure 2. Linearised Clausius-Clapeyron plot. Training data determine "
              "the fitted line; test data are shown separately.", styles["Small"]),
    Image(str(FIGS / "vapour_residuals.png"), width=164*mm, height=84*mm),
    Paragraph("Figure 3. Residuals of simulated pressure relative to the fitted model.",
              styles["Small"]),
    Paragraph("6. Conclusion", styles["Section"]),
    Paragraph(
        f"The weighted fit to the training data gives L = {latent_heat/1000:.3f} "
        f"+/- {latent_heat_se/1000:.3f} kJ/mol (one standard uncertainty) and "
        f"P0 = {p0:.4f} +/- {p0_se:.4f} kPa at T0 = {T0:.2f} K. The held-out test "
        f"RMS relative error is {testing_metrics[1]:.2f}%. The constant-L "
        "Clausius-Clapeyron model provides a useful description over the measured "
        "temperature interval. Deviations remain because the model assumes constant "
        "latent heat and the measurements contain noise.", styles["Body9"]),
    Paragraph("<b>Reproducibility:</b> Run this script from the repository; the data are read from data/. "
              "Running the script regenerates this PDF and all three plots.",
              styles["Small"]),
]

document = SimpleDocTemplate(str(PDF), pagesize=A4, rightMargin=15*mm,
    leftMargin=15*mm, topMargin=12*mm, bottomMargin=12*mm,
    title="PH1050 Vapour Pressure Data and Statistics")
document.build(story)

print(f"Data points: {len(data)} (training {len(training)}, test {len(testing)})")
print(f"Latent heat: {latent_heat/1000:.3f} +/- {latent_heat_se/1000:.3f} kJ/mol")
print(f"Reference pressure P0: {p0:.4f} +/- {p0_se:.4f} kPa at T0={T0:.2f} K")
print(f"Test RMS relative error: {testing_metrics[1]:.3f}%")
print(f"Created: {PDF.name}")
