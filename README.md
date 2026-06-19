# optical-bandgap-tauc

`optical-bandgap-tauc` is a research-grade Python command-line interface (CLI) tool designed for materials science laboratories. It automates the processing of raw Diffuse Reflectance Spectroscopy (DRS) data to compute Kubelka-Munk conversions, generate Tauc plots, and automatically detect the linear absorption-edge regime to derive optical band gaps ($E_g$) with high reproducibility.

---

## 1. Scientific Background & Math

The tool performs calculations in the following sequence:

1. **Kubelka-Munk Conversion**
   For reflectance $R$ (expressed as a fraction $0 \le R \le 1$):
   $$F(R) = \frac{(1 - R)^2}{2R}$$
   *If the input is already in absorbance units, the conversion is skipped.*

2. **Photon Energy Axis**
   Converts wavelength $\lambda$ (nm) to photon energy $E$ (eV) and sorts data in ascending order of energy:
   $$E = \frac{1239.84198}{\lambda}$$

3. **Tauc Transform**
   Calculates the Tauc variables $y_n(E) = (F(R) \cdot E)^n$ for the four standard transition types:
   * **Direct Allowed:** $n = 2$
   * **Indirect Allowed:** $n = 1/2$
   * **Direct Forbidden:** $n = 2/3$
   * **Indirect Forbidden:** $n = 1/3$

4. **Smoothing**
   Applies a Savitzky-Golay filter (`scipy.signal.savgol_filter`) to $y_n(E)$ with an adaptive window length based on data density ($W \approx 5\%$ of points, odd, polyorder=3).

5. **Linear Regression & Extrapolation**
   Fits a linear line $y = mE + c$ within the automatically detected linear window using least-squares. The optical band gap is the x-intercept:
   $$E_g = -\frac{c}{m}$$
   The confidence metric is reported via $R^2$.

---

## 2. Automatic Linear-Regime Detection Algorithm

To eliminate manual cursor-dragging in graphing software, this tool uses a robust derivative-plateau search algorithm:

1. **Derivative Smoothing:** Computes the first derivative $dy/dE$ (accounting for non-uniform $E$ spacing) and smoothes it with a Savitzky-Golay filter.
2. **Steepest Slope Identification:** Locates the peak of the derivative $E_{\text{peak}}$, representing the steepest point of the absorption edge.
3. **Plateau Search:** Expands a window outward from $E_{\text{peak}}$ in both directions as long as the derivative remains $\ge 90\%$ of the maximum derivative at $E_{\text{peak}}$.
4. **Inflection Trimming:** Computes the second derivative $d^2y/dE^2$. Within the candidate window, it identifies zero-crossings (inflections) where the second derivative exceeds $10\%$ of its maximum absolute value. The window is trimmed to stop *before* these crossings, preventing the fit from straddling baseline or saturation plateau regions.
5. **Fallback:** If the resulting window has $<5$ points, it falls back to a fixed $0.15$ eV window centered around $E_{\text{peak}}$ and flags the method as `fallback`.
6. **Anchored Curvature Cross-Check:** Runs a second independent method which finds the maximum of $d^2y/dE^2$ restricted to the edge region (where $dy/dE \ge 15\%$ of its peak) and fits a $0.15$ eV window starting from this point. If the two methods disagree by $>0.05$ eV, a warning is logged.

---

## 3. Installation

1. Clone or download this repository.
2. Navigate to the project root and install:
   ```bash
   pip install -e .
   ```
3. To install with development dependencies (e.g. for running tests):
   ```bash
   pip install -e ".[dev]"
   ```

---

## 4. Usage & Worked Example

### Single File Analysis
Analyze a reflectance spectrum file and generate plots and results tables:
```bash
bandgap-tauc analyze sample.csv --input-type reflectance-pct --transition all -o ./results
```

**Example Input File (`sample.csv`):**
```csv
Wavelength (nm),Reflectance (%R)
800.0,92.1
750.0,91.8
...
450.0,15.2
400.0,12.0
```

**Console Table Output:**
```text
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│                               Optical Band Gap Results Summary                                │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
  Sample     Transition Type         Eg (eV)    R²     Fit Window (eV)         Method         
 ───────────────────────────────────────────────────────────────────────────────────────────────
  sample_A   direct-allowed*           2.482  0.9982    2.420 - 2.540    derivative-plateau 
  sample_A   indirect-allowed          2.215  0.9540    2.150 - 2.290    derivative-plateau 
  sample_A   direct-forbidden          2.411  0.9912    2.350 - 2.470    derivative-plateau 
  sample_A   indirect-forbidden        2.302  0.9634    2.240 - 2.370    derivative-plateau 
 ───────────────────────────────────────────────────────────────────────────────────────────────
 * denotes recommended transition type
```

### Directory Batch Processing
Process all spectra in a folder, generate overlay plots, and compile comparison tables:
```bash
bandgap-tauc batch ./data/ -o ./batch_results --transition direct-allowed
```

---

## 5. Output Artifacts

* **Tauc plots:** Saved as PNG, PDF, or SVG in the output directory showing raw/smoothed curves, highlighted fit window, extrapolation line, and annotated $E_g$.
* **Results table:** Exported to JSON and CSV formats for single samples.
* **Batch summary (`batch_summary_comparison.csv`):** Comprises $E_g$ and $R^2$ values for all transitions across all samples, ranked by the recommended transition's $R^2$ value.
* **Batch comparison plots (`batch_overlay_*.png`):** Overlay curves of all samples for visual inspection.

---

## 6. How to Cite (Methods Section Blurb)

You can copy and adapt the following text for your methods section:

> "Optical band gaps ($E_g$) were extracted via automated Tauc plot analysis using the open-source utility `optical-bandgap-tauc` (v0.1.0). The software converts diffuse reflectance to the Kubelka-Munk function $F(R)$, converts wavelength to photon energy $E$ (eV), and computes the Tauc variable $(F(R) \cdot E)^n$. The linear absorption-edge regime was automatically located using a derivative-plateau search algorithm with a derivative threshold of 90%, boundary-checked against zero-crossings in the second derivative ($d^2y/dE^2$) smoothed with a Savitzky-Golay filter. A least-squares linear fit was applied on the detected window to extrapolate the band gap energy intercept ($Eg = -c/m$)."
