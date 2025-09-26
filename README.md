# Waveform Plotter

A Python tool to **read, visualize, and export waveform data** from `.txt` or `.csv` files.
It supports **interactive selection** or **JSON configuration** for batch runs, with flexible plotting, legends, and export options.

---

## 🚀 Quick Start

```bash
git clone https://github.com/N1kor4/from_waveform_to_python.git
cd from_waveform_to_python
pip install numpy matplotlib
python main.py
```

---

## ✨ Features

* 📂 **Automatic folder setup**: `Waveform/`, `Results/`, `Config/`
* 📑 **Input formats**:

  * `.txt` waveform files (with `TIME` header)
  * `.csv` files (comma or semicolon delimiters, auto-detected)
* ⚙️ **Configurable workflow**:

  * Interactive prompts **or** pre-defined `.json` configuration files
  * Save/load configurations for reproducibility
* 📊 **Flexible plotting**:

  * Custom subplot grid (choose number of columns)
  * Independent subplot titles, y-labels, legends, and x-limits
  * Global or per-subplot legends (inside or outside placement)
  * Wide figures (default width = 16)
* 🧮 **Axis formatting**:

  * Engineering notation for time axis (`s, ms, µs, ns, ps, fs`)
  * Scientific notation limited to 10³ steps
* 🖼 **High-quality output**:

  * Vector graphics (`pdf`, `svg`) or raster (`png`)
  * 300 DPI, Helvetica/Arial/DejaVu Sans font, Tol 16 color palette
* 📤 **CSV export**:

  * Export all signals
  * Export only selected signals
* ✅ **Headless mode**: works on servers without a GUI

---

## 📂 Folder Structure

```bash
.
├── waveform_plotter.py
├── Waveform/        # Place input waveform files here (.txt / .csv)
├── Results/         # Plots and CSVs are saved here
└── Config/          # Optional .json configuration files
```

---

## 🚀 Usage

### Interactive mode

1. Place your waveform files (`.txt` or `.csv`) inside the `Waveform/` folder.

2. Run the script:

   ```bash
   python waveform_plotter.py
   ```

3. Follow the on-screen prompts:

   * Select file(s)
   * Choose signals and subplot grouping syntax:

     * `:` → ranges (`1:4` → 1,2,3,4)
     * `-` → same subplot (`1-3-5`)
     * `_` → separate subplots (`1:4_7`)
     * `space` → separate figures (`1:4_7 5`)
   * Set titles, labels, legends, formats, etc.

### Config file mode

Instead of answering prompts, you can save your setup as a `.json` config in the `Config/` folder.
Next runs can directly load it.

---

## 📑 Input File Format

### `.txt` files

Must contain a line starting with `TIME` followed by headers:

```
TIME V1 V2 V3
0.0   1.0 2.0 3.0
0.1   1.1 2.1 3.1
...
```

### `.csv` files

* Auto-detects `,` or `;` as delimiters
* Decimal commas are handled automatically (`1,23` → `1.23`)

---

## 📤 Export Options

* **Figures**: Saved in `Results/<filename>/`

  * Formats: `pdf`, `png`, `svg`
* **CSV** (optional):

  * All signals
  * Only selected signals

---

## 📦 Requirements

* Python ≥ 3.7
* Install dependencies:

```bash
pip install numpy matplotlib
```

---

## 👤 Author

Developed by **Nicolas Rabreau** (2025).
