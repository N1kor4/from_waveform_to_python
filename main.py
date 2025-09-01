#===============================================================================
# Waveform Plotter - Nicolas Rabreau 2025
#===============================================================================
# - Reads waveform data from .txt or .csv files
# - Allows interactive selection of signals
# - Supports .json configuration files for batch processing
# - Generates plots with:
#       * Custom subplot grid layout (user chooses columns)
#       * Optional zoom per subplot (custom x-limits or auto)
#       * Flexible legends:
#           - Global (one for all subplots)
#           - Per-subplot (one legend per subplot)
#           - Inside or outside placement
#       * Wide figures (16 in width)
#       * Engineering notation for time axis (s, ms, µs, ns, ps, fs)
#       * Scientific notation only in powers of 10^3
# - Optionally exports selected signals to CSV
# - Configuration can be saved/loaded via JSON
#===============================================================================

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import csv
import logging
import json
from matplotlib.ticker import ScalarFormatter, EngFormatter

#===============================================================================
# Configure logging and matplotlib settings
#===============================================================================
logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)
mpl.use('Agg')  # headless plotting (no GUI needed)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.titleweight": "medium",
    "axes.labelsize": 15,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "figure.dpi": 300,
})

# Color palette (Tol 16)
tol_16 = [
    "#4477AA", "#EE6677", "#228833", "#CCBB44",
    "#66CCEE", "#AA3377", "#BBBBBB", "#000000",
    "#332288", "#DDCC77", "#117733", "#88CCEE",
    "#882255", "#661100", "#999933", "#AA4499"
]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=tol_16)

#===============================================================================
# File readers
#===============================================================================
def read_waveform_file(file_path):
    """Read .txt waveform file (with TIME header)."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    try:
        header_index = next(i for i, line in enumerate(lines) if line.strip().startswith("TIME"))
    except StopIteration:
        raise ValueError("No 'TIME' line found in file.")
    headers = lines[header_index].strip().split()
    data_lines = lines[header_index + 1:]
    data = np.array([
        list(map(lambda x: float(x.replace('E','e')), line.strip().split()))
        for line in data_lines if line.strip()
    ])
    return headers, data

def read_csv_file(file_path):
    """Read CSV waveform file with auto delimiter detection (comma or semicolon)."""
    with open(file_path, 'r') as f:
        sample = f.readline()
        f.seek(0)
        delimiter = ';' if sample.count(';') > sample.count(',') else ','
        reader = csv.reader(f, delimiter=delimiter)
        headers = next(reader)
        data = []
        for row in reader:
            if not row: continue
            clean_row = []
            for x in row:
                x = x.strip()
                if not x: continue
                if ',' in x and '.' not in x:
                    x = x.replace(',', '.')  # fix decimal comma
                clean_row.append(float(x))
            if clean_row: data.append(clean_row)
        return headers, np.array(data)

#===============================================================================
# CSV Export
#===============================================================================
def save_data_as_csv(headers, data, output_csv_path):
    """Save waveform data to CSV."""
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)
    logging.info(f"Saved CSV: {output_csv_path}")

def plot_signals(headers, data, save_folder, subplot_signals_groups,
                 extension="pdf", figure_title=None, subplot_titles=None,
                 subplot_legends=None, ylabel="Value", xlabel="Time", linewidth=1.5,
                 ncols=1, xlimits=None, fig_width=16, fig_height=4,
                 legend_mode="per_subplot", legend_location="outside"):

    time = data[:, 0]
    os.makedirs(save_folder, exist_ok=True)

    # Build filename
    filename_base = figure_title.strip().replace(" ", "_") if figure_title else \
        "_".join("_".join(sig.replace("/", "_") for sig in group) for group in subplot_signals_groups)
    if len(filename_base) > 50:
        filename_base = filename_base[:47] + "..."
    filename = f"{filename_base}.{extension}"
    save_path = os.path.join(save_folder, filename)

    # Subplot grid
    nsubplots = len(subplot_signals_groups)
    nrows = int(np.ceil(nsubplots / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(fig_width*ncols, fig_height*nrows),
                             squeeze=False,
                             constrained_layout=True)

    # Ensure ylabel is a list
    if isinstance(ylabel, str):
        ylabel_list = [ylabel]*nsubplots
    else:
        ylabel_list = ylabel

    # Track all handles for global legend
    global_handles, global_labels = [], []

    for j, group in enumerate(subplot_signals_groups):
        row, col = divmod(j, ncols)
        ax = axes[row][col]

        handles, labels = [], []

        for sig in group:
            try:
                idx = headers.index(sig)
                line, = ax.plot(time, data[:, idx], label=sig, linewidth=linewidth)
                handles.append(line)
                labels.append(sig)
                if sig not in global_labels:
                    global_handles.append(line)
                    global_labels.append(sig)
            except ValueError:
                logging.warning(f"Signal '{sig}' not found.")

        # Subplot title
        title = subplot_titles[j] if subplot_titles and j < len(subplot_titles) else ", ".join(group)
        ax.set_title(title, fontweight="medium")
        ax.set_ylabel(ylabel_list[j] if j < len(ylabel_list) else "Value")
        ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)

        # X-limits
        if xlimits and j < len(xlimits) and xlimits[j]:
            ax.set_xlim(xlimits[j])

        # Scientific / engineering format
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_scientific(True)
        fmt.set_powerlimits((-3, 3))
        ax.xaxis.set_major_formatter(EngFormatter(unit='s'))
        ax.yaxis.set_major_formatter(fmt)

        # Per-subplot legend
        show_legend = subplot_legends[j] if subplot_legends and j < len(subplot_legends) else False
        if legend_mode == "per_subplot" and show_legend and handles:
            if len(labels) > 8:
                ncol_legend = 2
            else:
                ncol_legend = 1

            if legend_location == "inside":
                ax.legend(handles, labels, loc="best", frameon=True, ncol=ncol_legend)
            else:
                ax.legend(handles, labels, loc="center left",
                        bbox_to_anchor=(1.02, 0.5),
                        frameon=True,
                        ncol=ncol_legend)
    
    # X-labels only on bottom row
    for col in range(ncols):
        axes[-1, col].set_xlabel(xlabel)

    # Remove unused axes
    for j in range(nsubplots, nrows*ncols):
        row, col = divmod(j, ncols)
        fig.delaxes(axes[row][col])

    # Global figure title
    if figure_title:
        fig.suptitle(figure_title, fontweight="bold", x=0.5, ha="center")

    # Global legend
    if legend_mode == "global" and global_handles:
        ncols_legend = min(len(global_labels), 4)
        if legend_location == "inside":
            fig.legend(global_handles, global_labels, loc="upper right", ncol=ncols_legend, frameon=True)
        else:
            # Outside right
            fig.subplots_adjust(right=0.75)  # leave space for legend
            fig.legend(global_handles, global_labels,
                       loc="center left",
                       bbox_to_anchor=(1.01, 0.5),
                       ncol=ncols_legend,
                       frameon=True)

    # Save and close
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved subplot grid: {save_path}")


#===============================================================================
# Input parsing helpers
#===============================================================================
def parse_user_input(input_str):
    """
    Parse input like '1:3_5 6-7' into structured subplot groups.
    Returns a list of figures, each as list of subplots, each as list of indices.
    """
    parsed_groups = []
    for group_str in input_str.strip().split():
        subplot_strs = group_str.split('_')
        subplot_indices = []
        for part in subplot_strs:
            if not part.strip():  # skip empty parts
                continue
            indices = []
            for sub in part.split('-'):
                if not sub.strip():
                    continue
                if ':' in sub:
                    start, end = map(int, sub.split(':'))
                    indices.extend(range(start, end + 1))
                else:
                    indices.append(int(sub))
            if indices:
                subplot_indices.append(indices)
        if subplot_indices:
            parsed_groups.append(subplot_indices)
    return parsed_groups

def parse_time_with_suffix(value_str):
    """Parse values like '300p' or '2n' into float seconds."""
    suffixes = {'p':1e-12,'n':1e-9,'u':1e-6,'m':1e-3,'':1.0}
    value_str = value_str.strip()
    for s in suffixes:
        if value_str.endswith(s) and value_str[:-len(s) or None]:
            return float(value_str[:-len(s) or None])*suffixes[s]
    return float(value_str)

#===============================================================================
# Config management
#===============================================================================
def choose_config(Config_FOLDER = "Config"):
    """Interactive config loader."""
    config_files = sorted([f for f in os.listdir(Config_FOLDER) if f.endswith(".json")])
    if not config_files:
        logging.info("No config files found.")
    else:
        print("Available configuration files:")
        for i,f in enumerate(config_files):
            print(f"{i}: {f}")
        selection = input("\nSelect config number to load or Enter to setup manually: ").strip()
        if selection != "":
            try:
                idx = int(selection)
                path = os.path.join(Config_FOLDER, config_files[idx])
                with open(path,"r") as f: cfg=json.load(f)
                logging.info(f"Loaded config: {config_files[idx]}")
                return cfg
            except Exception as e:
                logging.warning(f"Failed to load config: {e}. Proceeding manually.")
        else:
            logging.info("Proceeding with manual setup.")
            print("\n----------------------------------------------------")
    return None

def save_config_interactive(config, Config_FOLDER = "Config"):
    """Interactive config saver."""
    fname = input("Enter a name for this configuration (without extension): ").strip()
    if not fname: return
    path = os.path.join(Config_FOLDER, f"{fname}.json")
    with open(path,"w") as f: json.dump(config,f,indent=2)
    logging.info(f"Configuration saved to {path}")

#===============================================================================
# Helpers for user/config input
#===============================================================================
def get_value(config, key, prompt, cast_func=str, default=None, choices=None):
    """
    Get a value from config if available, otherwise prompt the user.
    Only prompts if the key does not exist in config.
    """
    # Use value from config if it exists
    if config is not None and key in config:
        val = config[key]
        # Cast if necessary
        try:
            val_cast = cast_func(val) if val is not None else default
        except Exception:
            logging.warning(f"Invalid type in config for '{key}', using default.")
            val_cast = default
        # Validate choices
        if choices and val_cast not in choices:
            logging.warning(f"Value '{val_cast}' for '{key}' not in valid choices {choices}. Using default.")
            val_cast = default
        return val_cast

    # Otherwise, prompt the user
    while True:
        val = input(f"{prompt} [{default}]: ").strip() or str(default or "")
        try:
            val_cast = cast_func(val) if val else default
            if choices and val_cast not in choices:
                print(f"Invalid choice. Valid options: {choices}")
                continue
            return val_cast
        except Exception:
            print("Invalid input, try again.")

def choose_file(config, all_files, txt_files, csv_files):
    """Choose file either from config or by user selection."""
    if config and "file" in config and config["file"] in all_files:
        print(f"[CONFIG] Using file from config: {config['file']}")
        return config["file"]
    # Manual selection
    file_index_map, counter = {}, 0
    if txt_files:
        print("\nAvailable .txt files:")
        for f in txt_files:
            print(f"{counter}: {f}")
            file_index_map[counter] = f
            counter += 1
    if csv_files:
        print("\nAvailable .csv files:")
        for f in csv_files:
            print(f"{counter}: {f}")
            file_index_map[counter] = f
            counter += 1
    try:
        selected_index = int(input("\nSelect file number to process: "))
        return file_index_map[selected_index]
    except (ValueError, KeyError):
        print("Invalid selection.")
        return None

def get_plot_groups(config, headers):
    """Parse plot groups either from config or manual input."""
    if config and "plot_groups" in config:
        return parse_user_input(config["plot_groups"]), config["plot_groups"]
    print("\n----------------------------------------------------")
    print("\nSignals available:")
    for i, h in enumerate(headers):
        print(f"{i}: {h}")
    print("\nEnter the curve numbers you want to plot using the following syntax:")
    print(" - Use ':' for ranges (e.g., '1:4')")
    print(" - Use '-' to group curves into the same plot (e.g., '1-5-8')")
    print(" - Use '_' to separate subplots within a figure (e.g., '1:4_7')")
    print(" - Use spaces to separate figures (each will produce one figure)")
    user_input = input("\nYour selection: ")
    print("\n----------------------------------------------------\n")
    return parse_user_input(user_input), user_input

def get_figure_settings(config, fig_idx, subplot_groups):
    """
    Return all figure & subplot settings.
    If JSON config exists, values are taken directly.
    Otherwise, manual prompts are used for everything including:
    - Figure title, ncols, fig size
    - Subplot titles
    - Subplot legends
    - X-limits
    - Y-axis labels per subplot
    - Legend mode & location
    Also includes global parameters: extension, xlabel, linewidth
    """
    n_subplots = len(subplot_groups)

    if config and "figures" in config and fig_idx < len(config["figures"]):
        fig_cfg = config["figures"][fig_idx]
        fig_title = fig_cfg.get("figure_title", None)
        ncols = fig_cfg.get("ncols", 1)
        fig_width = fig_cfg.get("fig_width", 16)
        fig_height = fig_cfg.get("fig_height", 4)
        subplot_titles = fig_cfg.get("subplot_titles", [None]*n_subplots)
        subplot_legends = fig_cfg.get("subplot_legends", [True]*n_subplots)
        xlimits = fig_cfg.get("xlimits", [None]*n_subplots)
        legend_mode = fig_cfg.get("legend_mode", "per_subplot")
        legend_location = fig_cfg.get("legend_location", "outside")
        per_subplot_ylabel = fig_cfg.get("subplot_ylabels", [None]*n_subplots)

        # Global params
        extension = config.get("extension", "pdf")
        xlabel = config.get("xlabel", "Time")
        linewidth = config.get("linewidth", 0.5)
    else:
        # --- Manual interactive setup ---
        print(f"\n=== Figure {fig_idx+1} ===")

        # Figure settings
        print("Setting up global plot parameters:")
        fig_title = input("  Figure title (optional): ").strip() or None
        extension = get_value(None, "extension", "  Plot format (png/pdf/svg)", str, "pdf", ["png","pdf","svg"])
        linewidth = get_value(None, "linewidth", "  Line width", float, 0.5)
        xlabel = get_value(None, "xlabel", "  X-axis label", str, "Time")
        ncols = get_value(None, None, "  Number of subplot columns", int, 1)
        fig_width = get_value(None, None, "  Subfigure width", float, 16)
        fig_height = get_value(None, None, "  Subfigure height", float, 4)

        # Subplot settings
        subplot_titles, subplot_legends, xlimits, per_subplot_ylabel = [], [], [], []
        for j, subgroup in enumerate(subplot_groups):
            print(f"\n  -- Subplot {j+1} --")
            print(f"     Signals: {', '.join(subgroup)}")
            stitle = input("     Title (optional): ").strip() or None
            subplot_titles.append(stitle)
            legend = input("     Show legend? (y/n): ").strip().lower() == 'y'
            subplot_legends.append(legend)
            zoom_input = input("     Time limits (e.g., 300p 2n) [auto]: ").strip()
            if zoom_input:
                try:
                    xmin_str, xmax_str = zoom_input.split()
                    xmin = parse_time_with_suffix(xmin_str)
                    xmax = parse_time_with_suffix(xmax_str)
                    xlimits.append((xmin, xmax))
                except Exception:
                    print("     Invalid format, using auto limits.")
                    xlimits.append(None)
            else:
                xlimits.append(None)
            # Y-axis label
            y_label = input(f"     Y-axis label [Voltage (V)]: ").strip() or "Voltage (V)"
            per_subplot_ylabel.append(y_label)

        # Legend mode & location
        print("\n  -- Legend Settings --")
        legend_mode_input = get_value(None, None, "  Legend mode (1: global, 2: per_subplot, 3: none)", str, "2", ["1","2","3"])
        legend_mode_map = {"1": "global", "2": "per_subplot", "3": "none"}
        legend_mode = legend_mode_map[legend_mode_input]
        legend_loc_input = get_value(None, None, "  Legend location (1: inside, 2: outside)", str, "2", ["1","2"])
        legend_loc_map = {"1": "inside", "2": "outside"}
        legend_location = legend_loc_map[legend_loc_input]

    return (fig_title, ncols, fig_width, fig_height,
            subplot_titles, subplot_legends, xlimits,
            per_subplot_ylabel, legend_mode, legend_location,
            extension, xlabel, linewidth)

#===============================================================================
# JSON flattening helper
#===============================================================================

def flatten_plot_groups(parsed):
    """
    Ensure that all sublists contain integers, not nested lists.
    """
    flat = []
    for fig in parsed:
        fig_flat = []
        for subplot in fig:
            # Flatten any nested list of indices
            indices = []
            for item in subplot:
                if isinstance(item, list):
                    indices.extend(item)
                else:
                    indices.append(item)
            fig_flat.append(indices)
        flat.append(fig_flat)
    return flat

#===============================================================================
# Main program
#===============================================================================
def main():
    print("\n====== Welcome to Waveform Plotter ======")
    print("This program plots waveform data from .txt or .csv files.\n")

    # --- Setup folders ---
    waveform_folder, results_folder, config_folder = "Waveform", "Results", "Config"
    for folder, msg in [(waveform_folder, "waveform"), (config_folder, "config")]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"[INFO] '{folder}' folder not found. It has been created.")
            print(f"Please put your {msg} files inside this folder.")
            return

    # --- Config selection ---
    config_data = choose_config(Config_FOLDER=config_folder)
    use_json_path = bool(config_data)  # JSON path vs manual path

    # --- File selection ---
    all_files = [f for f in os.listdir(waveform_folder) if f.lower().endswith((".txt", ".csv"))]
    if not all_files:
        print("\n[INFO] No .txt or .csv files found in 'Waveform' folder. Please add your waveform files.")
        return
    txt_files = sorted([f for f in all_files if f.endswith(".txt")])
    csv_files = sorted([f for f in all_files if f.endswith(".csv")])
    
    if use_json_path:
        file_name = config_data["file"]
        if not os.path.exists(os.path.join(waveform_folder, file_name)):
            logging.warning(f"Waveform file '{file_name}' from JSON config does not exist. Please check the file path.")
            # Optionally, fallback to manual selection:
            file_name = choose_file(None, all_files, txt_files, csv_files)
            use_json_path = False  # Switch to manual if file not found
    else:
        file_name = choose_file(None, all_files, txt_files, csv_files)
        if not file_name:
            return

    # --- Load data ---
    file_path = os.path.join(waveform_folder, file_name)
    if file_name.endswith(".csv"):
        headers, data = read_csv_file(file_path)
        allow_csv_prompt = False
    else:
        headers, data = read_waveform_file(file_path)
        allow_csv_prompt = True

    # --- Output folder ---
    file_base_name = os.path.splitext(file_name)[0]
    output_folder = os.path.join(results_folder, file_base_name)
    os.makedirs(output_folder, exist_ok=True)

    # --- Signal selection (figure by figure) ---
    if use_json_path:
        # parse string from JSON into list-of-lists-of-indices
        plot_groups = flatten_plot_groups(parse_user_input(config_data["plot_groups"]))
        user_input = config_data["plot_groups"]
    else:
        plot_groups, user_input = get_plot_groups(None, headers)


    # --- Loop over figures safely ---
    figures_list = []
    all_selected_indices = set()

    for fig_idx, figure_group in enumerate(plot_groups):
        # Map indices to header names for this figure
        subplot_groups = []
        for subplot in figure_group:
            subplot_headers = []
            for idx in subplot:
                if isinstance(idx, int) and 0 <= idx < len(headers):
                    subplot_headers.append(headers[idx])
                else:
                    logging.warning(f"Invalid index {idx} skipped for figure {fig_idx}")
            if subplot_headers:
                subplot_groups.append(subplot_headers)

        # Skip figure if no valid signals
        if not subplot_groups:
            logging.warning(f"No valid signals to plot for figure {fig_idx}, skipping this figure.")
            continue

        # --- Get figure & subplot settings ---
        (fig_title, ncols, fig_width, fig_height,
        subplot_titles, subplot_legends, xlimits,
        per_subplot_ylabel, legend_mode, legend_location,
        extension, xlabel, linewidth) = get_figure_settings(config_data if use_json_path else None, fig_idx, subplot_groups)

        # Track selected indices for CSV export
        for subplot in figure_group:
            for idx in subplot:
                if isinstance(idx, int) and 0 <= idx < len(headers):
                    all_selected_indices.add(idx)

        # --- Plot figure ---
        plot_signals(
            headers, data, output_folder, subplot_groups,
            extension,
            figure_title=fig_title,
            subplot_titles=subplot_titles,
            subplot_legends=subplot_legends,
            ylabel=per_subplot_ylabel,
            xlabel=xlabel,
            linewidth=linewidth,
            ncols=ncols,
            xlimits=xlimits,
            fig_width=fig_width,
            fig_height=fig_height,
            legend_mode=legend_mode,
            legend_location=legend_location
        )

        # --- Save per-figure config ---
        figure_cfg = {
            "figure_title": fig_title,
            "ncols": ncols,
            "fig_width": fig_width,
            "fig_height": fig_height,
            "subplot_titles": subplot_titles,
            "subplot_legends": subplot_legends,
            "xlimits": xlimits,
            "legend_mode": legend_mode,
            "legend_location": legend_location,
            "subplot_ylabels": per_subplot_ylabel,
        }
        figures_list.append(figure_cfg)


    # --- CSV export, only if manual ---
    if allow_csv_prompt and not use_json_path:
        print("\n----------------------------------------------------")
        print("\nExport options:\n1: All signals\n2: Selected only\n3: None")
        csv_choice = input("\nYour choice: ").strip()
        time_header = headers[0] if headers[0].upper() == "TIME" else "TIME"
        if csv_choice == "1":
            save_data_as_csv(headers, data, os.path.join(output_folder, f"{file_base_name}_all.csv"))
        elif csv_choice == "2":
            selected_headers = [time_header] + [headers[i] for i in sorted(all_selected_indices)]
            selected_data = data[:, [0] + sorted(all_selected_indices)]
            save_data_as_csv(selected_headers, selected_data, os.path.join(output_folder, f"{file_base_name}_selected.csv"))

    # --- Save config if manual ---
    if not use_json_path:
        print("\n----------------------------------------------------")
        if input("\nDo you want to save this setup as a config file? (y/n): ").strip().lower() == 'y':
            config_to_save = {
                "file": file_name,
                "plot_groups": user_input,
                "extension": extension,
                "xlabel": xlabel,
                "figures": figures_list
            }
            save_config_interactive(config_to_save, Config_FOLDER=config_folder)

    print("\n----------------------------------------------------")
    print("\nDone.\n")


#===============================================================================
if __name__ == "__main__":
    main()
