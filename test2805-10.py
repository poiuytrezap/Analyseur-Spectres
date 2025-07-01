import os
import pandas as pd
import matplotlib
matplotlib.use('TKAgg') 
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import seaborn as sns
import threading

from io import BytesIO
from pathlib import Path
from tkinter import ttk, filedialog, simpledialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk, ImageOps
from scipy.optimize import nnls
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import NMF, non_negative_factorization
from sklearn.linear_model import Lasso

GLOBAL_IMAGE_REFERENCES = []

GLOBAL_FIRST_ALPHA_COMP = None
GLOBAL_FIRST_REVERSE_COMP = None

GLOBAL_FIRST_ALPHA_COMP_2 = None
GLOBAL_FIRST_REVERSE_COMP_2 = None

from PIL import ImageTk, ImageOps

def optimize_lasso(
    X,
    H_current,
    sample_names,
    all_common_q,
    directory_path,
    suffix="lasso_coeffs_optimized",
    component_ids=None,
    component_colors=None,
    parent=None,
    dynamic_window=None
):
    
    n_samples, n_q = X.shape
    n_composantes = H_current.shape[0]

    if component_ids is None:
        component_ids = [f"Comp_{i}" for i in range(n_composantes)]

    coeffs_init = []
    for i in range(n_samples):
        lasso = Lasso(alpha=0.01, positive=True, max_iter=10000)
        weights = np.sqrt(np.abs(X[i]))
        base_curves = H_current.T   
        Xw = base_curves * weights[:, None]  
        yw = X[i] * weights                 
        lasso.fit(Xw, yw)
        coeffs_init.append(lasso.coef_)
    coeffs_init = np.vstack(coeffs_init)  

    W_init = coeffs_init.copy()           
    H_init = np.maximum(H_current, 0)      
    W_o, H_o, _ = non_negative_factorization(
        X,
        W=W_init,
        H=H_init,
        n_components=H_init.shape[0],
        init='custom',
        update_H=True,
        solver='cd',
        beta_loss='frobenius',
        max_iter=10000,
        tol=1e-3,
        random_state=42,
        alpha_W=1e-5,
        alpha_H=1e-5,
        l1_ratio=0
    )
    H_opt = np.maximum(H_o, 0)  

    coeffs_opt = []
    for i in range(n_samples):
        lasso2 = Lasso(alpha=0.01, positive=True, max_iter=10000)
        weights2 = np.sqrt(np.abs(X[i]))
        base_curves2 = H_opt.T   
        Xw2 = base_curves2 * weights2[:, None]
        yw2 = X[i] * weights2
        lasso2.fit(Xw2, yw2)
        coeffs_opt.append(lasso2.coef_)
    coeffs_opt = np.vstack(coeffs_opt)  

    reconstructed = coeffs_opt @ H_opt  

    errors = np.sqrt(np.mean((X - reconstructed) ** 2, axis=1))  
    residuals = X - reconstructed                              
    mse_per_channel = np.mean(residuals ** 2, axis=0)          
    error_spectrum = np.sqrt(mse_per_channel)                  

    os.makedirs(directory_path, exist_ok=True)
    df_coeffs = pd.DataFrame(coeffs_opt, columns=[f"Coeff_{cid}" for cid in component_ids])
    df_coeffs.insert(0, "Sample_Name", sample_names)
    df_coeffs.to_csv(os.path.join(directory_path, f"{suffix}.csv"), index=False)
    df_rms = pd.DataFrame({"q": all_common_q, "RMSE": error_spectrum})
    df_rms.to_csv(os.path.join(directory_path, f"RMS_{suffix}.csv"), index=False)

    return coeffs_opt, reconstructed, errors, error_spectrum

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import fcluster, linkage

def cluster_and_average_curves(H, threshold=0.1, method='cosine'):
    dists = pdist(H, metric=method)
    Z = linkage(dists, method='average')
    clusters = fcluster(Z, t=threshold, criterion='distance')
    n_families = np.unique(clusters).size

    family_means = []
    for label in np.unique(clusters):
        members = H[clusters == label]
        mean_curve = members.mean(axis=0)
        family_means.append(mean_curve)
    family_means = np.array(family_means)
    return clusters, family_means

def display_images_and_select_refinement(parent, images_for_ref, current_ids, title="Sélection", labels=None):
    if labels is None:
        labels = current_ids[:]

    sel_indices = set()
    thumbnail_refs = []  

    popup = tk.Toplevel(parent)
    popup.title(title)
    popup.transient(parent)
    popup.grab_set()
    
    container = tk.Frame(popup)
    container.pack(fill="both", expand=True)
    canvas = tk.Canvas(container, height=400)
    scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    def toggle(idx, btn):
        if idx in sel_indices:
            sel_indices.remove(idx)
            img = images_for_ref[idx].copy()
            img.thumbnail((100, 100))
            img_tk = ImageTk.PhotoImage(img)
            thumbnail_refs.append(img_tk) 
            btn.configure(image=img_tk)
            btn.image = img_tk
        else:
            sel_indices.add(idx)
            img = images_for_ref[idx].copy()
            img = ImageOps.expand(img, border=5, fill="red")
            img.thumbnail((100, 100))
            img_tk = ImageTk.PhotoImage(img)
            thumbnail_refs.append(img_tk)
            btn.configure(image=img_tk)
            btn.image = img_tk

    cols = 4
    for i, pil_img in enumerate(images_for_ref):
        row = i // cols
        col = i % cols

        thumb = pil_img.copy()
        thumb.thumbnail((100, 100))
        img_tk = ImageTk.PhotoImage(thumb)
        thumbnail_refs.append(img_tk)

        btn = tk.Button(scrollable_frame, image=img_tk, bd=0,
                        command=lambda idx=i, b=None: toggle(idx, b))
        btn.configure(command=lambda idx=i, b=btn: toggle(idx, b))
        btn.image = img_tk
        btn.grid(row=row * 2, column=col, padx=5, pady=5)

        lbl_text = labels[i] if i < len(labels) else ""
        lbl = tk.Label(scrollable_frame, text=lbl_text, font=("Arial", 8))
        lbl.grid(row=row * 2 + 1, column=col, padx=5, pady=(0,10))

    def on_validate():
        popup.grab_release()
        popup.destroy()

    btn_validate = tk.Button(popup, text="Valider", command=on_validate)
    btn_validate.pack(pady=10)

    parent.wait_window(popup)

    return sorted(list(sel_indices))

def re_run_nnls_for_rms(X, H_current, sample_names, all_q):
    if H_current.size == 0:
        return None

    reconstructed_data = []
    for i in range(X.shape[0]):
        coefs, _ = nnls(H_current.T, X[i])
        recon = H_current.T @ coefs
        reconstructed_data.append(recon)
    reconstructed_data = np.array(reconstructed_data)

    error = np.mean((X - reconstructed_data) ** 2)
    rms_global = np.sqrt(error)
    return rms_global

def read_and_clean_data(file_path, trim_start=0, trim_end=0):
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        data_start = None
        for i, line in enumerate(lines):
            if not line.lstrip().startswith("#"):
                if 'q(A-1)' in line or 'I(q)' in line:
                    data_start = i + 1
                else:
                    data_start = i
                break

        if data_start is None:
            raise ValueError(f"Aucune ligne de données trouvée dans {file_path}")

        df = pd.read_csv(
            file_path,
            skiprows=data_start,
            sep=r'\s+',
            header=None,
            names=['q', 'I(q)', 'Sig(q)'],
            comment='#',
            dtype=float,
            engine='python'
        )

        df = df.dropna()
        if trim_end > 0:
            df = df.iloc[trim_start:-trim_end]
        else:
            df = df.iloc[trim_start:]

        return df

    except Exception as e:
        print(f"[Error processing {file_path}]: {e}")
        return pd.DataFrame()

def interpolate_data(common_q, data_df):
    interp_func = interp1d(
        data_df['q'],
        data_df['I(q)'],
        kind='linear',
        bounds_error=False,
        fill_value='extrapolate'
    )
    return interp_func(common_q)

def apply_gaussian_smoothing(data, sigma=0):
    return gaussian_filter1d(data, sigma=sigma)

def find_optimal_components_by_variance(baseline_corrected_df, max_components=10, *thresholds):
    total_variance = np.sum(baseline_corrected_df.values**2)
    explained_variances = []
    rms_values = []

    default_thresholds = [
        0.9997, 0.9995, 0.9994, 0.9991,
        0.9989, 0.9987, 0.9985, 0.9983
    ]
    thresholds = thresholds if thresholds else default_thresholds

    for n_components in range(1, max_components + 1):
        nmf_model = NMF(
            n_components=n_components,
            init='random',
            random_state=42,
            max_iter=10000
        )
        nmf_features = nmf_model.fit_transform(baseline_corrected_df)
        reconstruction = np.dot(nmf_features, nmf_model.components_)
        explained_variance = 1 - np.sum(
            (baseline_corrected_df.values - reconstruction) ** 2
        ) / total_variance
        explained_variances.append(explained_variance)

        rms = np.sqrt(np.mean((baseline_corrected_df - reconstruction) ** 2))
        rms_values.append(rms)

        if n_components <= len(thresholds) \
           and explained_variance >= thresholds[n_components - 1]:
            return n_components, explained_variances, rms_values

    return n_components, explained_variances, rms_values

def compute_nmf_components(
        data_dict,
        data_files_list,
        order_name,
        all_common_q,
        save_dir,
        initial_files=1,
        step=1,
        max_components=10,
        sigma=10,
        thresholds=(0.9997, 0.9995, 0.9994, 0.9991,
                    0.9989, 0.9987, 0.9985, 0.9983),
        progress_callback=None
    ):

    global GLOBAL_FIRST_ALPHA_COMP
    global GLOBAL_FIRST_REVERSE_COMP

    global GLOBAL_FIRST_ALPHA_COMP_2
    global GLOBAL_FIRST_REVERSE_COMP_2

    all_components = []
    images_pil = []
    peak_positions = []
    peak_q_values = []
    components_spectra_indices = []
    component_file_names = []

    start_file_index = 0
    current_files = initial_files
    component_global_index = 0

    os.makedirs(save_dir, exist_ok=True)

    while start_file_index + current_files <= len(data_files_list):
        data_batch = {}
        files_batch = data_files_list[
            start_file_index: start_file_index + current_files
        ]
        for f_ in files_batch:
            if f_ in data_dict:
                data_batch[f_] = data_dict[f_]

        if not data_batch:
            break

        interpolated_data = []
        for df_ in data_batch.values():
            arr_ = interpolate_data(all_common_q, df_)
            arr_smooth = apply_gaussian_smoothing(arr_, sigma=sigma)
            interpolated_data.append(arr_smooth)

        df_interpol = pd.DataFrame(interpolated_data, columns=all_common_q)
        n_opt, _, _ = find_optimal_components_by_variance(
            df_interpol, max_components, *thresholds
        )

        nmf_model = NMF(
            n_components=n_opt,
            init='random',
            random_state=42,
            max_iter=10000
        )
        nmf_features = nmf_model.fit_transform(df_interpol)
        nmf_comps = nmf_model.components_

        for i in range(n_opt):
            comp_ = nmf_comps[i]
            all_components.append(comp_)

            peak_idx = np.argmax(comp_)
            peak_positions.append(peak_idx)
            peak_q = all_common_q[peak_idx]
            peak_q_values.append(peak_q)

            if start_file_index == 0 and i == 0:
                if order_name.lower().startswith("alphabetical"):
                    GLOBAL_FIRST_ALPHA_COMP = comp_
                elif order_name.lower().startswith("reverse"):
                    GLOBAL_FIRST_REVERSE_COMP = comp_

            if start_file_index == 0 and i == 0:
                if order_name.lower().startswith("reconstructed_alphabetical"):
                    GLOBAL_FIRST_ALPHA_COMP_2 = comp_
                elif order_name.lower().startswith("reconstructed_reverse"):
                    GLOBAL_FIRST_REVERSE_COMP_2 = comp_

            originating_file = data_files_list[
                start_file_index + (peak_idx % current_files)
            ]
            component_file_names.append(originating_file)

            fig = plt.Figure(figsize=(4, 3))
            ax = fig.add_subplot(111)
            ax.plot(all_common_q, comp_)
            ax.set_title(f"{order_name} Comp {component_global_index}")
            fname = os.path.join(
                save_dir,
                f"{order_name}_Comp_{component_global_index}.png"
            )
            fig.savefig(fname)
            plt.close(fig)

            im_ = Image.open(fname)
            images_pil.append(im_.copy())
            components_spectra_indices.append(start_file_index + i)
            component_global_index += 1

        start_file_index += step
        current_files = min(
            current_files + step,
            len(data_files_list) - start_file_index
        )

        if progress_callback:
            progress_callback()

    combined = list(zip(
        peak_positions,
        peak_q_values,
        all_components,
        images_pil,
        components_spectra_indices,
        component_file_names
    ))
    combined.sort(key=lambda x: x[0])

    if combined:
        peakpos_sorted, peak_q_sorted, comps_sorted, imgs_sorted, specidx_sorted, file_names_sorted = zip(*combined)
    else:
        peakpos_sorted, peak_q_sorted = [], []
        comps_sorted, imgs_sorted = [], []
        specidx_sorted, file_names_sorted = [], []

    for idx, comp in enumerate(comps_sorted):
        csv_comp_name = f'Component_{idx}_{order_name}.csv'
        df_comp = pd.DataFrame({
            'q': all_common_q,
            'I(q)': comp,
            'Sig(q)': np.zeros_like(comp)
        })
        df_comp.to_csv(
            os.path.join(save_dir, csv_comp_name),
            index=False
        )

    return (
        list(comps_sorted),
        list(imgs_sorted),
        list(specidx_sorted),
        list(peakpos_sorted),
        list(peak_q_sorted),
        list(file_names_sorted)
    )

class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Analyse de Spectres - Interface centrale")
        self.geometry("1400x800")
        self.loaded_dir = None

        self.data_dict_all = {}
        self.all_common_q = None
        self.sample_names = []
        self.X = None

        self.first_nmf_components = []
        self.first_nmf_images = []
        self.first_nmf_labels = []
        self.selected_indices_nmf1 = []
        self.second_nmf_history = []
        self._nmf2_history_index = 0
        self.show_lasso2_optimized = False

        self.reconstruct_dict = {}
        self.reconstructed_X = None
        self.second_nmf_components = []
        self.second_nmf_images = []
        self.second_nmf_labels = []
        self.selected_indices_nmf2 = []
        self.auto2_var = tk.BooleanVar(value=False)

        self.H2_initial = None
        self.error_spectrum2 = None
        self.transferred = False

        self.include_rms2 = False

        self.avg_rms_var = tk.StringVar(value="N/A")
        self.initial_lasso_error_var = tk.StringVar(value="N/A")
        self.optimized_lasso_error_var = tk.StringVar(value="N/A")
        self.total_error_var = tk.StringVar(value="N/A")

        self.auto_select_var = tk.BooleanVar(value=False)

        self.create_menu()
        self.create_tabs()
        self.populate_tab_load()
        self.populate_tab_analysis()
        self.populate_tab_nmf2()
        self.create_log_box()

    def create_menu(self):
        self.menu_bar = tk.Menu(self)
        menu_file = tk.Menu(self.menu_bar, tearoff=0)
        menu_file.add_command(label="Ouvrir dossier .dat", command=self.load_directory)
        menu_file.add_separator()
        menu_file.add_command(label="Quitter", command=self.quit)
        self.menu_bar.add_cascade(label="Fichier", menu=menu_file)

        menu_help = tk.Menu(self.menu_bar, tearoff=0)
        menu_help.add_command(label="À propos", command=self.dummy)
        self.menu_bar.add_cascade(label="Aide", menu=menu_help)

        self.config(menu=self.menu_bar)

    def create_tabs(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        self.tab_load = ttk.Frame(self.notebook)
        self.tab_analysis = ttk.Frame(self.notebook)
        self.tab_nmf2 = ttk.Frame(self.notebook)
        self.tab_visual = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_load, text="Chargement")
        self.notebook.add(self.tab_analysis, text="Analyse NMF & Lasso")
        self.notebook.add(self.tab_nmf2, text="NMF Reconstruite & Lasso")

    def create_log_box(self):
        self.log_box = tk.Text(self, height=6, state='disabled', bg="#f2f2f2", wrap='word')
        self.log_box.pack(fill='x', padx=10, pady=5)

    def log(self, msg):
        self.log_box.config(state='normal')
        self.log_box.insert('end', msg + '\n')
        self.log_box.config(state='disabled')
        self.log_box.see('end')

    def dummy(self):
        self.log("Fonction non implémentée")

    def populate_tab_load(self):
        frame = ttk.Frame(self.tab_load, padding=10)
        frame.pack(fill='both', expand=True)

        lbl = ttk.Label(frame, text="Choisissez un dossier contenant des fichiers .dat")
        lbl.grid(row=0, column=0, columnspan=2, pady=(0,10))

        load_button = ttk.Button(frame, text="Sélectionner Dossier .dat", command=self.load_directory)
        load_button.grid(row=1, column=0, padx=(0,10))

        import_csv_button = ttk.Button(frame, text="Importer courbe CSV", command=self.import_csv_curves)
        import_csv_button.grid(row=1, column=1, padx=(10,0))

        self.load_label = ttk.Label(frame, text="Aucun dossier sélectionné.")
        self.load_label.grid(row=2, column=0, columnspan=2, sticky='w', pady=(10,0))

        self.preview_listbox = tk.Listbox(frame, height=10, width=80)
        self.preview_listbox.grid(row=3, column=0, columnspan=2, pady=10, sticky='nsew')

        scrollbar = ttk.Scrollbar(frame, orient='vertical', command=self.preview_listbox.yview)
        scrollbar.grid(row=3, column=2, sticky='ns', pady=10)
        self.preview_listbox.config(yscrollcommand=scrollbar.set)

        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(3, weight=1)

    def load_directory(self):
        dir_path = filedialog.askdirectory(
            title="Sélectionnez le dossier contenant des fichiers .dat"
        )
        if not dir_path:
            self.log("Sélection de dossier annulée.")
            return

        all_dat = [
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if f.lower().endswith('.dat')
        ]
        if not all_dat:
            messagebox.showwarning("Aucun fichier .dat", "Le dossier ne contient pas de fichiers .dat.")
            return

        self.loaded_dir = dir_path
        self.load_label.config(text=f"Dossier sélectionné : {dir_path} ({len(all_dat)} fichiers .dat)")
        self.loaded_files = all_dat

        self.preview_listbox.delete(0, 'end')
        for f in all_dat:
            self.preview_listbox.insert('end', os.path.basename(f))

        self.log(f"{len(all_dat)} fichiers .dat trouvés dans {dir_path}.")

    def import_csv_curves(self):
        paths = filedialog.askopenfilenames(
            title="Sélectionner un ou plusieurs fichiers .csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not paths:
            return

        n_import = 0
        for p in paths:
            try:
                df_csv = pd.read_csv(p)

                if 'q' not in df_csv.columns or 'I(q)' not in df_csv.columns:
                    messagebox.showwarning(
                        "CSV invalide",
                        f"Le fichier « {os.path.basename(p)} » n'a pas les colonnes 'q' et 'I(q)'."
                    )
                    continue

                df_csv = df_csv[['q', 'I(q)']].copy()
                df_csv['q'] = pd.to_numeric(df_csv['q'], errors='coerce')
                df_csv['I(q)'] = pd.to_numeric(df_csv['I(q)'], errors='coerce')
                df_csv = df_csv.dropna(subset=['q', 'I(q)'])

                df_csv = df_csv.sort_values('q').drop_duplicates(subset='q')
                q_csv = df_csv['q'].values
                intensity_csv = df_csv['I(q)'].values

                if q_csv.size < 2:
                    messagebox.showwarning(
                        "CSV non exploitable",
                        f"Le fichier « {os.path.basename(p)} » contient moins de deux points distincts en 'q'."
                    )
                    continue

                if self.all_common_q is None:
                    self.all_common_q = q_csv.copy()
                    I_data = intensity_csv.copy()
                else:
                    if q_csv.shape == self.all_common_q.shape and np.allclose(q_csv, self.all_common_q):
                        I_data = intensity_csv.copy()
                    else:
                        messagebox.showwarning(
                            "Grille incompatible",
                            f"Le fichier « {os.path.basename(p)} » n'a pas la même grille 'q' que celle importée précédemment."
                        )
                        continue

                self.first_nmf_components.append(I_data)

                fig = plt.Figure(figsize=(4, 3))
                ax = fig.add_subplot(111)
                ax.plot(self.all_common_q, I_data, color='green')
                ax.set_title(f"CSV : {os.path.basename(p)}")
                tmp_fname = f"thumb_{os.path.splitext(os.path.basename(p))[0]}.png"
                tmp_path = os.path.join(os.path.dirname(p), tmp_fname)
                fig.savefig(tmp_path)
                plt.close(fig)

                im_ = Image.open(tmp_path)
                self.first_nmf_images.append(im_.copy())
                self.first_nmf_labels.append(os.path.basename(p))

                self.second_nmf_components.append(I_data)

                self.second_nmf_images.append(im_.copy())
                self.second_nmf_labels.append(os.path.basename(p))

                n_import += 1

            except Exception as e:
                messagebox.showerror(
                    "Erreur import CSV",
                    f"Impossible de traiter « {os.path.basename(p)} » :\n{e}"
                )
                continue

        if n_import > 0:
            self.display_nmf1_components()   
            self.display_nmf2_components()   
            self.log(f"{n_import} courbe(s) CSV importée(s).")

    def display_family_means_nmf1(self):
        if self.selected_indices_nmf1:
            comps = [self.first_nmf_components[i] for i in self.selected_indices_nmf1]
        else:
            comps = self.first_nmf_components

        H = np.array(comps)
        clusters, family_means = cluster_and_average_curves(H, threshold=0.001, method='cosine')
    
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        for i, curve in enumerate(family_means):
            ax.plot(self.all_common_q, curve, label=f'Famille {i+1}')
        ax.set_title("Moyennes par famille de composantes (NMF1)")
        ax.set_xlabel("q")
        ax.set_ylabel("Intensité")
        ax.legend(fontsize=8)

        popup = tk.Toplevel(self)
        popup.title("Family Means (NMF1)")
        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        self.add_family_means_to_nmf1()
    
    def generate_image(self, curve, q_grid, color='red'):
        fig = plt.Figure(figsize=(4, 3))
        ax = fig.add_subplot(111)
        ax.plot(q_grid, curve, color=color, linewidth=2)
        ax.set_title("Family Mean" if color == 'red' else "Component")
        buf = BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        im = Image.open(buf)
        return im


    def add_family_means_to_nmf1(self, threshold=0.001):
        if hasattr(self, "family_mean_indices_nmf1") and self.family_mean_indices_nmf1:
            return
        clusters, family_means = cluster_and_average_curves(
            np.array(self.first_nmf_components), threshold=threshold, method='cosine'
        )
        self.family_mean_indices_nmf1 = []
        start_idx = len(self.first_nmf_components)
        for i, fam_mean in enumerate(family_means):
            self.first_nmf_components.append(fam_mean)
            self.first_nmf_images.append(self.generate_image(fam_mean, self.all_common_q, color='red'))
            self.first_nmf_labels.append(f"[FAMILY MEAN #{i+1}]")
            self.family_mean_indices_nmf1.append(start_idx + i)
        self.display_nmf1_components()

    def add_family_means_to_nmf2(self, threshold=0.001):

        if hasattr(self, "family_mean_indices_nmf2") and self.family_mean_indices_nmf2:
            return
        clusters, family_means = cluster_and_average_curves(
            np.array(self.second_nmf_components), threshold=threshold, method='cosine'
        )
        self.family_mean_indices_nmf2 = []
        start_idx = len(self.second_nmf_components)
        for i, fam_mean in enumerate(family_means):
            self.second_nmf_components.append(fam_mean)
            self.second_nmf_images.append(self.generate_image(fam_mean, self.all_common_q, color='red'))
            self.second_nmf_labels.append(f"[FAMILY MEAN #{i+1}]")
            self.family_mean_indices_nmf2.append(start_idx + i)
        self.display_nmf2_components()

    def populate_tab_analysis(self):
        paned = ttk.Panedwindow(self.tab_analysis, orient='horizontal')
        paned.pack(fill='both', expand=True, pady=10, padx=10)

        self.frame_nmf1 = ttk.Frame(paned)
        paned.add(self.frame_nmf1, weight=1)

        self.frame_lasso = ttk.Frame(paned)
        paned.add(self.frame_lasso, weight=1)

        self._init_nmf1_frame(self.frame_nmf1)
        self._init_lasso_frame(self.frame_lasso)

        transfer_btn = ttk.Button(
            self.frame_lasso,
            text="Transférer vers NMF2",
            command=self.transfer_to_nmf2
        )
        transfer_btn.pack(anchor='n', pady=(0, 5))
        family_btn = ttk.Button(
            self.frame_nmf1,
            text="Afficher Family Means",
            command=self.display_family_means_nmf1
        )
        family_btn.pack(pady=10)

        self.progress = ttk.Progressbar(self.tab_analysis, orient='horizontal', mode='determinate')
        self.progress.pack(fill='x', padx=10, pady=(0,5))

    def _init_nmf1_frame(self, parent):
        frame = parent

        param_frame = ttk.LabelFrame(frame, text="Traitement NMF", padding=10)
        param_frame.pack(fill='x', padx=5, pady=(0,10))

        self.nmf_button = ttk.Button(param_frame, text="Lancer la NMF", command=self.launch_nmf)
        self.nmf_button.grid(row=0, column=0, padx=(0,10))

        self.auto_checkbox = ttk.Checkbutton(
            param_frame,
            text="Auto-sélection composantes 0 (Alpha & Reverse)",
            variable=self.auto_select_var,
            command=self.on_auto_checkbox_toggle
        )
        self.auto_checkbox.grid(row=0, column=1, padx=(0,10))

        container = ttk.Frame(frame)
        container.pack(fill='both', expand=True, padx=5, pady=5)

        self.nmf_scroll_canvas = tk.Canvas(container)
        self.nmf_scroll_canvas.pack(side='left', fill='both', expand=True)

        v_scrollbar = ttk.Scrollbar(container, orient='vertical', command=self.nmf_scroll_canvas.yview)
        v_scrollbar.pack(side='right', fill='y')
        self.nmf_scroll_canvas.configure(yscrollcommand=v_scrollbar.set)

        self.nmf_components_frame = ttk.Frame(self.nmf_scroll_canvas)
        self.nmf_components_window = self.nmf_scroll_canvas.create_window((0, 0), window=self.nmf_components_frame, anchor='nw')
        self.nmf_components_frame.bind(
            "<Configure>",
            lambda e: self.nmf_scroll_canvas.configure(scrollregion=self.nmf_scroll_canvas.bbox("all"))
        )

    def _init_lasso_frame(self, parent):
        frame = parent

        lbl = ttk.Label(frame, text="Résultats Lasso :", font=("Arial", 12, "bold"))
        lbl.pack(anchor='w', padx=5, pady=(0,5))

        container = ttk.Frame(frame)
        container.pack(fill='both', expand=True, padx=5, pady=5)

        self.lasso_scroll_canvas = tk.Canvas(container)
        self.lasso_scroll_canvas.pack(side='left', fill='both', expand=True)

        v_scrollbar = ttk.Scrollbar(container, orient='vertical', command=self.lasso_scroll_canvas.yview)
        v_scrollbar.pack(side='right', fill='y')
        self.lasso_scroll_canvas.configure(yscrollcommand=v_scrollbar.set)

        self.lasso_results_frame = ttk.Frame(self.lasso_scroll_canvas)
        self.lasso_results_window = self.lasso_scroll_canvas.create_window((0, 0), window=self.lasso_results_frame, anchor='nw')
        self.lasso_results_frame.bind(
            "<Configure>",
            lambda e: self.lasso_scroll_canvas.configure(scrollregion=self.lasso_scroll_canvas.bbox("all"))
        )

    def on_auto_checkbox_toggle(self):
        if self.auto_select_var.get():
            idx_alpha = None
            idx_reverse = None
            for idx, comp in enumerate(self.first_nmf_components):
                if GLOBAL_FIRST_ALPHA_COMP is not None and np.allclose(comp, GLOBAL_FIRST_ALPHA_COMP):
                    idx_alpha = idx
                if GLOBAL_FIRST_REVERSE_COMP is not None and np.allclose(comp, GLOBAL_FIRST_REVERSE_COMP):
                    idx_reverse = idx

            self.selected_indices_nmf1 = []
            if idx_alpha is not None:
                self.selected_indices_nmf1.append(idx_alpha)
            if idx_reverse is not None and idx_reverse != idx_alpha:
                self.selected_indices_nmf1.append(idx_reverse)
        else:
            self.selected_indices_nmf1 = []

        self.display_nmf1_components()
        self.update_lasso_tab()

    def launch_nmf(self):
        import threading

        if not self.loaded_dir:
            self.log("Veuillez d'abord sélectionner un dossier .dat dans l'onglet Chargement.")
            return

        trim_start = simpledialog.askinteger(
            "Trim start",
            "Entrez le nombre de lignes à supprimer au début de chaque fichier :",
            minvalue=0,
            initialvalue=0,
            parent=self
        )
        if trim_start is None:
            return

        trim_end = simpledialog.askinteger(
            "Trim End",
            "Entrez le nombre de lignes à supprimer à la fin de chaque fichier :",
            minvalue=0,
            initialvalue=0,
            parent=self
        )
        if trim_end is None:
            return
    
        self.nmf_button.config(state='disabled')
        self.log("Début du traitement NMF...")

        dat_files = sorted([f for f in os.listdir(self.loaded_dir) if f.lower().endswith('.dat')])
        n_files = len(dat_files)
        total_steps = n_files * 2  
        self.progress['value'] = 0
        self.progress['maximum'] = total_steps

        def increment():
            self.progress.step(1)

        def nmf_thread():
            try:
                data_dict_all = {}
                all_q_mins = []
                all_q_maxs = []
                for fn in dat_files:
                    fpath = os.path.join(self.loaded_dir, fn)
                    df_ = read_and_clean_data(fpath, trim_start=trim_start, trim_end=trim_end)
                    if not df_.empty:
                        data_dict_all[fn] = df_
                        all_q_mins.append(df_['q'].min())
                        all_q_maxs.append(df_['q'].max())

                if not data_dict_all:
                    self.after(0, lambda: self.log("[ERREUR] Aucun fichier .dat valide."))
                    return

                all_common_q = np.linspace(max(all_q_mins), min(all_q_maxs), num=500)
                sample_names = sorted(data_dict_all.keys())

                X_list = []
                for nm in sample_names:
                    arr_ = interpolate_data(all_common_q, data_dict_all[nm])
                    arr_smooth = apply_gaussian_smoothing(arr_, sigma=10)
                    X_list.append(arr_smooth)

                if not hasattr(self, 'raw_data_all'):
                    self.raw_data_all = []
                    self.raw_data_q = []
                self.raw_data_all.clear()
                self.raw_data_q.clear()
                for nm in sample_names:
                    df = data_dict_all[nm]
                    self.raw_data_q.append(df['q'].values.copy())
                    self.raw_data_all.append(df['I(q)'].values.copy())

                X = np.ascontiguousarray(X_list)

                base_dir = os.path.join(self.loaded_dir, "First_NMF")
                alpha_dir = os.path.join(base_dir, "Alphabetical")
                reverse_dir = os.path.join(base_dir, "Reverse")
                os.makedirs(alpha_dir, exist_ok=True)
                os.makedirs(reverse_dir, exist_ok=True)

                comps_alpha, imgs_alpha, idx_alpha, \
                peaks_alpha, peaksq_alpha, files_alpha = compute_nmf_components(
                    data_dict_all,
                    sample_names,
                    "Alphabetical",
                    all_common_q,
                    alpha_dir,
                    initial_files=1,
                    step=1,
                    max_components=10,
                    sigma=10,
                    thresholds=(0.9997, 0.9995, 0.9994, 0.9991,
                                0.9989, 0.9987, 0.9985, 0.9983),
                    progress_callback=lambda: self.after(0, increment())
                )

                sample_names_rev = list(reversed(sample_names))
                comps_reverse, imgs_reverse, idx_rev, \
                peaks_rev, peaksq_rev, files_rev = compute_nmf_components(
                    data_dict_all,
                    sample_names_rev,
                    "Reverse",
                    all_common_q,
                    reverse_dir,
                    initial_files=1,
                    step=1,
                    max_components=10,
                    sigma=10,
                    thresholds=(0.9997, 0.9995, 0.9994, 0.9991,
                                0.9989, 0.9987, 0.9985, 0.9983),
                    progress_callback=lambda: self.after(0, increment())
                )

                all_peaks_q_merged = peaksq_alpha + peaksq_rev
                all_comps_merged = comps_alpha + comps_reverse
                all_imgs_merged = imgs_alpha + imgs_reverse

                combined_first = list(zip(all_peaks_q_merged, all_comps_merged, all_imgs_merged))
                combined_first.sort(key=lambda x: x[0])

                if combined_first:
                    sorted_peaks_q, sorted_comps, sorted_imgs = zip(*combined_first)
                else:
                    sorted_peaks_q, sorted_comps, sorted_imgs = [], [], []

                self.data_dict_all = data_dict_all
                self.all_common_q = all_common_q
                self.sample_names = sample_names
                self.X = X
                self.first_nmf_components = list(sorted_comps) + self.first_nmf_components
                self.first_nmf_images = list(sorted_imgs) + self.first_nmf_images
                self.first_nmf_labels = [f"q={pq:.4f}" for pq in sorted_peaks_q] + self.first_nmf_labels
                self.selected_indices_nmf1 = []

                self.after(0, self.display_nmf1_components)
            except Exception as e:
                self.after(0, lambda: self.log(f"Erreur NMF : {e}"))
            finally:
                self.after(0, lambda: self.nmf_button.config(state='normal'))

        threading.Thread(target=nmf_thread).start()

    def display_nmf1_components(self):
        for widget in self.nmf_components_frame.winfo_children():
            widget.destroy()

        imgs = self.first_nmf_images
        labels = self.first_nmf_labels
        selected = set(self.selected_indices_nmf1)

        def toggle_selection(idx, btn):
            if idx in self.selected_indices_nmf1:
                self.selected_indices_nmf1.remove(idx)
                img = imgs[idx].copy()
                img.thumbnail((100, 100))
                img_tk = ImageTk.PhotoImage(img)
                GLOBAL_IMAGE_REFERENCES.append(img_tk)
                btn.config(image=img_tk)
                btn.image = img_tk
            else:
                self.selected_indices_nmf1.append(idx)
                img = ImageOps.expand(imgs[idx], border=5, fill='red')
                img.thumbnail((100, 100))
                img_tk = ImageTk.PhotoImage(img)
                GLOBAL_IMAGE_REFERENCES.append(img_tk)
                btn.config(image=img_tk)
                btn.image = img_tk
            self.update_lasso_tab()

        cols = 5
        for i, pil_img in enumerate(imgs):
            row = i // cols
            col = i % cols

            img_copy = pil_img.copy()
            img_copy.thumbnail((100, 100))
            if i in selected:
                img_copy = ImageOps.expand(img_copy, border=5, fill='red')
            img_tk = ImageTk.PhotoImage(img_copy)
            GLOBAL_IMAGE_REFERENCES.append(img_tk)

            btn = tk.Button(self.nmf_components_frame, image=img_tk, bd=0)
            btn.configure(command=lambda idx=i, b=btn: toggle_selection(idx, b))
            btn.image = img_tk
            btn.grid(row=row * 2, column=col, padx=5, pady=5)

            lbl = tk.Label(
                self.nmf_components_frame,
                text=labels[i],
                font=("Arial", 8)
            )
            lbl.grid(row=row * 2 + 1, column=col, padx=5, pady=(0,10))

    def update_lasso_tab(self):
        for widget in self.lasso_results_frame.winfo_children():
            widget.destroy()

        indices = self.selected_indices_nmf1
        if not indices:
            lbl = tk.Label(
                self.lasso_results_frame,
                text="Aucune composante sélectionnée.",
                font=("Arial", 12)
            )
            lbl.pack(pady=20)
            return

        if self.X is not None:
            X = self.X
            sample_names = self.sample_names
            all_common_q = self.all_common_q

        else:
            if not self.first_nmf_components:
                lbl2 = tk.Label(
                    self.lasso_results_frame,
                    text="Aucune courbe disponible pour le Lasso (importez d’abord des CSV).",
                    font=("Arial", 12)
                )
                lbl2.pack(pady=20)
                return

            X = np.vstack(self.first_nmf_components)  
            sample_names = self.first_nmf_labels[:]   
            all_common_q = self.all_common_q  

        H_selected = np.array([self.first_nmf_components[i] for i in indices])
        comp_ids = [f"Comp_{i}" for i in indices]

        coefficients = []
        for i in range(X.shape[0]):
            lasso = Lasso(alpha=0.01, positive=True, max_iter=10000)
            weights = np.sqrt(np.abs(X[i]))
            basis_curves = H_selected.T
            Xw = basis_curves * weights[:, None]
            yw = X[i] * weights
            lasso.fit(Xw, yw)
            coefficients.append(lasso.coef_)
        coefficients = np.array(coefficients)

        reconstructed = coefficients @ H_selected
        self.reconstructed_X = reconstructed

        errors = np.sqrt(np.mean((X - reconstructed) ** 2, axis=1))
        residuals = X - reconstructed
        mse_per_channel = np.mean(residuals ** 2, axis=0)
        error_spectrum = np.sqrt(mse_per_channel)
        worst_indices = np.argsort(errors)[-5:]

        self.reconstruct_dict = {}
        for idx, nm in enumerate(sample_names):
            df_ = pd.DataFrame({
                'q': all_common_q,
                'I(q)': reconstructed[idx],
                'Sig(q)': np.zeros_like(reconstructed[idx])
            })
            self.reconstruct_dict[nm] = df_

        if coefficients.shape[1] >= 2:
            with np.errstate(invalid='ignore'):
                comp_corr = np.corrcoef(coefficients.T)
            comp_corr = np.nan_to_num(comp_corr)

            fig1 = plt.Figure(figsize=(6, 5))
            ax1 = fig1.add_subplot(111)
            sns.heatmap(comp_corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax1)
            ax1.set_title("Matrice de corrélation des composantes (Lasso)")
            canvas1 = FigureCanvasTkAgg(fig1, master=self.lasso_results_frame)
            canvas1.draw()
            canvas1.get_tk_widget().pack(
                fill='both', expand=True, padx=5, pady=5
            )

        fig2 = plt.Figure(figsize=(6, 5))
        ax2 = fig2.add_subplot(111)
        n_comp = coefficients.shape[1]
        for cidx in range(n_comp):
            ax2.plot(coefficients[:, cidx], label=comp_ids[cidx])
        ax2.set_title("Coefficients Lasso par échantillon")
        ax2.set_xlabel("Indice échantillon")
        ax2.set_ylabel("Valeur coefficient")
        ax2.legend(fontsize=8)
        canvas2 = FigureCanvasTkAgg(fig2, master=self.lasso_results_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(
            fill='both', expand=True, padx=5, pady=5
        )

        fig3 = plt.Figure(figsize=(6, 4))
        ax3 = fig3.add_subplot(111)
        ax3.plot(errors, marker='o', linestyle='-')
        ax3.set_title("Erreur Lasso par échantillon")
        ax3.set_xlabel("Indice échantillon")
        ax3.set_ylabel("Erreur L2")
        canvas3 = FigureCanvasTkAgg(fig3, master=self.lasso_results_frame)
        canvas3.draw()
        canvas3.get_tk_widget().pack(
            fill='both', expand=True, padx=5, pady=5
        )

        fig4 = plt.Figure(figsize=(6, 4))
        ax4 = fig4.add_subplot(111)
        for idx_ in worst_indices:
            ax4.plot(
                all_common_q,
                residuals[idx_],
                label=f"{sample_names[idx_]}, MSE={np.mean(residuals[idx_]**2):.4f}"
            )
        ax4.set_title("Résidus worst samples (Lasso)")
        ax4.set_xlabel("q")
        ax4.set_ylabel("Résidu")
        ax4.legend(fontsize=7)
        canvas4 = FigureCanvasTkAgg(fig4, master=self.lasso_results_frame)
        canvas4.draw()
        canvas4.get_tk_widget().pack(
            fill='both', expand=True, padx=5, pady=5
        )

        fig5 = plt.Figure(figsize=(6, 4))
        ax5 = fig5.add_subplot(111)
        ax5.plot(all_common_q, error_spectrum, marker='o')
        ax5.set_title("RMSE par canal (q)")
        ax5.set_xlabel("q")
        ax5.set_ylabel("RMSE")
        canvas5 = FigureCanvasTkAgg(fig5, master=self.lasso_results_frame)
        canvas5.draw()
        canvas5.get_tk_widget().pack(
            fill='both', expand=True, padx=5, pady=5
        )

        if self.loaded_dir:
            df_coeffs = pd.DataFrame(
                coefficients,
                columns=[f'Coeff_{ci}' for ci in comp_ids]
            )
            df_coeffs.insert(0, 'Sample_Name', sample_names)
            df_coeffs.to_csv(
                os.path.join(self.loaded_dir, 'lasso_coefficients_reconstructed.csv'),
                index=False
            )
            df_rms = pd.DataFrame({'q': all_common_q, 'RMSE': error_spectrum})
            df_rms.to_csv(os.path.join(self.loaded_dir, 'RMS_reconstructed.csv'), index=False)
    
    def optimize_lasso2(self):
        self.show_lasso2_optimized = True
        self.update_lasso2_tab()

    def transfer_to_nmf2(self):
        if not self.first_nmf_components:
            self.log("Aucune composante issue de la 1ʳᵉ NMF à transférer.")
            return

        self.second_nmf_components = list(self.first_nmf_components)
        self.second_nmf_images     = list(self.first_nmf_images)
        self.second_nmf_labels     = list(self.first_nmf_labels)
        self.selected_indices_nmf2 = []

        if self.X is not None:
            self.reconstructed_X = self.X.copy()
        else:
            self.reconstructed_X = None

        self.transferred = True

        self.notebook.select(self.tab_nmf2)
        self.display_nmf2_components()
        self.log("Composantes de la 1ʳᵉ NMF transférées vers l'onglet NMF Reconstruite & Lasso.")

    def populate_tab_nmf2(self):
        paned = ttk.Panedwindow(self.tab_nmf2, orient='horizontal')
        paned.pack(fill='both', expand=True, pady=10, padx=10)

        frame_nmf2_left = ttk.Frame(paned)
        paned.add(frame_nmf2_left, weight=1)

        param_frame2 = ttk.LabelFrame(frame_nmf2_left, text="Traitement NMF Reconstruite", padding=10)
        param_frame2.pack(fill='x', padx=5, pady=(0,10))

        for col in range(5):
            param_frame2.columnconfigure(col, weight=1)

        self.nmf2_button = ttk.Button(param_frame2, text="Lancer NMF Reconstruite", command=self.launch_nmf2)
        self.nmf2_button.grid(row=0, column=0, sticky='w', padx=(0,10))

        self.auto2_checkbox = ttk.Checkbutton(
            param_frame2,
            text="Auto-sélection composantes 0 (Alpha & Reverse)",
            variable=self.auto2_var,
            command=self.on_auto2_toggle
        )
        self.auto2_checkbox.grid(row=0, column=1, sticky='w', padx=(0,10))

        self.refine_button = ttk.Button(param_frame2, text="Raffiner Courbes", command=self.ask_and_refine)
        self.refine_button.grid(row=0, column=2, sticky='w', padx=(0,10))

        self.add_rms_button = ttk.Button(param_frame2, text="Ajouter RMS", command=self.add_rms_second)
        self.add_rms_button.grid(row=0, column=3, sticky='w', padx=(0,10))

        self.refine0_button = ttk.Button(param_frame2, text="Affiner 0%", command=self.ask_and_refine_zero)
        self.refine0_button.grid(row=1, column=1, sticky='w', padx=(0,10))

        self.reset_button = ttk.Button(param_frame2, text="Annuler Raffinement", command=self.reset_second)
        self.reset_button.grid(row=1, column=2, sticky='w', padx=(0,10))

        self.family2_btn = ttk.Button(param_frame2, text="Afficher/Ajouter Family Means", command=self.add_family_means_to_nmf2)
        self.family2_btn.grid(row=1, column=3, sticky='w', padx=(0,10))

        self.optimize_lasso2_btn = ttk.Button(param_frame2, text="Optimiser Lasso", command=self.optimize_lasso2)
        self.optimize_lasso2_btn.grid(row=1, column=4, sticky='w', padx=(0,10))

        self.btn_show_only_family_eans = ttk.Button(param_frame2, text="Afficher seulement Family Means", command=self.show_only_family_means_nmf2)
        self.btn_show_only_family_eans.grid(row=2, column=1, sticky='w', padx=(0,10))

        self.btn_restore_nmf2=ttk.Button(param_frame2, text="Restaurer Composantes", command=self.restore_all_nmf2_components)
        self.btn_restore_nmf2.grid(row=2, column=2, sticky='w', padx=(0,10))

        container2_left = ttk.Frame(frame_nmf2_left)
        container2_left.pack(fill='both', expand=True, padx=5, pady=5)

        self.nmf2_scroll_canvas = tk.Canvas(container2_left)
        self.nmf2_scroll_canvas.pack(side='left', fill='both', expand=True)

        v_scrollbar2_left = ttk.Scrollbar(container2_left, orient='vertical', command=self.nmf2_scroll_canvas.yview)
        v_scrollbar2_left.pack(side='right', fill='y')
        self.nmf2_scroll_canvas.configure(yscrollcommand=v_scrollbar2_left.set)

        self.nmf2_components_frame = ttk.Frame(self.nmf2_scroll_canvas)
        self.nmf2_components_window = self.nmf2_scroll_canvas.create_window((0, 0), window=self.nmf2_components_frame, anchor='nw')
        self.nmf2_components_frame.bind(
            "<Configure>",
            lambda e: self.nmf2_scroll_canvas.configure(scrollregion=self.nmf2_scroll_canvas.bbox("all"))
        )

        frame_nmf2_right = ttk.Frame(paned)
        paned.add(frame_nmf2_right, weight=1)

        lbl2 = ttk.Label(frame_nmf2_right, text="Résultats Lasso Reconstruite :", font=("Arial", 12, "bold"))
        lbl2.pack(anchor='w', padx=5, pady=(0,5))

        stats_frame = ttk.Frame(frame_nmf2_right, padding=(5, 5))
        stats_frame.pack(fill='x', padx=5, pady=(0,10))

        ttk.Label(stats_frame, text="Moyenne RMS :").grid(row=0, column=0, sticky='w', padx=(0,10))
        ttk.Label(stats_frame, textvariable=self.avg_rms_var).grid(row=0, column=1, sticky='w')

        ttk.Label(stats_frame, text="Erreur Lasso Initiale :").grid(row=1, column=0, sticky='w', padx=(0,10))
        ttk.Label(stats_frame, textvariable=self.initial_lasso_error_var).grid(row=1, column=1, sticky='w')

        ttk.Label(stats_frame, text="Erreur Lasso après optimisation :").grid(row=2, column=0, sticky='w', padx=(0,10))
        ttk.Label(stats_frame, textvariable=self.optimized_lasso_error_var).grid(row=2, column=1, sticky='w')

        ttk.Label(stats_frame, text="Erreur totale :").grid(row=3, column=0, sticky='w', padx=(0,10))
        ttk.Label(stats_frame, textvariable=self.total_error_var).grid(row=3, column=1, sticky='w')

        container2_right = ttk.Frame(frame_nmf2_right)
        container2_right.pack(fill='both', expand=True, padx=5, pady=5)

        self.components_plot_frame = ttk.Frame(container2_right)
        self.components_plot_frame.pack(fill='x', padx=5, pady=5)

        self.lasso2_scroll_canvas = tk.Canvas(container2_right)
        self.lasso2_scroll_canvas.pack(side='left', fill='both', expand=True)

        v_scrollbar2_right = ttk.Scrollbar(container2_right, orient='vertical', command=self.lasso2_scroll_canvas.yview)
        v_scrollbar2_right.pack(side='right', fill='y')
        self.lasso2_scroll_canvas.configure(yscrollcommand=v_scrollbar2_right.set)

        self.lasso2_results_frame = ttk.Frame(self.lasso2_scroll_canvas)
        self.lasso2_results_window = self.lasso2_scroll_canvas.create_window((0, 0), window=self.lasso2_results_frame, anchor='nw')
        self.lasso2_results_frame.bind(
            "<Configure>",
            lambda e: self.lasso2_scroll_canvas.configure(scrollregion=self.lasso2_scroll_canvas.bbox("all"))
        )

        self.progress2 = ttk.Progressbar(self.tab_nmf2, orient='horizontal', mode='determinate')
        self.progress2.pack(fill='x', padx=10, pady=(0,5))

    def show_only_family_means_nmf2(self):
        if not hasattr(self, "family_mean_indices_nmf2") or not self.family_mean_indices_nmf2:
            self.log("Aucune family mean détectée. Cliquez d'abord sur 'Afficher/Ajouter Family Means'.")
            return

        if not hasattr(self, "_backup_second_nmf_components"):
            self._backup_second_nmf_components = (self.second_nmf_components[:],
                                              self.second_nmf_images[:],
                                              self.second_nmf_labels[:])

        indices = self.family_mean_indices_nmf2
        self.second_nmf_components = [self.second_nmf_components[i] for i in indices]
        self.second_nmf_images     = [self.second_nmf_images[i] for i in indices]
        self.second_nmf_labels     = [self.second_nmf_labels[i] for i in indices]
        self.display_nmf2_components()
        self.update_lasso2_tab()
        self.log("Affichage limité aux family means pour la seconde NMF.")

    def restore_all_nmf2_components(self):
        if hasattr(self, "_backup_second_nmf_components"):
            self.second_nmf_components, self.second_nmf_images, self.second_nmf_labels = [
                l[:] for l in self._backup_second_nmf_components
            ]
        self.selected_indices_nmf2 = []
        self.display_nmf2_components()
        self.update_lasso2_tab()
        self.log("Toutes les composantes NMF2 restaurées.")


    def _save_nmf2_history(self):
        state = (
            [c.copy() for c in self.second_nmf_components],
            [img.copy() for img in self.second_nmf_images],
            list(self.second_nmf_labels)
        )
        current_idx = getattr(self, "_nmf2_history_index", None)
        if current_idx is not None and current_idx < len(self.second_nmf_history) - 1:
            self.second_nmf_history = self.second_nmf_history[:current_idx+1]
        self.second_nmf_history.append(state)
        self._nmf2_history_index = len(self.second_nmf_history) - 1

    def ask_and_refine(self):
        if not self.selected_indices_nmf2:
            self.log("Veuillez sélectionner au moins une composante 2ᵉ NMF avant de raffiner.")
            return

        H_all       = np.array(self.second_nmf_components)  
        all_ids     = list(self.second_nmf_labels)         
        X           = self.X.copy()                        
        q_grid      = self.all_common_q                     
        sample_names = self.sample_names[:]                 

        subset_idx  = self.selected_indices_nmf2[:]         
        H_subset    = [H_all[i] for i in subset_idx]       
        id_subset   = [all_ids[i] for i in subset_idx]    

        images_for_ref = []
        for comp in H_subset:
            plt.figure(figsize=(4,3))
            plt.plot(q_grid, comp, linewidth=1.5)
            plt.tight_layout()
            buf = BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            img = Image.open(buf)
            images_for_ref.append(img)

        ref_local = display_images_and_select_refinement(
            parent         = self,
            images_for_ref = images_for_ref,
            current_ids    = id_subset,
            title          = "Sélectionnez des composantes à raffiner (2ᵉ NMF)",
            labels         = id_subset
        )

        if not ref_local:
            self.log("[INFO] Aucune composante sélectionnée pour le raffinement.")
            return

        ref_global = [ subset_idx[i] for i in ref_local ]

        user_percentage = simpledialog.askinteger(
            "Puissance du raffinement",
            "Entrez le pourcentage de raffinement (0–100) :",
            minvalue=0, maxvalue=100,
            parent=self
        )
        if user_percentage is None:
            return
        alpha = user_percentage / 100.0

        H_current = H_all.copy()  

        for global_i in ref_global:

            old_curve = H_current[global_i].copy()

            comps_sel = []
            for gi in ref_global:
                comps_sel.append(H_current[gi].copy())
            H_init_ref = np.array(comps_sel)
            H_init_ref = np.maximum(H_init_ref, 0)

            W_init_ref = []
            for row_ in X:
                cfs, _ = nnls(H_init_ref.T, row_)
                W_init_ref.append(cfs)
            W_init_ref = np.array(W_init_ref)

            W_o, H_o, _ = non_negative_factorization(
                X,
                W=W_init_ref,
                H=H_init_ref,
                n_components=H_init_ref.shape[0],
                init='custom',
                update_H=True,
                solver='cd',
                beta_loss='frobenius',
                max_iter=10000,
                tol=1e-3,
                random_state=42,
                alpha_W=1e-5,
                alpha_H=1e-5,
                l1_ratio=0
            )
            H_o = np.maximum(H_o, 0)

            local_i = ref_global.index(global_i)
            new_curve = H_o[local_i].copy()

            final_curve = old_curve * (1 - alpha) + new_curve * alpha

            comp_name = all_ids[global_i]
            plt.figure(figsize=(5,4))
            plt.plot(q_grid, old_curve,   linestyle='--', label=f"Before {comp_name}")
            plt.plot(q_grid, new_curve,   linewidth=1.5,  label=f"Brut {comp_name}")
            plt.plot(q_grid, final_curve, linestyle=':',   label=f"Refined {user_percentage}%")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.show()

            err_before = re_run_nnls_for_rms(X, H_current, sample_names, q_grid)

            H_test = H_current.copy()
            H_test[global_i] = final_curve.copy()

            err_after = re_run_nnls_for_rms(X, H_test, sample_names, q_grid)

            if self.error_spectrum2 is not None:
                max_rmse = np.max(self.error_spectrum2)
                idx_max  = np.argmax(self.error_spectrum2)
                q_at_max = q_grid[idx_max]
            else:
                max_rmse = 0.0
                q_at_max  = 0.0

            user_validates = messagebox.askyesno(
                "Validation raffinage",
                f"Composante : {comp_name}\n\n"
                f"Erreur totale AVANT : {err_before:.4f}\n"
                f"Erreur totale APRÈS : {err_after:.4f}\n"
                f"RMSE max = {max_rmse:.4f} à q = {q_at_max:.4f}\n\n"
                "Valider ce raffinement ?"
            )
            if user_validates:
                H_current[global_i] = final_curve.copy()

        self.second_nmf_components = [h.copy() for h in H_current]

        if self.reconstructed_X is not None:
            self.H2_initial = self.reconstructed_X.copy()
        else:
            self.H2_initial = None

        self._save_nmf2_history()
        self.display_nmf2_components()
        self.update_lasso2_tab()

        self.log("[INFO] Raffinement terminé et second NMF mis à jour.")

    def ask_and_refine_zero(self):
        if not self.selected_indices_nmf2:
            self.log("Veuillez sélectionner au moins une composante 2ᵉ NMF pour Raffiner 0 %.")
            return

        H_all      = np.array(self.second_nmf_components)  
        all_ids    = list(self.second_nmf_labels)          
        sel_idx    = self.selected_indices_nmf2[:]         

        H_sel      = H_all[sel_idx, :].copy()             
        sel_ids    = [all_ids[i] for i in sel_idx]         

        X          = self.X.copy()                        
        q_grid     = self.all_common_q                    
        sample_names = self.sample_names[:]                

        H_init_ref = np.maximum(H_sel, 0)  

        W_init_ref = []
        for row_ in X:
            coefs, _ = nnls(H_init_ref.T, row_)
            W_init_ref.append(coefs)
        W_init_ref = np.array(W_init_ref)  

        W_o, H_o, _ = non_negative_factorization(
            X,
            W=W_init_ref,
            H=H_init_ref,
            n_components=H_init_ref.shape[0],
            init='custom',
            update_H=True,
            solver='cd',
            beta_loss='frobenius',
            max_iter=10000,
            tol=1e-3,
            random_state=42,
            alpha_W=1e-5,
            alpha_H=1e-5,
            l1_ratio=0
        )
        H_o = np.maximum(H_o, 0)  

        for local_i, global_i in enumerate(sel_idx):
            comp_name   = all_ids[global_i]
            old_curve   = H_all[global_i]
            new_curve   = H_o[local_i]

            plt.figure(figsize=(5, 4))
            plt.plot(q_grid, old_curve, linestyle='--', linewidth=1.5, label=f"Avant {comp_name}")
            plt.plot(q_grid, new_curve, linewidth=1.5, label=f"Raffiné {comp_name}")
            plt.title(f"Comparatif Raffinement 0% – Composante {comp_name}")
            plt.xlabel("q")
            plt.ylabel("Intensité")
            plt.legend()
            plt.tight_layout()
            plt.show()

        if self.reconstructed_X is not None:
            self.H2_initial = self.reconstructed_X.copy()
        else:
            self.H2_initial = None

        self.display_nmf2_components()

        self.log("[INFO] Raffinement 0 % appliqué sur les composantes sélectionnées (aucune modification).")

    def launch_nmf2(self):
        import threading

        self.transferred = False

        if not self.loaded_dir:
            self.log("Veuillez d'abord sélectionner un dossier .dat dans l'onglet Chargement.")
            return

        if not self.first_nmf_components:
            self.log("Veuillez d'abord exécuter la 1ère NMF et sélectionner au moins une composante.")
            return

        if not self.selected_indices_nmf1:
            self.log("Veuillez sélectionner au moins une composante de la 1ère NMF.")
            return

        self.nmf2_button.config(state='disabled')
        self.log("Début du traitement NMF Reconstruite...")

        n_samples = len(self.sample_names)
        total_steps2 = n_samples * 2  
        self.progress2['value'] = 0
        self.progress2['maximum'] = total_steps2

        def increment2():
            self.progress2.step(1)

        def nmf2_thread():
            try:
                H_sel = np.array([self.first_nmf_components[i] for i in self.selected_indices_nmf1])
                X = self.X
                coefficients = []
                for i in range(len(self.sample_names)):
                    lasso = Lasso(alpha=0.01, positive=True, max_iter=10000)
                    weights = np.sqrt(np.abs(X[i]))
                    basis_curves = H_sel.T
                    Xw = basis_curves * weights[:, None]
                    yw = X[i] * weights
                    lasso.fit(Xw, yw)
                    coefficients.append(lasso.coef_)
                coefficients = np.array(coefficients)
                reconstructed = coefficients @ H_sel
                self.reconstructed_X = reconstructed
                self.H2_initial = reconstructed.copy()

                self.reconstruct_dict = {}
                for idx, nm in enumerate(self.sample_names):
                    df_ = pd.DataFrame({
                        'q': self.all_common_q,
                        'I(q)': reconstructed[idx],
                        'Sig(q)': np.zeros_like(reconstructed[idx])
                    })
                    self.reconstruct_dict[nm] = df_

                base_dir = os.path.join(self.loaded_dir, "Second_NMF")
                alpha2_dir = os.path.join(base_dir, "Reconstructed_Alphabetical")
                reverse2_dir = os.path.join(base_dir, "Reconstructed_Reverse")
                os.makedirs(alpha2_dir, exist_ok=True)
                os.makedirs(reverse2_dir, exist_ok=True)

                rec_comps_alpha, rec_imgs_alpha, rec_idx_alpha, rec_peaks_alpha, rec_peaks_q_alpha, rec_files_alpha = compute_nmf_components(
                    self.reconstruct_dict,
                    self.sample_names,
                    "Reconstructed_Alphabetical",
                    self.all_common_q,
                    alpha2_dir,
                    initial_files=1,
                    step=1,
                    max_components=10,
                    sigma=10,
                    thresholds=(0.9997, 0.9995, 0.9994, 0.9991,
                                0.9989, 0.9987, 0.9985, 0.9983),
                    progress_callback=lambda: self.after(0, increment2())
                )

                sample_names_rev = list(reversed(self.sample_names))
                rec_comps_reverse, rec_imgs_reverse, rec_idx_reverse, rec_peaks_reverse, rec_peaks_q_reverse, rec_files_reverse = compute_nmf_components(
                    self.reconstruct_dict,
                    sample_names_rev,
                    "Reconstructed_Reverse",
                    self.all_common_q,
                    reverse2_dir,
                    initial_files=1,
                    step=1,
                    max_components=10,
                    sigma=10,
                    thresholds=(0.9997, 0.9995, 0.9994, 0.9991,
                                0.9989, 0.9987, 0.9985, 0.9983),
                    progress_callback=lambda: self.after(0, increment2())
                )

                all_peaks_rec_merged = rec_peaks_q_alpha + rec_peaks_q_reverse
                all_comps_rec_merged = rec_comps_alpha + rec_comps_reverse
                all_imgs_rec_merged = rec_imgs_alpha + rec_imgs_reverse

                combined_second = list(zip(all_peaks_rec_merged, all_comps_rec_merged, all_imgs_rec_merged))
                combined_second.sort(key=lambda x: x[0])

                if combined_second:
                    sorted_rec_peaks_q, sorted_rec_comps, sorted_rec_imgs = zip(*combined_second)
                else:
                    sorted_rec_peaks_q, sorted_rec_comps, sorted_rec_imgs = [], [], []

                self.second_nmf_components = list(sorted_rec_comps)
                self.second_nmf_images = list(sorted_rec_imgs)
                self.second_nmf_labels = [f"q={pq:.4f}" for pq in sorted_rec_peaks_q]
                self.selected_indices_nmf2 = []

                self.H2_initial = self.reconstructed_X.copy()
                self.error_spectrum2 = None
                self.after(0, self.display_nmf2_components)

            except Exception as e:
                self.after(0, lambda: self.log(f"Erreur 2ᵉ NMF : {e}"))
            finally:
                self.after(0, lambda: self.nmf2_button.config(state='normal'))

        threading.Thread(target=nmf2_thread).start()

    def display_nmf2_components(self):
        for widget in self.nmf2_components_frame.winfo_children():
            widget.destroy()

        imgs = self.second_nmf_images
        labels = self.second_nmf_labels
        selected = set(self.selected_indices_nmf2)

        def toggle_selection2(idx, btn):
            if idx in self.selected_indices_nmf2:
                self.selected_indices_nmf2.remove(idx)
                img = imgs[idx].copy()
                img.thumbnail((100, 100))
                img_tk = ImageTk.PhotoImage(img)
                GLOBAL_IMAGE_REFERENCES.append(img_tk)
                btn.config(image=img_tk)
                btn.image = img_tk
            else:
                self.selected_indices_nmf2.append(idx)
                img = ImageOps.expand(imgs[idx], border=5, fill='red')
                img.thumbnail((100, 100))
                img_tk = ImageTk.PhotoImage(img)
                GLOBAL_IMAGE_REFERENCES.append(img_tk)
                btn.config(image=img_tk)
                btn.image = img_tk

            self.update_lasso2_tab()

        cols = 5
        for i, pil_img in enumerate(imgs):
            row = i // cols
            col = i % cols

            img_copy = pil_img.copy()
            img_copy.thumbnail((100, 100))
            if i in selected:
                img_copy = ImageOps.expand(img_copy, border=5, fill='red')
            img_tk = ImageTk.PhotoImage(img_copy)
            GLOBAL_IMAGE_REFERENCES.append(img_tk)

            btn = tk.Button(self.nmf2_components_frame, image=img_tk, bd=0)
            btn.configure(command=lambda idx=i, b=btn: toggle_selection2(idx, b))
            btn.image = img_tk
            btn.grid(row=row * 2, column=col, padx=5, pady=5)

            lbl = tk.Label(self.nmf2_components_frame, text=labels[i], font=("Arial", 8))
            lbl.grid(row=row * 2 + 1, column=col, padx=5, pady=(0,10))

    def on_auto2_toggle(self):
        idx_alpha = None
        idx_reverse = None

        if self.transferred:
            for idx, comp in enumerate(self.second_nmf_components):
                if GLOBAL_FIRST_ALPHA_COMP is not None and np.allclose(comp, GLOBAL_FIRST_ALPHA_COMP):
                    idx_alpha=idx
                if GLOBAL_FIRST_REVERSE_COMP is not None and np.allclose(comp, GLOBAL_FIRST_REVERSE_COMP):
                    idx_reverse=idx
        
        else:
            for idx, comp in enumerate(self.second_nmf_components):
                if GLOBAL_FIRST_ALPHA_COMP_2 is not None and np.allclose(comp, GLOBAL_FIRST_ALPHA_COMP_2):
                    idx_alpha = idx
                if GLOBAL_FIRST_REVERSE_COMP_2 is not None and np.allclose(comp, GLOBAL_FIRST_REVERSE_COMP_2):
                    idx_reverse = idx
        
        self.selected_indices_nmf2 = []
        if idx_alpha is not None:
            self.selected_indices_nmf2.append(idx_alpha)
        if idx_reverse is not None and idx_reverse != idx_alpha:
            self.selected_indices_nmf2.append(idx_reverse)

        self.display_nmf2_components()
        self.update_lasso2_tab()

    def update_lasso2_tab(self):
        for widget in self.components_plot_frame.winfo_children():
            widget.destroy()
        for widget in self.lasso2_results_frame.winfo_children():
            widget.destroy()

        indices = self.selected_indices_nmf2
        if not indices and not self.include_rms2:
            lbl = tk.Label(self.lasso2_results_frame, text="Aucune composante sélectionnée.", font=("Arial", 12))
            lbl.pack(pady=20)
            return

        fig_comp = plt.Figure(figsize=(6,3))
        ax_comp = fig_comp.add_subplot(111)
        for idx in indices:
            ax_comp.plot(self.all_common_q, self.second_nmf_components[idx], label=f"Comp {idx}")
        if self.include_rms2 and self.error_spectrum2 is not None:
            ax_comp.plot(self.all_common_q, self.error_spectrum2, '--', label='RMS', linewidth=1.5)
        ax_comp.set_title("Composantes 2ᵉ NMF Sélectionnées" + (" + RMS" if self.include_rms2 else ""))
        ax_comp.set_xlabel("q")
        ax_comp.set_ylabel("Intensité")
        ax_comp.legend(fontsize=8)
        canvas_comp = FigureCanvasTkAgg(fig_comp, master=self.components_plot_frame)
        canvas_comp.draw()
        canvas_comp.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)

        if not indices:
            return

        comps_list = [self.second_nmf_components[i] for i in indices]
        if self.include_rms2 and self.error_spectrum2 is not None:
            comps_list.append(self.error_spectrum2)
        H_selected = np.array(comps_list)

        X2 = self.reconstructed_X
        sample_names = self.sample_names
        all_common_q = self.all_common_q
        comp_ids = [f"Comp2_{i}" for i in indices]
        if self.include_rms2 and self.error_spectrum2 is not None:
            comp_ids.append("RMS")

        coeffs = []
        for i in range(X2.shape[0]):
            lasso = Lasso(alpha=0.01, positive=True, max_iter=10000)
            weights = np.sqrt(np.abs(X2[i]))
            basis_curves = H_selected.T
            Xw = basis_curves * weights[:, None]
            yw = X2[i] * weights
            lasso.fit(Xw, yw)
            coeffs.append(lasso.coef_)
        coeffs = np.array(coeffs)

        reconstructed2 = coeffs @ H_selected
        errors = np.sqrt(np.mean((X2 - reconstructed2) ** 2, axis=1))
        residuals = X2 - reconstructed2
        mse_per_channel = np.mean(residuals ** 2, axis=0)
        error_spectrum = np.sqrt(mse_per_channel)
        worst_indices = np.argsort(errors)[-5:]

        self.error_spectrum2 = error_spectrum

        avg_rms = np.mean(error_spectrum)
        self.avg_rms_var.set(f"{avg_rms:.4e}")
        initial_lasso_error = np.mean(errors)
        self.initial_lasso_error_var.set(f"{initial_lasso_error:.4e}")
        total_error = np.linalg.norm(X2 - reconstructed2)
        self.total_error_var.set(f"{total_error:.4e}")

        if coeffs.shape[1] >= 2:
            with np.errstate(invalid='ignore'):
                comp_corr = np.corrcoef(coeffs.T)
            comp_corr = np.nan_to_num(comp_corr)

            fig1 = plt.Figure(figsize=(6, 5))
            ax1 = fig1.add_subplot(111)
            sns.heatmap(comp_corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax1)
            ax1.set_title("Matrice de corrélation (2ᵉ NMF Lasso)")
            canvas1 = FigureCanvasTkAgg(fig1, master=self.lasso2_results_frame)
            canvas1.draw()
            canvas1.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)

        fig2 = plt.Figure(figsize=(6, 5))
        ax2 = fig2.add_subplot(111)
        n_comp = coeffs.shape[1]
        for cidx in range(n_comp):
            ax2.plot(coeffs[:, cidx], label=comp_ids[cidx])
        ax2.set_title("Coefficients Lasso par échantillon (2ᵉ NMF)")
        ax2.set_xlabel("Indice échantillon")
        ax2.set_ylabel("Valeur coefficient")
        ax2.legend(fontsize=8)
        canvas2 = FigureCanvasTkAgg(fig2, master=self.lasso2_results_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)

        fig3 = plt.Figure(figsize=(6, 4))
        ax3 = fig3.add_subplot(111)
        ax3.plot(errors, marker='o', linestyle='-')
        ax3.set_title("Erreur Lasso par échantillon (2ᵉ NMF)")
        ax3.set_xlabel("Indice échantillon")
        ax3.set_ylabel("Erreur L2")
        canvas3 = FigureCanvasTkAgg(fig3, master=self.lasso2_results_frame)
        canvas3.draw()
        canvas3.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)

        fig4 = plt.Figure(figsize=(6, 4))
        ax4 = fig4.add_subplot(111)
        for idx_ in worst_indices:
            ax4.plot(
                all_common_q,
                residuals[idx_],
                label=f"{sample_names[idx_]}, MSE={np.mean(residuals[idx_]**2):.4f}"
            )
        ax4.set_title("Résidus worst samples (2ᵉ NMF Lasso)")
        ax4.set_xlabel("q")
        ax4.set_ylabel("Résidu")
        ax4.legend(fontsize=7)
        canvas4 = FigureCanvasTkAgg(fig4, master=self.lasso2_results_frame)
        canvas4.draw()
        canvas4.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)

        fig5 = plt.Figure(figsize=(6, 4))
        ax5 = fig5.add_subplot(111)
        ax5.plot(all_common_q, error_spectrum, marker='o')
        ax5.set_title("RMSE par canal (q) (2ᵉ NMF)")
        ax5.set_xlabel("q")
        ax5.set_ylabel("RMSE")
        canvas5 = FigureCanvasTkAgg(fig5, master=self.lasso2_results_frame)
        canvas5.draw()
        canvas5.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)

        component_ids_opt = [f"CompOpt_{i}" for i in indices]
        component_colors_opt = None
        dynamic_window_ref = None

        if self.show_lasso2_optimized : 
            opt_dir = Path(self.loaded_dir) / "Optimized_Lasso"
            coeffs_opt, recon_opt, errors_opt, rms_opt = optimize_lasso(
                X2,
                H_selected,
                sample_names,
                all_common_q,
                str(opt_dir),
                suffix="2ndNMF_lasso_optimized",
                component_ids=component_ids_opt,
                component_colors=component_colors_opt,
                parent=self.lasso2_results_frame,
                dynamic_window=dynamic_window_ref
            )
        else:
            pass

        if errors_opt is not None:
            avg_opt = np.mean(errors_opt)
            self.optimized_lasso_error_var.set(f"{avg_opt:.4e}")
        else:
            self.optimized_lasso_error_var.set("N/A")

        fig_compare = plt.figure(figsize=(6,5))
        ax_cmp = fig_compare.add_subplot(111)

        n_comp=coeffs.shape[1]
        for cidx in range(n_comp):
            ax_cmp.plot(
                coeffs[:, cidx],
                linestyle='--',
                label=f"Base {comp_ids[cidx]}"
            )
            ax_cmp.plot(
                coeffs_opt[:, cidx],
                linestyle='-',
                label = f"Opt {component_ids_opt[cidx]}"
            )
        
        ax_cmp.set_title("Comparaison coefficients Lasso : Base vs Optimisés")
        ax_cmp.set_xlabel("Indice échantillon")
        ax_cmp.set_ylabel("Valeur Coefficient")
        ax_cmp.legend(fontsize=7)

        canvas_cmp = FigureCanvasTkAgg(fig_compare, master=self.lasso2_results_frame)
        canvas_cmp.draw()
        canvas_cmp.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)

        moyenne_errors_opt = np.mean(errors_opt) if errors_opt is not None else 0.0
        self.optimized_lasso_error_var.set(f"{moyenne_errors_opt:.4e}")

        X = self.X.copy() 
        all_common_q = self.all_common_q
        sample_names = self.sample_names
        H_all = np.array(self.second_nmf_components)
        H = H_all.copy()  

        X_reconstructed = []
        for i in range(len(sample_names)):
            coefs, _ = nnls(H.T, X[i])
            recon = H.T @ coefs
            X_reconstructed.append(recon)
        X_reconstructed = np.array(X_reconstructed)

        reconstruction_optimized_explicit = np.array(self.reconstruction_finale)  

        i = 0
        q_raw = self.q_raw_data[i]
        I_raw = self.intensite_data[i]

        f_interp = interp1d(q_raw, I_raw, bounds_error=False, fill_value="extrapolate")
        raw_data_interpolated = f_interp(all_common_q)

        rec_data = X_reconstructed[i]
        rec_data_optimized = reconstruction_optimized_explicit[i]

        for i in range(len(self.sample_names)):
            q_raw = self.raw_data_q[i]
            I_raw = self.raw_data_all[i]
            f_interp = interp1d(q_raw, I_raw, bounds_error=False, fill_value="extrapolate")
            raw_data_interpolated = f_interp(self.all_common_q)
            rec_data = X_reconstructed[i]
            rec_data_optimized = reconstruction_optimized_explicit[i]

            df_err_before = pd.DataFrame({
                'q': self.all_common_q,
                'raw data': raw_data_interpolated,
                'reconstructed data': rec_data,
                'error': rec_data - raw_data_interpolated
            })

            df_err_after = pd.DataFrame({
                'q': self.all_common_q,
                'raw data': raw_data_interpolated,
                'reconstructed data (optimized)': rec_data_optimized,
                'error (optimized)': rec_data_optimized - raw_data_interpolated
            })

            df_err_before.to_csv(os.path.join(self.loaded_dir, f"erreur_avant_optim_{self.sample_names[i]}.csv"), index=False)
            df_err_after.to_csv(os.path.join(self.loaded_dir, f"erreur_apres_optim_{self.sample_names[i]}.csv"), index=False)

        self.lasso2_coeffs_optimized = coeffs_opt
        self.lasso2_reconstructed_optimized = recon_opt
        self.lasso2_errors_optimized = errors_opt
        self.lasso2_rms_optimized = rms_opt

    def add_rms_second(self):
        if self.error_spectrum2 is None:
            self.log("Veuillez d'abord lancer un Lasso ou un raffinement avant d'ajouter le RMS.")
            return

        fig = plt.Figure(figsize=(4, 3))
        ax = fig.add_subplot(111)
        ax.plot(self.all_common_q, self.error_spectrum2, color='blue')
        ax.set_title("RMS figé")
        buf = BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        im_rms = Image.open(buf)

        self.second_nmf_components.append(self.error_spectrum2.copy())
        self.second_nmf_images.append(im_rms.copy())
        self.second_nmf_labels.append("RMS figé")

        new_idx = len(self.second_nmf_components) - 1
        self.selected_indices_nmf2.append(new_idx)

        self.display_nmf2_components()
        self.update_lasso2_tab()

        self.log("RMS figé ajouté aux composantes de la 2ᵉ NMF.")

    def reset_second(self):
        if not self.second_nmf_history:
            self.log("Aucun historique de raffinement trouvé.")
            return

        max_step = len(self.second_nmf_history) - 1
        step = simpledialog.askinteger(
            "Annuler Raffinement",
            f"À quelle étape du raffinement voulez-vous revenir ? (0 = initial, {max_step} = dernier raffinement)",
            minvalue=0, maxvalue=max_step, initialvalue=max_step,
            parent=self
        )
        if step is None:
            self.log("[INFO] Annulation de l'opération d'annulation")
            return

        comps, imgs, labels = self.second_nmf_history[step]
        self.second_nmf_components = [c.copy() for c in comps]
        self.second_nmf_images = [img.copy() for img in imgs]
        self.second_nmf_labels = list(labels)
        self._nmf2_history_index = step

        self.display_nmf2_components()
        self.update_lasso2_tab()
        self.log(f"Revenu à l'étape de raffinement #{step}.")

if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
