import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import io
import xlsxwriter

# Mappings and color rules.
account_type_to_process_group = {
    "accounts payable": "PTP",
    "inventory": "Inventory",
    "order to cash": "OTC",
    "payroll": "Payroll",
    "financial close": "Financial Close",
    "fixed assets": "Fixed Assets",
    "tax": "Tax",
    "treasury": "Treasury",
    "real estate": "RE",
    "business combinations": "Business Combinations"
}

color_rules = {
    ("In Scope", "Yes"): '#C6EFCE', # Light Green - In Scope & Mapped (Good)
    ("In Scope", "No"): '#FF9999', # Red - In Scope & Not Mapped (⚠️ Review)
    ("Out of Scope", "Yes"): '#FFEB9C', # Light Yellow - Out of Scope & Mapped (💡 Info: Check if it should be In Scope)
    ("Out of Scope", "No"): '#D9D9D9', # Light Grey - Out of Scope & Not Mapped (Expected)
    "⚠️ Review: In Scope & Non-Key": '#F4B084', # Orange - In Scope but Non-Key (⚠️ Review)
    "⚠️ Review: Out of Scope & Key": '#D9D2E9', # Purple - Out of Scope but Key (⚠️ Review)
    "⚠️ Review: In Scope & not Mapped in RCM": '#FF6347' # Tomato Red for unmapped In-Scope
}

def clean_number(value):
    """Cleans and converts a string value to a float, handling commas and percentage signs."""
    if isinstance(value, str):
        value = value.replace(',', '').replace('%', '').strip()
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

def map_control_id_to_process(control_id):
    """Maps a Control ID to a specific process group based on its numeric prefix."""
    try:
        id_str = str(control_id)
        if not id_str or id_str.lower() == 'nan':
            return "Unknown"

        id_fragment = id_str.split("-")[-1].strip()
        numeric_str = ''.join([ch for ch in id_fragment if ch.isdigit() or ch == '.'])
        
        if not numeric_str:
            return "Unknown"

        numeric_part = float(numeric_str[:6])

        if 1.0 <= numeric_part < 2.0: return "PTP"
        elif 2.0 <= numeric_part < 3.0: return "Payroll"
        elif 3.0 <= numeric_part < 4.0: return "OTC"
        elif 4.0 <= numeric_part < 5.0: return "Inventory"
        elif 5.0 <= numeric_part < 6.0: return "Financial Close"
        elif 6.0 <= numeric_part < 7.0: return "Fixed Assets"
        elif 7.0 <= numeric_part < 8.0: return "Treasury"
        elif 8.0 <= numeric_part < 9.0: return "Tax"
        elif 9.0 <= numeric_part < 10.0: return "RE"
        elif 10.0 <= numeric_part < 11.0: return "Business Combinations"
        else: return "Other"
    except (IndexError, ValueError, TypeError):
        return "Unknown"

def drop_blank_rows(df):
    """Removes rows that are entirely empty (all NaN) or contain only whitespace strings."""
    if df.empty:
        return df
    df_cleaned = df.dropna(how='all')
    if not df_cleaned.empty:
        df_cleaned = df_cleaned.loc[~(df_cleaned.apply(lambda x: x.astype(str).str.strip() == "").all(axis=1))]
    return df_cleaned


class SOXApp:
    def __init__(self, master):
        self.master = master
        master.title("SOX Audit Automation GUI")
        master.resizable(True, True)

        self.main_frame = ttk.Frame(master)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(self.main_frame)
        self.scrollbar = ttk.Scrollbar(self.main_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Styling
        s = ttk.Style()
        s.configure('TButton', font=('Arial', 10), padding=6)
        s.configure('TLabel', font=('Arial', 10))
        s.configure('TEntry', font=('Arial', 10))
        s.configure('TCombobox', font=('Arial', 10))
        s.configure('TScale', background='#f0f0f0')
        s.configure('TProgressbar', thickness=15)
        s.configure('TFrame', background='#f0f0f0')
        s.configure('TLabelframe.Label', font=('Arial', 11, 'bold'))

        self.rcm_file = ""
        self.trial_file = ""
        self.rcm_sheet = tk.StringVar()
        self.trial_sheet = tk.StringVar()
        self.output_excel_file = "Final Automation Report.xlsx"
        self.output_pdf_file = "SOX_Charts.pdf"
        self.threshold = tk.DoubleVar(value=80.0)

        # All UI elements now added to self.scrollable_frame
        # Help/Instructions Frame
        help_frame = ttk.LabelFrame(self.scrollable_frame, text="Instructions", padding="10 10 10 10")
        help_frame.grid(row=0, column=0, columnspan=4, padx=10, pady=10, sticky="ew")
        tk.Label(help_frame, text="1. Upload your RCM and Trial Balance files (Excel only).\n"
                                     "2. Select the correct sheets for each file.\n"
                                     "3. Adjust the 'In Scope' threshold if needed.\n"
                                     "4. Choose the Account Types you want to analyze.\n"
                                     "5. Click 'Run Analysis' to generate the Excel report and PDF charts.\n"
                                     "   - RCM: Must contain 'Control Description', 'Control ID', 'Entity', 'Key? (Y/N)' columns.\n"
                                     "   - Trial Balance: Must contain 'Account Type' column as the first, followed by entity columns with values.",
                       justify=tk.LEFT).pack(anchor="w")


        # File Upload Frame
        file_frame = ttk.LabelFrame(self.scrollable_frame, text="File Upload & Selection", padding="10 10 10 10")
        file_frame.grid(row=1, column=0, columnspan=4, padx=10, pady=5, sticky="ew")

        tk.Label(file_frame, text="RCM Excel File:").grid(row=0, column=0, sticky='w', pady=2)
        self.rcm_entry = tk.Entry(file_frame, width=60)
        self.rcm_entry.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        tk.Button(file_frame, text="Browse", command=self.load_rcm).grid(row=0, column=2, padx=2, pady=2)

        tk.Label(file_frame, text="RCM Sheet:").grid(row=1, column=0, sticky='w', pady=2)
        self.rcm_dropdown = ttk.Combobox(file_frame, textvariable=self.rcm_sheet, state="readonly", width=57)
        self.rcm_dropdown.grid(row=1, column=1, padx=5, pady=2, sticky='w')

        tk.Label(file_frame, text="Trial Balance Excel File:").grid(row=2, column=0, sticky='w', pady=2)
        self.trial_entry = tk.Entry(file_frame, width=60)
        self.trial_entry.grid(row=2, column=1, padx=5, pady=2, sticky="ew")
        tk.Button(file_frame, text="Browse", command=self.load_trial).grid(row=2, column=2, padx=2, pady=2)

        tk.Label(file_frame, text="Trial Balance Sheet:").grid(row=3, column=0, sticky='w', pady=2)
        self.trial_dropdown = ttk.Combobox(file_frame, textvariable=self.trial_sheet, state="readonly", width=57)
        self.trial_dropdown.grid(row=3, column=1, padx=5, pady=2, sticky='w')

        # Data Preview Frame
        preview_frame = ttk.LabelFrame(self.scrollable_frame, text="Data Previews (First 5 Rows)", padding="10 10 10 10")
        preview_frame.grid(row=4, column=0, columnspan=4, padx=10, pady=5, sticky="ew")

        tk.Label(preview_frame, text="RCM Data:").grid(row=0, column=0, sticky="w", pady=2)
        self.rcm_preview_text = tk.Text(preview_frame, height=5, width=40, state="disabled", wrap="none")
        self.rcm_preview_text.grid(row=1, column=0, padx=5, pady=2, sticky="ew")
        rcm_scroll_x = tk.Scrollbar(preview_frame, orient="horizontal", command=self.rcm_preview_text.xview)
        rcm_scroll_x.grid(row=2, column=0, sticky="ew")
        self.rcm_preview_text.config(xscrollcommand=rcm_scroll_x.set)


        tk.Label(preview_frame, text="Trial Balance Data:").grid(row=0, column=1, sticky="w", pady=2)
        self.tb_preview_text = tk.Text(preview_frame, height=5, width=40, state="disabled", wrap="none")
        self.tb_preview_text.grid(row=1, column=1, padx=5, pady=2, sticky="ew")
        tb_scroll_x = tk.Scrollbar(preview_frame, orient="horizontal", command=self.tb_preview_text.xview)
        tb_scroll_x.grid(row=2, column=1, sticky="ew")
        self.tb_preview_text.config(xscrollcommand=tb_scroll_x.set)
        
        preview_frame.grid_columnconfigure(0, weight=1)
        preview_frame.grid_columnconfigure(1, weight=1)

        # Analysis Options Frame
        options_frame = ttk.LabelFrame(self.scrollable_frame, text="Analysis Options", padding="10 10 10 10")
        options_frame.grid(row=3, column=0, columnspan=4, padx=10, pady=5, sticky="ew")

        tk.Label(options_frame, text="Threshold for In Scope (%):").grid(row=0, column=0, sticky='w', pady=2)
        self.threshold_slider = ttk.Scale(options_frame, from_=50, to=100, orient="horizontal",
                                             variable=self.threshold, command=self.update_threshold_label, length=200)
        self.threshold_slider.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        self.threshold_label = tk.Label(options_frame, text=f"{self.threshold.get():.0f}%")
        self.threshold_label.grid(row=0, column=2, sticky='w')
        options_frame.grid_columnconfigure(1, weight=1)

        tk.Label(options_frame, text="Select Account Types:").grid(row=1, column=0, sticky='nw', pady=5)
        self.account_listbox = tk.Listbox(options_frame, selectmode=tk.MULTIPLE, width=50, height=6, exportselection=False)
        self.account_listbox.grid(row=1, column=1, columnspan=2, padx=5, pady=5, sticky='ew')
        
        listbox_scroll_y = ttk.Scrollbar(options_frame, orient="vertical", command=self.account_listbox.yview)
        listbox_scroll_y.grid(row=1, column=3, sticky='ns', padx=(0,5))
        self.account_listbox.config(yscrollcommand=listbox_scroll_y.set)

        self.select_all_checkbox_var = tk.BooleanVar()
        self.select_all_checkbox = ttk.Checkbutton(options_frame, text="Select All Account Types", 
                                                    variable=self.select_all_checkbox_var, command=self.toggle_select_all)
        self.select_all_checkbox.grid(row=2, column=1, columnspan=2, sticky='w', pady=5)


        # Output Frame
        output_frame = ttk.LabelFrame(self.scrollable_frame, text="Output Options & Control", padding="10 10 10 10")
        output_frame.grid(row=4, column=0, columnspan=4, padx=10, pady=5, sticky="ew")

        tk.Label(output_frame, text="Save Excel Report As:").grid(row=0, column=0, sticky='w', pady=2)
        self.save_entry = tk.Entry(output_frame, width=50)
        self.save_entry.insert(0, self.output_excel_file)
        self.save_entry.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        tk.Button(output_frame, text="Browse", command=self.save_as).grid(row=0, column=2, padx=2, pady=2)

        self.progress = ttk.Progressbar(output_frame, orient="horizontal", length=500, mode="determinate")
        self.progress.grid(row=1, column=0, columnspan=3, pady=10, sticky="ew")

        self.run_button = tk.Button(output_frame, text="Run Analysis", command=self.run_analysis, height=2, width=15)
        self.run_button.grid(row=2, column=0, pady=10, padx=5, sticky="w")
        
        self.reset_button = tk.Button(output_frame, text="Reset Application", command=self.reset_app, height=2, width=15)
        self.reset_button.grid(row=2, column=1, pady=10, padx=5, sticky="w")

        # Configure column weights for resizing
        self.scrollable_frame.grid_columnconfigure(0, weight=1)
        self.scrollable_frame.grid_columnconfigure(1, weight=1)
        self.scrollable_frame.grid_columnconfigure(2, weight=1)
        self.scrollable_frame.grid_columnconfigure(3, weight=1)

    def update_threshold_label(self, event=None):
        self.threshold_label.config(text=f"{self.threshold.get():.0f}%")

    def load_rcm(self):
        self.rcm_file = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        self.rcm_entry.delete(0, tk.END)
        self.rcm_entry.insert(0, self.rcm_file)
        self.rcm_preview_text.config(state="normal")
        self.rcm_preview_text.delete("1.0", tk.END)
        if self.rcm_file:
            try:
                xls = pd.ExcelFile(self.rcm_file)
                self.rcm_dropdown['values'] = xls.sheet_names
                if xls.sheet_names:
                    self.rcm_sheet.set(xls.sheet_names[0])
                    df_preview = pd.read_excel(self.rcm_file, sheet_name=xls.sheet_names[0])
                    self.rcm_preview_text.insert(tk.END, df_preview.head().to_string())
            except Exception as e:
                messagebox.showerror("Error", f"Could not read RCM file or sheets: {e}")
            finally:
                self.rcm_preview_text.config(state="disabled")

    def load_trial(self):
        self.trial_file = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        self.trial_entry.delete(0, tk.END)
        self.trial_entry.insert(0, self.trial_file)
        self.tb_preview_text.config(state="normal")
        self.tb_preview_text.delete("1.0", tk.END)

        if self.trial_file:
            try:
                xls = pd.ExcelFile(self.trial_file)
                self.trial_dropdown['values'] = xls.sheet_names
                if xls.sheet_names:
                    self.trial_sheet.set(xls.sheet_names[0])
                    df_full = pd.read_excel(self.trial_file, sheet_name=xls.sheet_names[0])
                    self.tb_preview_text.insert(tk.END, df_full.head().to_string())

                    required_tb_col = 'Account Type'
                    if required_tb_col not in df_full.columns:
                        messagebox.showerror("Validation Error", f"Trial Balance file is missing the required column: '{required_tb_col}'. Please check your file.")
                        self.account_listbox.delete(0, tk.END)
                        return
                    
                    # Check if there are columns beyond 'Account Type' assumed to be brands
                    # The image shows columns 'B', 'C' after 'Account Type' that would be entity values.
                    # This check now explicitly ensures there's at least one brand column.
                    # The error message from the images (e.g., "Entity" missing) indicates 
                    # that a specific column for Entitys might be expected. 
                    # If "Entity" is expected to be a column in the Trial Balance (not just data under generic column names), 
                    # you should add an explicit check for it here.
                    # Based on the UI description ("followed by entity columns with values"), 
                    # and the preview image, it seems generic entity columns are expected, not necessarily one named "Entity".
                    # Let's align the error message with this behavior.
                    if len(df_full.columns) < 2: 
                        messagebox.showerror("Validation Error", "Trial Balance file must contain at least one column for entity values in addition to the 'Account Type' column.")
                        self.account_listbox.delete(0, tk.END)
                        return

                    unique_accounts = sorted(df_full['Account Type'].dropna().astype(str).str.strip().unique())
                    self.account_listbox.delete(0, tk.END)
                    for acc in unique_accounts:
                        self.account_listbox.insert(tk.END, acc)
            except Exception as e:
                messagebox.showerror("Error", f"Could not read Trial Balance file or sheets: {e}")
            finally:
                self.tb_preview_text.config(state="disabled")

    def save_as(self):
        self.output_excel_file = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            initialfile="Final Automation Report.xlsx"
        )
        if self.output_excel_file:
            self.save_entry.delete(0, tk.END)
            self.save_entry.insert(0, self.output_excel_file)
            base_name = os.path.splitext(os.path.basename(self.output_excel_file))[0]
            self.output_pdf_file = os.path.join(os.path.dirname(self.output_excel_file), f"{base_name}_Charts.pdf")

    def toggle_select_all(self):
        if self.select_all_checkbox_var.get():
            self.account_listbox.select_set(0, tk.END)
        else:
            self.account_listbox.selection_clear(0, tk.END)

    def reset_app(self):
        self.rcm_file = ""
        self.trial_file = ""
        self.rcm_entry.delete(0, tk.END)
        self.trial_entry.delete(0, tk.END)
        self.rcm_sheet.set("")
        self.trial_sheet.set("")
        self.rcm_dropdown['values'] = []
        self.trial_dropdown['values'] = []
        self.account_listbox.delete(0, tk.END)
        self.select_all_checkbox_var.set(False)
        self.threshold.set(80.0)
        self.update_threshold_label()
        self.save_entry.delete(0, tk.END)
        self.output_excel_file = "Final Automation Report.xlsx"
        self.output_pdf_file = "SOX_Charts.pdf"
        self.save_entry.insert(0, self.output_excel_file)
        self.progress['value'] = 0
        self.rcm_preview_text.config(state="normal")
        self.rcm_preview_text.delete("1.0", tk.END)
        self.rcm_preview_text.config(state="disabled")
        self.tb_preview_text.config(state="normal")
        self.tb_preview_text.delete("1.0", tk.END)
        self.tb_preview_text.config(state="disabled")
        messagebox.showinfo("Reset", "Application has been reset.")


    def run_analysis(self):
        self.progress['value'] = 0
        try:
            if not self.rcm_file or not self.trial_file:
                messagebox.showerror("Input Error", "Please upload both RCM and Trial Balance files.")
                return
            if not self.rcm_sheet.get() or not self.trial_sheet.get():
                messagebox.showerror("Input Error", "Please select sheets for both RCM and Trial Balance files.")
                return

            selected_indices = self.account_listbox.curselection()
            if not selected_indices:
                messagebox.showerror("Input Error", "Please select at least one Account Type for analysis.")
                return
            account_types = [self.account_listbox.get(i).strip().lower() for i in selected_indices]

            rcm_df = pd.read_excel(self.rcm_file, sheet_name=self.rcm_sheet.get())
            trial_df = pd.read_excel(self.trial_file, sheet_name=self.trial_sheet.get())

            required_rcm_columns = ['Control Description', 'Control ID', 'Entity', 'Key? (Y/N)']
            missing_rcm_cols = [col for col in required_rcm_columns if col not in rcm_df.columns]
            if missing_rcm_cols:
                messagebox.showerror("Validation Error", f"RCM file is missing one or more required columns: {', '.join(missing_rcm_cols)}. Please check your RCM file.")
                return

            required_tb_col = 'Account Type'
            if required_tb_col not in trial_df.columns:
                messagebox.showerror("Validation Error", f"Trial Balance file is missing the required column: '{required_tb_col}'.")
                return
            # This check aligns with the UI instruction about 'Account Type' being first, followed by entity columns.
            if len(trial_df.columns) < 2:
                messagebox.showerror("Validation Error", "Trial Balance file must contain at least one column for entity values in addition to the 'Account Type' column.")
                return

            rcm_df = drop_blank_rows(rcm_df)
            trial_df = drop_blank_rows(trial_df)

            output_dir = os.path.dirname(self.output_excel_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            pdf_pages = PdfPages(self.output_pdf_file)

            all_brand_summaries = []
            all_matched_controls = []
            
            individual_sheets_to_export_ordered = [] 

            total_account_types = len(account_types)
            processed_count = 0

            for account_type_input in account_types:
                self.progress['value'] = (processed_count / total_account_types) * 100
                self.master.update_idletasks()

                search_phrase = account_type_input.lower()
                
                current_matched_controls = rcm_df[
                    rcm_df['Control Description'].astype(str).str.lower().str.contains(search_phrase, na=False)
                ].copy()
                
                current_matched_controls['Mapped Process Group'] = current_matched_controls['Control ID'].apply(map_control_id_to_process)
                
                expected_group = account_type_to_process_group.get(search_phrase, None)
                if expected_group:
                    current_matched_controls = current_matched_controls[
                        current_matched_controls['Mapped Process Group'].str.lower() == expected_group.lower()
                    ]

                current_account_row = trial_df[trial_df['Account Type'].astype(str).str.strip().str.lower() == search_phrase]
                
                if current_matched_controls.empty or current_account_row.empty:
                    processed_count += 1
                    continue
                
                values_row = current_account_row.iloc[0].drop('Account Type', errors='ignore').fillna(0) 
                current_brand_values = values_row.apply(clean_number).reset_index()
                current_brand_values.columns = ['Entity', 'Account Value']
                
                current_brand_values = current_brand_values[current_brand_values['Account Value'] != 0].copy()

                if current_brand_values.empty:
                    processed_count += 1
                    continue

                total_value = current_brand_values['Account Value'].sum()
                current_brand_values['% of Total'] = current_brand_values['Account Value'] / total_value if total_value != 0 else 0.0
                
                current_brand_values = current_brand_values.sort_values(by='Account Value', ascending=False).reset_index(drop=True)
                current_brand_values['Cumulative %'] = current_brand_values['% of Total'].cumsum()

                current_threshold = self.threshold.get() / 100.0
                scope_flags, threshold_reached = [], False
                for cum in current_brand_values['Cumulative %']:
                    if not threshold_reached:
                        scope_flags.append("In Scope")
                        if cum >= current_threshold:
                            threshold_reached = True
                    else:
                        scope_flags.append("Out of Scope")
                current_brand_values['Scope'] = scope_flags

                matched_rcm_brands = current_matched_controls['Entity'].dropna().unique().tolist()
                current_brand_values['Mapped in RCM'] = current_brand_values['Entity'].apply(lambda x: "Yes" if x in matched_rcm_brands else "No")

                key_status_map = current_matched_controls.set_index('Entity')['Key? (Y/N)'].fillna('').astype(str).to_dict()
                current_brand_values['Key Status'] = current_brand_values['Entity'].apply(lambda x: key_status_map.get(x, "") if x in matched_rcm_brands else "")

                def derive_auditor_check_flag(row):
                    flag_messages = []

                    if row['Scope'] == "In Scope" and row['Mapped in RCM'] == "No":
                        flag_messages.append("⚠️ Review: In Scope & not Mapped in RCM")
                    else:
                        key_status = str(row['Key Status']).strip().lower()
                        if row['Mapped in RCM'] == "Yes":
                            if row['Scope'] == "In Scope" and key_status in ["no", "non-key"]:
                                flag_messages.append("⚠️ Review: In Scope & Non-Key")
                            elif row['Scope'] == "Out of Scope" and key_status in ["yes", "key"]:
                                flag_messages.append("⚠️ Review: Out of Scope & Key")
                    
                    return ", ".join(flag_messages) if flag_messages else ""

                current_brand_values['Flag - Manual Auditor Check'] = current_brand_values.apply(derive_auditor_check_flag, axis=1)
                
                current_brand_values['Account Type'] = account_type_input.title()
                
                summary_cols_order = ['Account Type', 'Entity', 'Account Value', '% of Total', 'Cumulative %', 
                                      'Scope', 'Mapped in RCM', 'Key Status', 'Flag - Manual Auditor Check']
                current_brand_values = current_brand_values[[col for col in summary_cols_order if col in current_brand_values.columns]]
                current_brand_values = drop_blank_rows(current_brand_values).reset_index(drop=True)

                if 'Entity' in current_matched_controls.columns:
                    temp_scope_map = current_brand_values.set_index('Entity')['Scope'].to_dict()
                    current_matched_controls['Scope'] = current_matched_controls['Entity'].map(temp_scope_map).fillna("Not Analyzed in TB Scope")
                else:
                    current_matched_controls['Scope'] = "N/A - Entity column missing in RCM for scope mapping" 
                
                current_matched_controls['Account Type'] = account_type_input.title()
                rcm_cols_order = ['Account Type'] + [col for col in current_matched_controls.columns if col != 'Account Type']
                current_matched_controls = current_matched_controls[rcm_cols_order]
                current_matched_controls = drop_blank_rows(current_matched_controls).reset_index(drop=True)


                all_brand_summaries.append(current_brand_values)
                all_matched_controls.append(current_matched_controls)
                individual_sheets_to_export_ordered.append((account_type_input.title(), current_brand_values, current_matched_controls))

                processed_count += 1


            with pd.ExcelWriter(self.output_excel_file, engine="xlsxwriter") as writer:
                workbook = writer.book
                
                data_base_fmt = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'border': 1})
                header_fmt = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'border': 1, 'bold': True, 'text_wrap': True}) 
                
                color_excel_formats = {}
                for key, color_code in color_rules.items():
                    color_excel_formats[key] = workbook.add_format({'bg_color': color_code, 'align': 'center', 'valign': 'vcenter', 'border': 1})


                if all_brand_summaries:
                    df_summary_consolidated = pd.concat(all_brand_summaries, ignore_index=True)

                    sheet_name = "ALL_AccountType_Summary"
                    df_summary_consolidated.to_excel(writer, sheet_name=sheet_name, index=False, header=False) 
                    sheet = writer.sheets[sheet_name]
                    
                    for col_num, col_name in enumerate(df_summary_consolidated.columns):
                        sheet.write(0, col_num, col_name, header_fmt)
                        if col_name == 'Control Description':
                            sheet.set_column(col_num, col_num, 40)
                        elif col_name in ['Account Type', 'Entity', 'Scope', 'Mapped in RCM', 'Key Status']:
                            sheet.set_column(col_num, col_num, 20)
                        elif col_name in ['Account Value', '% of Total', 'Cumulative %']:
                            sheet.set_column(col_num, col_num, 15)
                        elif col_name == 'Flag - Manual Auditor Check':
                            sheet.set_column(col_num, col_num, 35) 
                        else:
                            sheet.set_column(col_num, col_num, 15) 

                    for row_num, row in df_summary_consolidated.iterrows():
                        flag_value = row.get('Flag - Manual Auditor Check', '')
                        scope_mapped_tuple = (row.get('Scope'), row.get('Mapped in RCM'))
                        
                        highlight_color_key = None
                        if flag_value and flag_value in color_rules:
                            highlight_color_key = flag_value
                        elif scope_mapped_tuple in color_rules:
                            highlight_color_key = scope_mapped_tuple

                        current_cell_fmt = color_excel_formats.get(highlight_color_key, data_base_fmt) 

                        for col_num, value in enumerate(row):
                            display_value = "" if pd.isna(value) else value
                            col_name = df_summary_consolidated.columns[col_num]
                            
                            final_cell_format = workbook.add_format({
                                'bg_color': current_cell_fmt.bg_color, 
                                'align': 'center', 
                                'valign': 'vcenter', 
                                'border': 1
                            })

                            if col_name == 'Account Value':
                                final_cell_format.set_num_format('#,##0')
                            elif col_name in ['% of Total', 'Cumulative %']:
                                final_cell_format.set_num_format('0.00%')
                            
                            sheet.write(row_num + 1, col_num, display_value, final_cell_format)


                if all_matched_controls:
                    df_rcm_consolidated = pd.concat(all_matched_controls, ignore_index=True)

                    sheet_name = "ALL_RCM_Combined"
                    df_rcm_consolidated.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
                    sheet = writer.sheets[sheet_name]

                    for col_num, col_name in enumerate(df_rcm_consolidated.columns):
                        sheet.write(0, col_num, col_name, header_fmt)
                        if col_name in ['Control Description', 'Risk Description']:
                            sheet.set_column(col_num, col_num, 40)
                        elif col_name in ['Account Type', 'Control ID', 'Entity', 'Mapped Process Group', 'Scope', 'Key? (Y/N)']:
                            sheet.set_column(col_num, col_num, 25)
                        else:
                            sheet.set_column(col_num, col_num, 15) 

                    for row_num, row in df_rcm_consolidated.iterrows():
                        for col_num, value in enumerate(row):
                            sheet.write(row_num + 1, col_num, "" if pd.isna(value) else value, data_base_fmt)


                for (acc_type, brand_values_df, matched_controls_df) in individual_sheets_to_export_ordered:
                    sheet_name_summary = f"{acc_type[:25].replace(' ', '_')}_summary"
                    original_name = sheet_name_summary
                    counter = 1
                    while sheet_name_summary in writer.book.sheetnames: # Fixed: changed to property, not method
                        sheet_name_summary = f"{original_name}_{counter}"
                        counter += 1
                    
                    sheet_summary = writer.book.add_worksheet(sheet_name_summary)
                    
                    # Write headers for summary sheet
                    for col_num, col_name in enumerate(brand_values_df.columns):
                        sheet_summary.write(0, col_num, col_name, header_fmt)
                        if col_name == 'Control Description':
                            sheet_summary.set_column(col_num, col_num, 40)
                        elif col_name in ['Account Type', 'Entity', 'Scope', 'Mapped in RCM', 'Key Status']:
                            sheet_summary.set_column(col_num, col_num, 20)
                        elif col_name in ['Account Value', '% of Total', 'Cumulative %']:
                            sheet_summary.set_column(col_num, col_num, 15)
                        elif col_name == 'Flag - Manual Auditor Check':
                            sheet_summary.set_column(col_num, col_num, 35) 
                        else:
                            sheet_summary.set_column(col_num, col_num, 15) 

                    # Write data for summary sheet with formatting
                    for row_num, row in brand_values_df.iterrows():
                        flag_value = row.get('Flag - Manual Auditor Check', '')
                        scope_mapped_tuple = (row.get('Scope'), row.get('Mapped in RCM'))
                        
                        highlight_color_key = None
                        if flag_value and flag_value in color_rules:
                            highlight_color_key = flag_value
                        elif scope_mapped_tuple in color_rules:
                            highlight_color_key = scope_mapped_tuple

                        current_cell_fmt = color_excel_formats.get(highlight_color_key, data_base_fmt) 

                        for col_num, value in enumerate(row):
                            display_value = "" if pd.isna(value) else value
                            col_name = brand_values_df.columns[col_num]
                            
                            final_cell_format = workbook.add_format({
                                'bg_color': current_cell_fmt.bg_color, 
                                'align': 'center', 
                                'valign': 'vcenter', 
                                'border': 1
                            })

                            if col_name == 'Account Value':
                                final_cell_format.set_num_format('#,##0')
                            elif col_name in ['% of Total', 'Cumulative %']:
                                final_cell_format.set_num_format('0.00%')
                            
                            sheet_summary.write(row_num + 1, col_num, display_value, final_cell_format)

                    # Create RCM sheet for the current account type
                    sheet_name_rcm = f"{acc_type[:25].replace(' ', '_')}_RCM"
                    original_name_rcm = sheet_name_rcm
                    counter_rcm = 1
                    while sheet_name_rcm in writer.book.sheetnames: # Fixed: changed to property, not method
                        sheet_name_rcm = f"{original_name_rcm}_{counter_rcm}"
                        counter_rcm += 1
                    
                    sheet_rcm = writer.book.add_worksheet(sheet_name_rcm)

                    # Write headers for RCM sheet
                    for col_num, col_name in enumerate(matched_controls_df.columns):
                        sheet_rcm.write(0, col_num, col_name, header_fmt)
                        if col_name in ['Control Description', 'Risk Description']:
                            sheet_rcm.set_column(col_num, col_num, 40)
                        elif col_name in ['Account Type', 'Control ID', 'Entity', 'Mapped Process Group', 'Scope', 'Key? (Y/N)']:
                            sheet_rcm.set_column(col_num, col_num, 25)
                        else:
                            sheet_rcm.set_column(col_num, col_num, 15) 
                    
                    # Write data for RCM sheet
                    for row_num, row in matched_controls_df.iterrows():
                        for col_num, value in enumerate(row):
                            sheet_rcm.write(row_num + 1, col_num, "" if pd.isna(value) else value, data_base_fmt)
                
                    # Plotting and saving charts to PDF
                    if not brand_values_df.empty:
                        # Pie chart for Scope distribution
                        fig1, ax1 = plt.subplots(figsize=(10, 8))
                        scope_counts = brand_values_df['Scope'].value_counts()
                        ax1.pie(scope_counts, labels=scope_counts.index, autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral', 'lightgoldenrodyellow', 'lightgray'])
                        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                        ax1.set_title(f'Scope Distribution for {acc_type}')
                        plt.tight_layout()
                        pdf_pages.savefig(fig1)
                        plt.close(fig1)

                        # Bar chart for Account Value by Brand
                        fig2, ax2 = plt.subplots(figsize=(12, 8))
                        ax2.bar(brand_values_df['Entity'], brand_values_df['Account Value'], color='skyblue')
                        ax2.set_xlabel('Entity')
                        ax2.set_ylabel('Account Value')
                        ax2.set_title(f'Account Value by Brand for {acc_type}')
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        pdf_pages.savefig(fig2)
                        plt.close(fig2)

                        # Cumulative percentage line chart
                        fig3, ax3 = plt.subplots(figsize=(12, 8))
                        ax3.plot(brand_values_df['Entity'], brand_values_df['Cumulative %'], marker='o', linestyle='-', color='purple')
                        ax3.axhline(y=current_threshold, color='r', linestyle='--', label=f'In Scope Threshold ({self.threshold.get():.0f}%)')
                        ax3.set_xlabel('Entity')
                        ax3.set_ylabel('Cumulative % of Total Value')
                        ax3.set_title(f'Cumulative Percentage of Account Value for {acc_type}')
                        ax3.set_ylim(0, 1.05)
                        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
                        plt.xticks(rotation=45, ha='right')
                        plt.legend()
                        plt.grid(True)
                        plt.tight_layout()
                        pdf_pages.savefig(fig3)
                        plt.close(fig3)
                        
            pdf_pages.close()
            self.progress['value'] = 100
            messagebox.showinfo("Analysis Complete", 
                                f"Analysis completed successfully!\n"
                                f"Excel Report saved to: {os.path.abspath(self.output_excel_file)}\n"
                                f"PDF Charts saved to: {os.path.abspath(self.output_pdf_file)}")
        except Exception as e:
            messagebox.showerror("Error during analysis", f"An error occurred during analysis: {e}")
            self.progress['value'] = 0


def main():
    root = tk.Tk()
    app = SOXApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
