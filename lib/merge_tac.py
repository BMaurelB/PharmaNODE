import pandas as pd
import numpy as np
# Load the datasets
try:
    df_mr4 = pd.read_csv('/Users/benjaminmaurel/Documents/Data_LG/FileSenderDownload_7-2-2025__9-5-56/validation_modele_vs_trapeze/pccp/mr4_tls_pccp.csv', sep=';')
    df_pccp = pd.read_csv('/Users/benjaminmaurel/Documents/Data_LG/FileSenderDownload_7-2-2025__9-5-56/validation_modele_vs_trapeze/pccp/pccp_tac.csv', sep=';', skiprows=1)
    df_pccp_age = pd.read_csv('/Users/benjaminmaurel/Documents/Data_LG/FileSenderDownload_7-2-2025__9-5-56/validation_modele_vs_trapeze/pccp/pccp_age.csv', sep=';')

except Exception as e:
    print(f"Error loading files: {e}")

# Data Cleaning and Preparation
# Clean column names by stripping leading/trailing spaces
df_mr4.columns = df_mr4.columns.str.strip()
df_pccp.columns = df_pccp.columns.str.strip()

# Rename columns for clarity and consistency
df_pccp.rename(columns={'#ID': 'ID'}, inplace=True)
df_pccp_age.rename(columns={'id': 'ID'}, inplace=True)
# Replace comma with dot for decimal conversion
for col in df_mr4.columns:
    if df_mr4[col].dtype == 'object':
        df_mr4[col] = df_mr4[col].str.replace(',', '.', regex=True)

for col in df_pccp.columns:
    if df_pccp[col].dtype == 'object':
        df_pccp[col] = df_pccp[col].str.replace(',', '.', regex=True)

for col in df_pccp_age.columns:
    if df_pccp_age[col].dtype == 'object':
        df_pccp_age[col] = df_pccp_age[col].str.replace(',', '.', regex=True)
# Convert necessary columns to numeric types
for df in [df_mr4, df_pccp]:
    for col in ['AMT', 'DOSE', 'PERI']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

# Define the mapping between PERI and visit
peri_to_visit = {
    1: 'D7',
    2: 'D14',
    3: 'M1',
    4: 'M3',
    5: 'M6'
}

# Create the dictionary to store the matches
id_map = {}

# Find unique IDs in mr4
unique_ids_mr4 = df_mr4['ID'].unique()

for id_mr4 in unique_ids_mr4:
    # Get all records for the current ID in mr4
    records_mr4 = df_mr4[df_mr4['ID'] == id_mr4]

    for _, row_mr4 in records_mr4.iterrows():
        peri = row_mr4['PERI']
        amt = row_mr4['AMT']

        if pd.notna(peri) and pd.notna(amt) and peri in peri_to_visit and amt>0:
            visit = peri_to_visit[peri]

            # Find matching records in pccp
            matching_pccp = df_pccp[(df_pccp['visit'] == visit) & (df_pccp['DOSE'] == amt)]
            if not matching_pccp.empty:
                unique_pccp_ids = matching_pccp['ID'].unique()

                if len(unique_pccp_ids) == 1:
                    id_pccp = unique_pccp_ids[0]
                    if id_mr4 not in id_map:
                        id_map[id_mr4] = id_pccp
                    # elif id_map[id_mr4] != id_pccp:
                    #     # If a mapping already exists and it's different, this is a conflict
                    #     print(f"Error: Conflicting matches for ID {id_mr4}")
                    #     id_map[id_mr4] = "Error: Multiple Matches"
                    #     break
                elif len(unique_pccp_ids) > 1:
                    diff = 99
                    ids = unique_pccp_ids.tolist()
                    lenght = len(unique_pccp_ids)
                    for patient_possible in unique_pccp_ids:
                        if records_mr4['AGE'].iloc[0]!= '.':
                            new_diff = np.abs(float(df_pccp_age[df_pccp_age['ID'] == patient_possible]['age']) - float(records_mr4['AGE'].iloc[0]))
                            if new_diff>1:
                                ids.remove(patient_possible)
                                lenght -=1
                    if lenght ==0:
                        pass
                    elif len(ids) > 1:
                        import pdb; pdb.set_trace()
                        print('Multiple Matches')

    # if id_mr4 in id_map and id_map[id_mr4] == "Error: Multiple Matches":
    #     continue

print("ID Mapping:")
print(id_map)
import pdb; pdb.set_trace()