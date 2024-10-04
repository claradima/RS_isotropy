import pandas as pd
import json

# Load the CSV file
input_file = "/home/claramariadima/SNO/RS_isotropy/runs_cov/run_selection_tables.csv"
df = pd.read_csv(input_file)

# Function to extract thresholds from meta_data JSON
def extract_thresholds(meta_data):
    try:
        meta_data_json = json.loads(meta_data)
        dqpmtproc = meta_data_json.get("dqhl", {}).get("notes", {}).get("dqpmtproc", {})
        general_cov_thresh = dqpmtproc.get("general_cov_thresh", None)
        panel_cov_thresh = dqpmtproc.get("panel_cov_thresh", None)
        crate_cov_thresh = dqpmtproc.get("crate_cov_thresh", None)
        return general_cov_thresh, panel_cov_thresh, crate_cov_thresh
    except (json.JSONDecodeError, TypeError):
        return None, None, None

# Apply the extraction function to the 'meta_data' column
df[['general_cov_thresh', 'panel_cov_thresh', 'crate_cov_thresh']] = df['meta_data'].apply(lambda x: pd.Series(extract_thresholds(x)))

# Drop rows where any of the extracted thresholds are missing (optional)
df.dropna(subset=['general_cov_thresh', 'panel_cov_thresh', 'crate_cov_thresh'], inplace=True)

# Sort the dataframe by general_cov_thresh, then panel_cov_thresh, then crate_cov_thresh
df_sorted = df.sort_values(by=['general_cov_thresh', 'panel_cov_thresh', 'crate_cov_thresh'], ascending=True)

# Save the sorted dataframe to a new CSV file
output_file = "/home/claramariadima/SNO/RS_isotropy/runs_cov/coverage_ordered_runs.csv"
df_sorted.to_csv(output_file, index=False)

print(f"Data sorted and saved to {output_file}")

print("Now creating simplified table ... ")

# Create a simplified DataFrame with only the required columns
df_simplified = df[['timestamp', 'run_min', 'general_cov_thresh', 'panel_cov_thresh', 'crate_cov_thresh']]

# Sort the simplified DataFrame by general_cov_thresh, then panel_cov_thresh, then crate_cov_thresh
df_simplified_sorted = df_simplified.sort_values(by=['general_cov_thresh', 'panel_cov_thresh', 'crate_cov_thresh'], ascending=True)

# Save the simplified sorted DataFrame to a new CSV file
output_file_simplified = "/home/claramariadima/SNO/RS_isotropy/runs_cov/coverage_ordered_runs_simplified.csv"
df_simplified_sorted.to_csv(output_file_simplified, index=False)

# Create a new DataFrame with only the first entry for each unique combination
df_unique = df_simplified_sorted.drop_duplicates(subset=['general_cov_thresh', 'panel_cov_thresh', 'crate_cov_thresh'])

# Save the unique entries to a new CSV file called runs.csv
output_file_runs = "/home/claramariadima/SNO/RS_isotropy/runs_cov/runs.csv"
df_unique.to_csv(output_file_runs, index=False)

print(f"Simplified data sorted and saved to {output_file_simplified}")
print(f"Unique entries saved to {output_file_runs}")
