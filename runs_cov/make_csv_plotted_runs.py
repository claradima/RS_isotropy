import pandas as pd

# List of entries to filter by
run_min_entries = [
    309434, 309439, 302640, 309504, 309501, 309448, 302159, 302145,
    302114, 356762, 356844, 356832, 355216, 304332, 302179, 356633,
    356007, 356010, 356009, 352664, 352620, 300724, 352937, 351950,
    351884, 351847, 311236, 351845, 358123, 309623, 310937, 358103,
    354046, 357527, 359089, 357397, 302187, 357486, 355415,
    362215, 303334, 311222, 355145, 358230, 304616, 358273, 304555,
    358296, 357973, 355478, 353689, 302948, 351266, 353081,
    353093, 300512, 300515, 304823, 304799, 300091
]

# Read the original CSV file
df = pd.read_csv('runs.csv')

# Filter the DataFrame based on the 'run_min' column
filtered_df = df[df['run_min'].isin(run_min_entries)]

# Write the filtered DataFrame to a new CSV file
filtered_df.to_csv('filtered_runs.csv', index=False)

print("Filtered rows have been saved to 'filtered_runs.csv'.")
