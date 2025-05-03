import pandas as pd
from pathlib import Path

# Define input and output file paths
input_path = Path("WELFake_Dataset.csv")
output_path = Path("WELFake_reduced.csv")

# Load the full CSV
df = pd.read_csv("WELFake_Dataset.csv")

# Sample 80% of the rows randomly (shuffle=True by default)
reduced_df = df.sample(frac=0.8, random_state=42)

# Save to new CSV
reduced_df.to_csv(output_path, index=False)

print(f"Reduced file saved to {output_path}")
print(f"Original rows: {len(df)}, Reduced rows: {len(reduced_df)}")
