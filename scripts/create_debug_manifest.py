import pandas as pd

# Read the full split manifest
df = pd.read_csv("metadata/split_manifest.csv")

# Keep only training videos
train = df[df["split"] == "train"]

# Select 5 real and 5 fake videos
real = train[train["label"] == 1].head(5)
fake = train[train["label"] == 0].head(5)

# Combine and save
debug = pd.concat([real, fake])

debug.to_csv("metadata/debug_manifest.csv", index=False)

print("Debug manifest created!")
print(debug)
