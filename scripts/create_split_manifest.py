import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

# Load video list
df = pd.read_csv("metadata/video_list.csv")

# Split real and fake separately (stratified manually)
real_df = df[df["label"] == 1]
fake_df = df[df["label"] == 0]

def split_group(group):
    train, temp = train_test_split(
        group,
        test_size=0.30,
        random_state=SEED
    )

    val, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=SEED
    )

    return train, val, test

real_train, real_val, real_test = split_group(real_df)
fake_train, fake_val, fake_test = split_group(fake_df)

real_train["split"] = "train"
real_val["split"] = "val"
real_test["split"] = "test"

fake_train["split"] = "train"
fake_val["split"] = "val"
fake_test["split"] = "test"

final_df = pd.concat([
    real_train,
    real_val,
    real_test,
    fake_train,
    fake_val,
    fake_test
])

final_df.to_csv(
    "metadata/split_manifest.csv",
    index=False
)

print("Done!")
print(final_df["split"].value_counts())
