import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("shoes.csv")


# Ensure numeric columns for NumPy operations
np_array = df.select_dtypes(include=[np.number]).to_numpy()

# ===============================
# NUMPY OPERATIONS
# ===============================

# Fixed Type Arrays
print("Data Types:", np_array.dtype)

# Creating Arrays
array1 = np.array([1, 2, 3])
array2 = np.arange(10)
print("Array1:", array1)
print("Array2:", array2)

# Indexing and Slicing
print("First Row:", np_array[0])
print("First 3 columns of first 5 rows:\n", np_array[:5, :3])

# Reshaping
if np_array.size % 2 == 0:
    reshaped = np_array.reshape(-1, 2)
    print("Reshaped shape:", reshaped.shape)

# Concatenation and Splitting
split1, split2 = np.array_split(np_array, 2)
concat_array = np.concatenate((split1, split2))
print("Concatenated Shape:", concat_array.shape)

# Universal Functions & Aggregation
print("Mean Price:", np.mean(np_array[:, 0]))
print("Sum Offer Price:", np.sum(np_array[:, 1]))

# Broadcasting
print("Discounts:", np_array[:, 0] - np_array[:, 1])

# Comparisons, Boolean Arrays, Masks
mask = (np_array[:, 0] - np_array[:, 1]) > 1000
print("High Discounts:\n", np_array[mask])

# Fancy Indexing
indices = [0, 2, 4]
print("Selected Rows:\n", np_array[indices])

# Sorting
sorted_array = np_array[np.argsort(np_array[:, 0])]
print("Sorted by Price:\n", sorted_array[:5])

# Partial Sorting
k = 5
part_sorted = np.partition(np_array[:, 0], k)[:k]
print(f"Top {k} Cheapest Prices:\n", part_sorted)

# Clipping
clipped_prices = np.clip(np_array[:, 0], 0, 5000)
print("Clipped Prices:\n", clipped_prices)

# Structured Arrays and Compound Types
structured = np.array([("Crocs", 5), ("FILA", 6)], dtype=[('brand', 'U10'), ('size', 'i4')])
print("Structured Array:\n", structured)

# ===============================
# PANDAS OPERATIONS
# ===============================

# Series and DataFrame Objects
print("Series Example:\n", df["brand"].head())

# Indexing & Selecting
print("Selecting color column:\n", df["color"].head())

# Universal Functions, Index Alignment
print("Price + Offer Price:\n", df["price"] + df["offer_price"])

# Handling Missing Data
df.fillna(0, inplace=True)
print("Null Values:\n", df.isnull().sum())

# Hierarchical Indexing
df_hier = df.set_index(["brand", "color"])
print("Hierarchical Index Sample:\n", df_hier.head())

# Descriptive Stats
print("Descriptive Stats:\n", df.describe())

# Correlation
print("Correlation:\n", df[["price", "offer_price"]].corr())

# Value Counts
print("Brand Counts:\n", df["brand"].value_counts())

# .apply() usage
df["price_category"] = df["price"].apply(lambda x: "High" if x > 3000 else "Low")
print("Price Categories:\n", df[["price", "price_category"]].head())

# Sorting
sorted_df = df.sort_values(by="price", ascending=False)
print("Top 5 Expensive Products:\n", sorted_df[["brand", "price"]].head())

# Filtering
filtered = df[(df["brand"] == "FILA") & (df["price"] < 2000)]
print("Filtered FILA under 2000:\n", filtered)

# Rename column
df.rename(columns={"offer_price": "discounted_price"}, inplace=True)

# Drop column
df.drop(columns=["price_category"], inplace=True)

# ===============================
# COMBINING DATASETS
# ===============================

# Concat
df_concat = pd.concat([df, df], axis=0)
print("Concatenated Shape:", df_concat.shape)

# Merge
merged_df = df.merge(df, on="brand", how="inner")
print("Merged Shape:", merged_df.shape)

# Grouping & Aggregation
grouped = df.groupby("brand")["price"].mean()
print("Grouped by brand:\n", grouped)

# Pivot Table
pivot = df.pivot_table(index="color", values="discounted_price", aggfunc="mean")
print("Pivot Table:\n", pivot)

# ===============================
# MATPLOTLIB VISUALIZATIONS
# ===============================

# Bar Plot
grouped.plot(kind="bar", title="Avg Price by Brand", figsize=(8, 4))
plt.ylabel("Avg Price")
plt.tight_layout()
plt.show()

# Pie Chart
df["color"].value_counts().plot(kind="pie", autopct="%1.1f%%", title="Color Distribution")
plt.ylabel("")
plt.show()

# Scatter Plot
plt.scatter(df["price"], df["discounted_price"], alpha=0.6, color='green')
plt.xlabel("Price")
plt.ylabel("Discounted Price")
plt.title("Price vs Discounted Price")
plt.grid(True)
plt.show()

# Histogram
plt.hist(df["price"], bins=20, color='skyblue', edgecolor='black')
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# Box Plot
df.boxplot(column=["price", "discounted_price"])
plt.title("Box Plot of Price and Discounted Price")
plt.ylabel("Amount")
plt.show()

# Line Plot (First 20 records)
sampled = df[["price", "discounted_price"]].head(20)
plt.plot(sampled["price"], label="Original Price", marker='o')
plt.plot(sampled["discounted_price"], label="Discounted", marker='x')
plt.title("Price vs Discounted Price (First 20)")
plt.legend()
plt.grid()
plt.show()

# ===============================
# EXPORT MODIFIED CSV
# ===============================
df.to_csv("modified_data.csv", index=False)
print("Saved modified data to modified_data.csv")
