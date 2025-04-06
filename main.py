import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# LOAD & CLEAN DATA
# ===============================
df = pd.read_csv("shoes.csv")

# Strip whitespace and lowercase column names to avoid KeyErrors
df.columns = df.columns.str.strip().str.lower()

print("Available columns:", df.columns)

# ===============================
# NUMPY OPERATIONS
# ===============================

# Ensure numeric columns for NumPy operations
np_array = df.select_dtypes(include=[np.number]).to_numpy()

# Fixed Type Arrays
print("\nData Types:", np_array.dtype)

# Creating Arrays
array1 = np.array([1, 2, 3])
array2 = np.arange(10)
print("\nArray1:", array1)
print("Array2:", array2)

# Indexing and Slicing
print("\nFirst Row:", np_array[0])
print("First 3 columns of first 5 rows:\n", np_array[:5, :3])

# Reshaping
if np_array.size % 2 == 0:
    reshaped = np_array.reshape(-1, 2)
    print("\nReshaped shape:", reshaped.shape)

# Concatenation and Splitting
split1, split2 = np.array_split(np_array, 2)
concat_array = np.concatenate((split1, split2))
print("\nConcatenated Shape:", concat_array.shape)

# Universal Functions & Aggregation
print("\nMean of column 0:", np.mean(np_array[:, 0]))
print("Sum of column 1:", np.sum(np_array[:, 1]))

# Broadcasting
print("\nColumn 0 - Column 1:\n", np_array[:, 0] - np_array[:, 1])

# Comparisons, Boolean Arrays, Masks
mask = (np_array[:, 0] - np_array[:, 1]) > 1000
print("\nHigh Discounts:\n", np_array[mask])

# Fancy Indexing
indices = [0, 2, 4]
print("\nSelected Rows:\n", np_array[indices])

# Sorting
sorted_array = np_array[np.argsort(np_array[:, 0])]
print("\nSorted by Column 0:\n", sorted_array[:5])

# Partial Sorting
k = 5
part_sorted = np.partition(np_array[:, 0], k)[:k]
print(f"\nTop {k} Cheapest Prices:\n", part_sorted)

# Clipping
clipped_prices = np.clip(np_array[:, 0], 0, 5000)
print("\nClipped Prices:\n", clipped_prices)

# Structured Arrays
structured = np.array([("Crocs", 5), ("FILA", 6)], dtype=[('brand', 'U10'), ('size', 'i4')])
print("\nStructured Array:\n", structured)

# ===============================
# PANDAS OPERATIONS
# ===============================

# Series and DataFrame Objects
if 'brand' in df.columns:
    print("\nSeries Example:\n", df["brand"].head())
else:
    print("\nColumn 'brand' not found in dataset.")

# Indexing & Selecting
if 'color' in df.columns:
    print("\nSelecting color column:\n", df["color"].head())

# Universal Functions
if 'price' in df.columns and 'offer_price' in df.columns:
    print("\nPrice + Offer Price:\n", df["price"] + df["offer_price"])

# Handling Missing Data
df.fillna(0, inplace=True)
print("\nNull Values After Fill:\n", df.isnull().sum())

# Hierarchical Indexing
if 'brand' in df.columns and 'color' in df.columns:
    df_hier = df.set_index(["brand", "color"])
    print("\nHierarchical Index Sample:\n", df_hier.head())

# Descriptive Stats
print("\nDescriptive Stats:\n", df.describe())

# Correlation
if 'price' in df.columns and 'offer_price' in df.columns:
    print("\nCorrelation:\n", df[["price", "offer_price"]].corr())

# Value Counts
if 'brand' in df.columns:
    print("\nBrand Counts:\n", df["brand"].value_counts())

# .apply() usage
if 'price' in df.columns:
    df["price_category"] = df["price"].apply(lambda x: "High" if x > 3000 else "Low")
    print("\nPrice Categories:\n", df[["price", "price_category"]].head())

# Sorting
if 'price' in df.columns:
    sorted_df = df.sort_values(by="price", ascending=False)
    print("\nTop 5 Expensive Products:\n", sorted_df[["brand", "price"]].head())

# Filtering
if 'brand' in df.columns and 'price' in df.columns:
    filtered = df[(df["brand"] == "FILA") & (df["price"] < 2000)]
    print("\nFiltered FILA under 2000:\n", filtered)

# Rename column
if 'offer_price' in df.columns:
    df.rename(columns={"offer_price": "discounted_price"}, inplace=True)

# Drop column
if "price_category" in df.columns:
    df.drop(columns=["price_category"], inplace=True)

# ===============================
# COMBINING DATASETS
# ===============================

# Concat
df_concat = pd.concat([df, df], axis=0)
print("\nConcatenated Shape:", df_concat.shape)

# Merge
if 'brand' in df.columns:
    merged_df = df.merge(df, on="brand", how="inner")
    print("Merged Shape:", merged_df.shape)

# Grouping & Aggregation
if 'brand' in df.columns and 'price' in df.columns:
    grouped = df.groupby("brand")["price"].mean()
    print("\nGrouped by brand:\n", grouped)

# Pivot Table
if 'color' in df.columns and 'discounted_price' in df.columns:
    pivot = df.pivot_table(index="color", values="discounted_price", aggfunc="mean")
    print("\nPivot Table:\n", pivot)

# ===============================
# MATPLOTLIB VISUALIZATIONS
# ===============================

# Bar Plot
if 'brand' in df.columns and 'price' in df.columns:
    grouped.plot(kind="bar", title="Avg Price by Brand", figsize=(8, 4))
    plt.ylabel("Avg Price")
    plt.tight_layout()
    plt.show()

# Pie Chart
if 'color' in df.columns:
    df["color"].value_counts().plot(kind="pie", autopct="%1.1f%%", title="Color Distribution")
    plt.ylabel("")
    plt.show()

# Scatter Plot
if 'price' in df.columns and 'discounted_price' in df.columns:
    plt.scatter(df["price"], df["discounted_price"], alpha=0.6, color='green')
    plt.xlabel("Price")
    plt.ylabel("Discounted Price")
    plt.title("Price vs Discounted Price")
    plt.grid(True)
    plt.show()

# Histogram
if 'price' in df.columns:
    plt.hist(df["price"], bins=20, color='skyblue', edgecolor='black')
    plt.title("Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.show()

# Box Plot
if 'price' in df.columns and 'discounted_price' in df.columns:
    df.boxplot(column=["price", "discounted_price"])
    plt.title("Box Plot of Price and Discounted Price")
    plt.ylabel("Amount")
    plt.show()

# Line Plot (First 20 records)
if 'price' in df.columns and 'discounted_price' in df.columns:
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
print("\nSaved modified data to modified_data.csv")
