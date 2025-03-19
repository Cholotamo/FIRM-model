import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# CSV data as a multi-line string; you could also load from a file using pd.read_csv('your_file.csv')
csv_data = """Period,Threshold,Buy_F1,Sell_F1,Hold_F1
9,0.01,0.06299212598425197,0.4391891891891892,0.0
9,0.02,0.0,0.3486238532110092,0.656441717791411
9,0.03,0.0,0.08333333333333333,0.8341232227488151
9,0.04,0.0,0.0,0.9299363057324841
9,0.05,0.0,0.0,0.9629629629629629
9,0.06,0.0,0.0,0.9859154929577465
9,0.07,0.0,0.0,0.9940119760479041
9,0.08,0.0,0.0,1.0
9,0.09,0.0,0.0,1.0
9,0.1,0.0,0.0,1.0
20,0.01,0.42857142857142855,0.5108225108225108,0.21978021978021978
20,0.02,0.35714285714285715,0.5887850467289719,0.5466666666666666
20,0.03,0.0,0.5393258426966292,0.654320987654321
20,0.04,0.0,0.47058823529411764,0.8361581920903954
20,0.05,0.0,0.42424242424242425,0.89749430523918
20,0.06,0.0,0.0,0.9114470842332614
20,0.07,0.0,0.0,0.9344608879492601
20,0.08,0.0,0.0,0.9608247422680413
20,0.09,0.0,0.0,0.9838709677419355
20,0.1,0.0,0.0,0.9899799599198397
200,0.01,0.3448275862068966,0.0,0.019801980198019802
200,0.02,0.7855421686746988,0.0,0.0
200,0.03,0.26334519572953735,0.0,0.2191780821917808
200,0.04,0.53125,0.0,0.4054982817869416
200,0.05,0.576271186440678,0.0,0.2631578947368421
200,0.06,0.4186046511627907,0.0,0.41924398625429554
200,0.07,0.13333333333333333,0.0,0.48231511254019294
200,0.08,0.3125,0.0,0.296028880866426
200,0.09,0.5714285714285714,0.0,0.819047619047619
200,0.1,0.6666666666666666,0.0,0.9026548672566371
"""

# Read CSV data into a DataFrame
df = pd.read_csv(StringIO(csv_data))

### Faceted Line Plots ###
# To compare F1 scores for different classes across thresholds for each period,
# we first transform the data from wide to long format.
df_melt = df.melt(id_vars=['Period', 'Threshold'],
                  value_vars=['Buy_F1', 'Sell_F1', 'Hold_F1'],
                  var_name='Class',
                  value_name='F1')

# Create a FacetGrid: one subplot per Period
g = sns.FacetGrid(df_melt, col="Period", sharey=False, height=4, aspect=1.2)
g.map_dataframe(sns.lineplot, x='Threshold', y='F1', hue='Class', marker="o")
g.add_legend()
g.set_titles(col_template="Period: {col_name}")
plt.suptitle("F1 Scores vs. Threshold for Different Periods", y=1.05)
plt.tight_layout()
plt.show()

### Heatmaps ###
# For each class, pivot the data to have Period as rows and Threshold as columns.
classes = ['Buy_F1', 'Sell_F1', 'Hold_F1']
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, cls in zip(axes, classes):
    pivot = df.pivot(index="Period", columns="Threshold", values=cls)
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", ax=ax)
    ax.set_title(f"Heatmap for {cls}")

plt.suptitle("Heatmaps of F1 Scores for Different Classes", y=1.05)
plt.tight_layout()
plt.show()

### Grouped Bar Chart with Threshold Annotations ###
# For each Period and Class, determine the row where F1 is maximum.
df_max = df_melt.loc[df_melt.groupby(['Period', 'Class'])['F1'].idxmax()].reset_index(drop=True)

plt.figure(figsize=(10, 6))
ax = sns.barplot(data=df_max, x='Period', y='F1', hue='Class')
plt.title("Maximum F1 Score per Class and Period with Corresponding Thresholds")
plt.ylabel("Maximum F1 Score")

# Annotate each bar with the corresponding threshold value.
for i, row in df_max.iterrows():
    # Get the corresponding bar (patch) from the current axis.
    # Note: the order of patches corresponds to the order of rows in df_max.
    patch = ax.patches[i]
    # Compute the center position of the bar.
    x = patch.get_x() + patch.get_width() / 2
    y = patch.get_height()
    # Annotate with the threshold value rounded to 2 decimals.
    ax.text(x, y + 0.01, f"Thr: {row['Threshold']:.2f}", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()