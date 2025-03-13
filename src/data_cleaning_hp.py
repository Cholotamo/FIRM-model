import os
import pandas as pd

input_dir = "input_data_cleaning"
output_dir = "output_data_cleaning"
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if file.endswith(".xlsx"):
        input_file = os.path.join(input_dir, file)
        
        # Read the Excel file, skipping the first 6 rows
        df = pd.read_excel(input_file, skiprows=6)
        
        # Remove the last column
        df = df.iloc[:, :-1]
        
        # Print the DataFrame to inspect the data
        print(f"Data from {file}:")
        print(df.head())
        
        # Ensure the dates are sorted
        df = df.sort_values(by='Date')
        
        # Calculate the rate of change of 'PX_LAST' from the previous day
        df['Rate_of_Change'] = df['PX_LAST'].pct_change()
        
        # Sort date again in reverse
        df = df.sort_values(by='Date', ascending=False)

        # Format the 'Date' column to the desired format
        # df['Date'] = df['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        
        # Save the cleaned data to a new CSV file
        output_file = os.path.join(output_dir, file.replace(".xlsx", ".csv"))
        df.to_csv(output_file, index=False)
        
        print(f"Cleaned data saved to {output_file}")