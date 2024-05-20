import pandas as pd
from datetime import datetime
import os


def process_files(directory_path, output_path):
    data_list = []

    # List all CSV files in the directory
    file_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.csv')]

    for file_path in file_paths:
        try:
            # Read the CSV file with ISO-8859-2 encoding and semicolon delimiter
            df = pd.read_csv(file_path, encoding='ISO-8859-2', delimiter=';')

            # Extract the date from the 'stan_rekordu_na' column
            date_str = df['stan_rekordu_na'].iloc[0]

            # Attempt to parse the date with multiple formats
            try:
                date = datetime.strptime(date_str, "%d-%m-%Y").strftime("%Y-%m-%d")
            except ValueError:
                date = datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d")

            # Determine the column name to use
            if 'liczba_wszystkich_zakazen' in df.columns:
                column_name = 'liczba_wszystkich_zakazen'
            elif 'liczba_przypadkow' in df.columns:
                column_name = 'liczba_przypadkow'
            else:
                print(f"Missing relevant column in file: {file_path}")
                print(f"Columns found: {df.columns}")
                continue

            # Extract the number of total infections for the entire country
            total_infections = df.loc[df['wojewodztwo'] == 'Cały kraj', column_name].values[0]

            # Append the extracted data to the list
            data_list.append([date, total_infections])
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    # Convert the list to a DataFrame
    result_df = pd.DataFrame(data_list, columns=["Data", "liczba_wszystkich_zakazen"])

    # Save the DataFrame to a CSV file
    result_df.to_csv(output_path, index=False, encoding='utf-8')



if __name__ == "__main__":
    # Directory containing all CSV files
    directory_path = "C:/Users/adria/DAZHBOG/DAZHBOG-lite/DANE_COVID"

    # Output file path
    output_path = "/DATA_ALL.csv"

    # Process the files
    process_files(directory_path, output_path)
