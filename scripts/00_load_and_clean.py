# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 10:36:54 2025

@author: Diak
"""

import pandas as pd
import numpy as np

# Set display options
pd.set_option("display.max_columns", 50)    

# === 1. Load raw data ===
df_author = pd.read_csv('data/merged_author_data.csv', encoding='utf-8')
df_data_2019 = pd.read_csv('data/data_2019.csv', encoding='utf-8')
df_ranking = pd.read_csv('data/2019_rankings.csv', encoding='utf-8')
df_institution = pd.read_csv('data/merged_institutions_data.csv', encoding='utf-8')



# === 2. Normalize column names ===
for df in [df_author, df_institution, df_ranking, df_data_2019]:
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_')
        .str.replace(r'[^\w\d_]', '', regex=True)
    )
    
print(df_data_2019.filter(like='unnamed').head())

# === 3. Drop unnamed columns if they exist ===
for df in [df_author, df_institution, df_ranking, df_data_2019]:
    unnamed_cols = [col for col in df.columns if col.startswith('unnamed')]
    df.drop(columns=unnamed_cols, inplace=True)
    
    
# === 4. Inspect df_author structure ===    
# Check the first few rows and general info
print(df_author.head())
print(df_author.info())
print(df_author.columns)

# Check nulls
print(df_author.isnull().sum())

# Drop completely empty 'last_known_institution' column
df_author.drop(columns=['last_known_institution'], inplace=True)

# Check duplicated authors
print(df_author[df_author.duplicated(subset='author', keep=False)])

# Check shape after cleaning
print(df_author.shape)


# === 5. Inspect df_data_2019 structure ===    
# Check the first few rows and general info
print(df_data_2019.head())
print(df_data_2019.info())
print(df_data_2019.columns)

# Count missing values in each column
missing_counts = df_data_2019.isnull().sum()
print(missing_counts[missing_counts > 0].sort_values(ascending=False))

# Essential columns we *require* to be present
essential_cols = ['altmetric_id', 'doi', 'pubdate', 'code']
non_essential_cols = [col for col in df_data_2019.columns if col not in essential_cols]

# Check if there are rows where all other fields are empty
only_essential = df_data_2019[non_essential_cols].isnull().all(axis=1)
print(f"Rows wich are empty except altmetric_id, doi, pubdate): {only_essential.sum()} / {len(df_data_2019)}")

# Fill missing codes with a placeholder
df_data_2019['code'] = df_data_2019['code'].fillna('unknown')

# Fill NaN values in altmetric-type columns with 0 (assuming missing means "no mention")
altmetric_cols = [
    'linkedin', 'misc', 'facebook', 'googleplus', 'video', 'weibo', 'twitter', 'wikipedia',
    'blogs', 'news', 'reddit', 'policy', 'patent', 'qa', 'pinterest', 'syllabi', 'f1000',
    'book_reviews', 'peer_reviews', 'stot', 'stot_log', 'stot_log_stand', 'stot_log_jb_stand'
]

df_data_2019[altmetric_cols] = df_data_2019[altmetric_cols].fillna(0)

# Convert publication date to datetime format
df_data_2019['pubdate'] = pd.to_datetime(df_data_2019['pubdate'], errors='coerce')

# Check
print(df_data_2019['pubdate'].dtype)
print(df_data_2019['pubdate'].min(), '→', df_data_2019['pubdate'].max())



# === 5. Inspect df_ranking structure === 
print(df_ranking.head())
print(df_ranking.info())
print(df_ranking.columns)


# Remove commas and convert to numeric
df_ranking['stats_number_students'] = (
    df_ranking['stats_number_students']
    .str.replace(',', '')
    .astype(float)
)

# Clean and convert 'stats_pc_intl_students' safely
df_ranking['stats_pc_intl_students'] = (
    df_ranking['stats_pc_intl_students']
    .astype(str)                          # Ensure string type
    .str.strip()                          # Remove leading/trailing spaces
    .replace('', np.nan)                  # Replace truly empty strings with NaN
    .str.replace('%', '', regex=False)    # Remove percentage symbol
    .replace('', np.nan)                  # In case removing '%' left an empty string
    .astype(float)                        # Finally convert to float
)





# === 5. Inspect df_institution structure === 
# Check the first few rows and general info
print(df_institution.head())
print(df_institution.info())
print(df_institution.columns)

# Filter institutional data to only include 2019 publications
print(df_data_2019['pub_year'].unique())
print(df_data_2019['pubdate'].str[:4].unique())
df_data_2019['all_citaitons'].describe()

# Normalize DOI formats in both datasets
print(df_data_2019['doi'].dropna().sample(5))
print(df_institution['doi'].dropna().sample(5))

df_data_2019['doi'] = df_data_2019['doi'].str.strip().str.lower()
df_institution['doi'] = df_institution['doi'].str.strip().str.lower()

# Remove 'https://doi.org/' prefix from institution DOIs
df_institution['doi'] = df_institution['doi'].str.replace('https://doi.org/', '', regex=False)


# Keep only rows where the DOI exists in the 2019 dataset
df_institution_2019 = df_institution[df_institution['doi'].isin(df_data_2019['doi'])]
print(len(df_institution_2019))



# Show a few non-null values if any
print(df_institution['raw_affiliation_string'].dropna().count






















# === 4. Filter institutional data to only include 2019 publications ===
print(df_data_2019['pub_year'].unique())
print(df_data_2019['pubdate'].str[:4].unique())
df_data_2019['all_citaitons'].describe()

# === 4.1 Normalize DOI formats in both datasets ===
print(df_data_2019['doi'].dropna().sample(5))
print(df_institution['doi'].dropna().sample(5))

df_data_2019['doi'] = df_data_2019['doi'].str.strip().str.lower()
df_institution['doi'] = df_institution['doi'].str.strip().str.lower()

# Remove 'https://doi.org/' prefix from institution DOIs
df_institution['doi'] = df_institution['doi'].str.replace('https://doi.org/', '', regex=False)
