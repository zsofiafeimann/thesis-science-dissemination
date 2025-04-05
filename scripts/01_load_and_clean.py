# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 14:34:03 2025

@author: Diak
"""

import pandas as pd
pip install fuzzywuzzy
from fuzzywuzzy import process, fuzz
import os

# Set display options
pd.set_option("display.max_columns", 50)    

# === 1. Load raw data ===
df_author = pd.read_csv('data/merged_author_data.csv', encoding='utf-8')
df_institution = pd.read_csv('data/merged_institutions_data.csv', encoding='utf-8')
df_ranking = pd.read_csv('data/2019_rankings.csv', encoding='utf-8')
df_data_2019 = pd.read_csv('data/data_2019.csv', encoding='utf-8')


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


# === 4.2 Keep only rows where the DOI exists in the 2019 dataset ===
df_institution_2019 = df_institution[df_institution['doi'].isin(df_data_2019['doi'])]
print(len(df_institution_2019))


# === 5. Merge institution and ranking datasets on exact name matches ===
df_institution_ranked_exact = df_institution_2019.merge(
    df_ranking,
    left_on='display_name',
    right_on='name',
    how='inner'  # only keep exact matches
)

# === 5.1 Check how many exact name matches were found ===
print("Number of records in df_institution_2019:", len(df_institution_2019))
print("Number of records in df_ranking:", len(df_ranking))
print("Number of exact matches:", len(df_institution_ranked_exact))

# Optionally: How many unique institutions matched exactly
exact_unique_matches = df_institution_ranked_exact['display_name'].nunique()
print("Unique institutions with exact match:", exact_unique_matches)


# === 6. Fuzzy match remaining institution names with ranking names ===
# Identify ranking institution names not matched exactly
ranking_names = set(df_ranking['name'].unique())
exact_matched_names = set(df_institution_ranked_exact['display_name'].unique())
not_matched_names = ranking_names - exact_matched_names

print("Number of ranking institutions:", len(ranking_names))
print("Number of exact matches:", len(exact_matched_names))
print("Number of unmatched ranking names (to fuzzy match):", len(not_matched_names))


# === 7. Run fuzzy matching on unmatched ranking institution names ===
from fuzzywuzzy import process, fuzz

fuzzy_matches = {}

# Loop through unmatched ranking names and find best fuzzy match in institution data
for name in not_matched_names:
    best_match, score = process.extractOne(
        name,
        df_institution_2019['display_name'].unique(),
        scorer=fuzz.token_sort_ratio
    )
    if score >= 95:
        fuzzy_matches[name] = (best_match, score)
        
print(f"Number of high-confidence fuzzy matches (score >= 95): {len(fuzzy_matches)}")

# === 8. Convert fuzzy match results to DataFrame for merging ===
fuzzy_match_df = pd.DataFrame.from_dict(
    fuzzy_matches,
    orient='index',
    columns=['matched_name', 'score']
).reset_index().rename(columns={'index': 'ranking_name'})

fuzzy_match_df.head(20)



# === 9. Merge fuzzy matches with ranking and institution data ===
# Merge fuzzy match table with original ranking to get full institution info
df_ranking_fuzzy = df_ranking.merge(
    fuzzy_match_df,
    left_on='name',
    right_on='ranking_name',
    how='inner'
)

#  === 9.1. Merge the above with institution data using the matched institution name ===
df_institution_ranked_fuzzy = df_institution_2019.merge(
    df_ranking_fuzzy,
    left_on='display_name',
    right_on='matched_name',
    how='inner'
)

# === 10. Combine exact and fuzzy matched institution data ===
df_institution_ranked_all = pd.concat(
    [df_institution_ranked_exact, df_institution_ranked_fuzzy],
    ignore_index=True
)

print(f"Total matched records (exact + fuzzy): {len(df_institution_ranked_all)}")
print(f"Unique institutions matched (combined): {df_institution_ranked_all['display_name'].nunique()}")


# === 11. # Check which columns are completely empty (no values at all) ===
completely_empty_cols = df_institution_ranked_all.columns[
    df_institution_ranked_all.isna().all()
].tolist()

print("Columns with no values at all:")
print(completely_empty_cols)

# Drop the completely empty column
df_institution_ranked_all.drop(columns=['raw_affiliation_string'], inplace=True)

# Check where ranking_name has a value
df_institution_ranked_all[df_institution_ranked_all['ranking_name'].notna()][
    ['ranking_name', 'matched_name', 'display_name', 'score']
].head(20)

print(df_institution_ranked_all.columns.to_list())
df_institution_ranked_all.head()


# === 12. Drop fuzzy matching helper columns ===
columns_to_drop = ['ranking_name', 'matched_name', 'score']
df_institution_ranked = df_institution_ranked_all.drop(columns=columns_to_drop)


# === 13. Save the cleaned data files ===
# Ensure the cleaned_data directory exists
output_dir = r'C:\Users\Diak\Documents\thesis\thesis-science-dissemination\cleaned_data'
os.makedirs(output_dir, exist_ok=True)

# Save the cleaned data to the specified location
df_institution_ranked_all.to_csv(os.path.join(output_dir, "institution_ranked_all_withfuzzy.csv"), index=False)
df_author.to_csv(os.path.join(output_dir, "author_cleaned.csv"), index=False)
df_ranking.to_csv(os.path.join(output_dir, "ranking_cleaned.csv"), index=False)
df_data_2019.to_csv(os.path.join(output_dir, "data_2019_cleaned.csv"), index=False)
df_institution_2019.to_csv(os.path.join(output_dir, "institution_2019_cleaned.csv"), index=False)
df_institution_ranked.to_csv(os.path.join(output_dir, "institution_ranked.csv"), index=False)


print("Files saved successfully!")