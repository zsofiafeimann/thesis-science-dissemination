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

# === 1. Load clean data ===
df_institution_2019 = pd.read_csv('cleaned_data/institution_2019_clean.csv', encoding='utf-8', sep = '|')
df_ranking = pd.read_csv('cleaned_data/ranking_clean.csv', encoding='utf-8', sep = '|')


# === 2. Normalize institution display names ===
df_institution_2019['display_name'] = df_institution_2019['display_name'].str.strip().str.lower()
df_ranking['name'] = df_ranking['name'].str.strip().str.lower()

# === 3. Merge institution and ranking datasets on exact name matches ===
df_institution_ranked_exact = df_institution_2019.merge(
    df_ranking,
    left_on='display_name',
    right_on='name',
    how='inner'  # only keep exact matches
)

# Check how many exact name matches were found
print("Number of records in df_institution_2019:", len(df_institution_2019))
print("Number of records in df_ranking:", len(df_ranking))
print("Number of exact matches:", len(df_institution_ranked_exact))

# Optionally: How many unique institutions matched exactly
exact_unique_matches = df_institution_ranked_exact['display_name'].nunique()
print("Unique institutions with exact match:", exact_unique_matches)


# === 4. Fuzzy match remaining institution names with ranking names ===
# Identify ranking institution names not matched exactly
ranking_names = set(df_ranking['name'].unique())
exact_matched_names = set(df_institution_ranked_exact['display_name'].unique())
not_matched_names = ranking_names - exact_matched_names

print("Number of ranking institutions:", len(ranking_names))
print("Number of exact matches:", len(exact_matched_names))
print("Number of unmatched ranking names (to fuzzy match):", len(not_matched_names))


# Run fuzzy matching on unmatched ranking institution names
from fuzzywuzzy import process, fuzz

# Normalization function for institution names
def normalize(text):
    return (
        text.lower()
        .replace("’", "'")
        .replace("–", "-")
        .replace("“", '"')
        .replace("”", '"')
        .replace("&", "and")
        .strip()
    )

# Create a mapping from normalized display names to original ones
name_map = {
    normalize(name): name
    for name in df_institution_2019['display_name'].unique()
}

# Initialize separate match categories
high_confidence_matches = {}  # score >= 97
manual_review_matches = {}    # score between 90 and 96

# Perform fuzzy matching for unmatched ranking names
for name in not_matched_names:
    normalized_name = normalize(name)
    best_match_norm, score = process.extractOne(
        normalized_name,
        list(name_map.keys()),
        scorer=fuzz.token_sort_ratio
    )
    best_match_original = name_map[best_match_norm]

    if score >= 97:
        high_confidence_matches[name] = (best_match_original, score)
    elif 90 <= score < 97:
        manual_review_matches[name] = (best_match_original, score)
        
        
        
# High-confidence matches
fuzzy_high_df = pd.DataFrame.from_dict(
    high_confidence_matches, orient='index', columns=['matched_name', 'score']
).reset_index().rename(columns={'index': 'ranking_name'})

# Manual review matches
fuzzy_manual_df = pd.DataFrame.from_dict(
    manual_review_matches, orient='index', columns=['matched_name', 'score']
).reset_index().rename(columns={'index': 'ranking_name'})


fuzzy_manual_df.to_csv('cleaned_data/manual_review_fuzzy_matches.csv', index=False, encoding='utf-8', sep = '|' )
fuzzy_high_df.to_csv('cleaned_data/high-confidence_fuzzy_matches.csv', index=False, encoding='utf-8', sep = '|' )


# Load the cleaned, manually reviewed fuzzy match files
high_df = pd.read_csv('cleaned_data/high_checked_fuzzy_matches.csv', encoding='utf-8', sep = '|')
manual_df = pd.read_csv('cleaned_data/manual_checked_fuzzy_matches.csv', encoding='utf-8', sep = '|')

# Concatenate both and keep only accepted rows
fuzzy_all = pd.concat([manual_df, high_df], ignore_index=True)
fuzzy_accepted = fuzzy_all[fuzzy_all['keep'] == 1]



# === 5. Merge fuzzy match, and exact match institution names with ranking names ===
# Merge accepted fuzzy matches with ranking data on the ranking_name
df_ranking_fuzzy = df_ranking.merge(
    fuzzy_accepted,
    left_on='name',
    right_on='ranking_name',
    how='inner'
)

# Merge fuzzy-ranked institutions with institution data
df_institution_ranked_fuzzy = df_institution_2019.merge(
    df_ranking_fuzzy,
    left_on='display_name',
    right_on='matched_name',
    how='inner'
)

# Combine with exact matches (previously created)
df_institution_ranked_all = pd.concat(
    [df_institution_ranked_exact, df_institution_ranked_fuzzy],
    ignore_index=True
)

# Save the merged institution + ranking dataset (exact + fuzzy matches)
df_institution_ranked_all.to_csv('cleaned_data/df_institution_ranked_all_raw.csv', index=False, encoding='utf-8', sep = '|')


# === 6. Clean new institution ranked df ===
# General info
df_institution_ranked_all.info()
print("Shape:", df_institution_ranked_all.shape)
print(df_institution_ranked_all.sample(5))
print("Columns:", df_institution_ranked_all.columns.tolist())


# Create a copy of the original merged dataset 
df_institution_ranked_all_clean = df_institution_ranked_all.copy()

# Drop unnecessary technical columns (fuzzy match + raw affiliation)
fuzzy_cols = ['ranking_name', 'matched_name', 'score', 'keep', 'source', 'raw_affiliation_string']
df_institution_ranked_all_clean.drop(
    columns=[col for col in fuzzy_cols if col in df_institution_ranked_all_clean.columns],
    inplace=True
)

print(df_institution_ranked_all_clean.columns)



df_institution_ranked_all_clean.to_csv('cleaned_data/institution_ranked_all_clean.csv', index=False, encoding='utf-8', sep = '|')





# === 7. Check unmatch institution names ===
matched_institutions = set(df_institution_ranked_all['name'].unique())
all_ranked_institutions = set(df_ranking['name'].unique())

unmatched_institutions = all_ranked_institutions - matched_institutions

print(f"Unmatched ranked institutions: {len(unmatched_institutions)}")
unmatched_df = df_ranking[df_ranking['name'].isin(unmatched_institutions)]
unmatched_cleaned = unmatched_df[['rank', 'name', 'aliases', 'location']]
unmatched_cleaned.to_csv('cleaned_data/unmatched_ranked_institutions.csv', index=False, encoding='utf-8', sep = '|')


