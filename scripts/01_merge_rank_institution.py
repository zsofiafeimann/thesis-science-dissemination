# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 17:32:33 2025

@author: Diak
"""

# === 1. Import and Setup ===
import pandas as pd
import re
import os
from unidecode import unidecode

# === 2. Display Settings ===
pd.set_option("display.max_columns", 50)

# === 3. Load Cleaned CSVs ===
df_institution_2019 = pd.read_csv('cleaned_data/institution_2019_clean.csv', encoding='utf-8', sep='|')
df_ranking = pd.read_csv('cleaned_data/ranking_clean.csv', encoding='utf-8', sep='|')

# Backup original names for reference
df_institution_2019['display_name_original'] = df_institution_2019['display_name']
df_ranking['name_original'] = df_ranking['name']

# === 4. Name Normalization Function ===
def normalization(text):
    """
    Clean and normalize institution names for reliable comparison.
    - Lowercases and removes accents
    - Removes invisible and special characters
    - Standardizes common education terms
    - Removes stopwords
    """
    if not isinstance(text, str):
        return text

    # Lowercase and remove accents
    text = unidecode(text).lower()

    # Remove invisible characters
    invisible_chars = [
        '\u200b', '\u00a0', '\ufeff', '\u202f', '\u2060',
        '\u180e', '\u200e', '\u200f'
    ]
    for ch in invisible_chars:
        text = text.replace(ch, '')

    # Replace dashes with spaces
    text = text.replace("-", " ")

    # Remove unwanted punctuation (keep digits)
    text = re.sub(r"[.,/\\&+:'\";=_@%!?()\[\]{}<>#^*~|‘’ʻʼʽˆ`ˋ´ˊ˘]", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Standardize common education terms
    replacements = {
        # Institute variants
        r"\binstitutet\b": "institute",
        r"\binstitute\b": "institute",
        r"\binstituto\b": "institute",
        r"\binstituut\b": "institute",
        r"\binstituttet\b": "institute",
        r"\binstitutt\b": "institute",
        r"\binstitut\b": "institute",
        # University variants
        r"\buniversite\b": "university",
        r"\buniversitat\b": "university",
        r"\buniversiteit\b": "university",
        r"\buniverzita\b": "university",
        r"\buniversidad\b": "university",
        r"\buniversità\b": "university",
        r"\buniversidade\b": "university",
        r"\buniversität\b": "university",
        r"\büniversite\b": "university",
        r"\buniverzitet\b": "university",
        r"\büniversitesi\b": "university",
        r"\byliopisto\b": "university",
        r"\begyetem\b": "university",
        # College variants
        r"\bcollege\b": "college",
        r"\bcollegio\b": "college",
        r"\bcolégio\b": "college",
        r"\bkolleg\b": "college",
        r"\bkolej\b": "college",
        r"\bkollégium\b": "college",
        r"\bkolegji\b": "college",
        # School / academy
        r"\bschool\b": "school",
        r"\bschule\b": "school",
        r"\bescola\b": "school",
        r"\bescuela\b": "school",
        r"\bskola\b": "school",
        r"\bécole\b": "school",
        r"\bakademie\b": "academy",
        r"\bacademy\b": "academy"
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)

    # Remove content inside parentheses
    text = re.sub(r'\s*\([^)]*\)', '', text)

    # Remove common stopwords
    stopwords = {
        "of", "the", "and", "in", "for", "a", "an", "at", "on", "to",
        "de", "del", "du", "di", "la", "le", "les", "des",
        "von", "der", "den", "da", "do", "das", "dos",
        "y", "e", "et", "und"
    }
    words = text.split()
    words = [w for w in words if w not in stopwords]
    text = " ".join(words)

    return text

# === 5. Apply Normalization to Names ===
df_institution_2019['display_name_clean'] = df_institution_2019['display_name'].apply(normalization)
df_ranking['name_clean'] = df_ranking['name'].apply(normalization)

# Check for duplicate normalized ranking names
df_ranking['name_clean'].value_counts().loc[lambda x: x > 1]

# === 6. Map Ranking Names to Rank Values ===
ranking_name_mapping = {}
for idx, row in df_ranking.iterrows():
    norm_name = row['name_clean']
    ranking_name_mapping[norm_name] = row['rank']

# === 7. Exact Match Based on Normalized Names ===
df_institution_2019['rank'] = df_institution_2019['display_name_clean'].map(ranking_name_mapping)

# === 8. Summary of Exact Matches ===
exact_matches = df_institution_2019['rank'].notna().sum()
total_rows = len(df_institution_2019)
print(f"Exact matches: {exact_matches} of {total_rows} ({(exact_matches / total_rows) * 100:.2f}%)")
unique_exact = df_institution_2019[df_institution_2019['rank'].notna()]['display_name_clean'].nunique()

# === 9. Fuzzy Matching Using RapidFuzz (first with a sample df) ===
from rapidfuzz import process, fuzz
from tqdm import tqdm
tqdm.pandas()

def fuzzy_match_wrapper_score(inst_name, threshold=90):
    """
    Fuzzy match institution name against ranking names.
    Returns: (matched name, match score) if above threshold, else (None, None)
    """
    norm_name = normalization(inst_name)
    result = process.extractOne(
        norm_name,
        list(ranking_name_mapping.keys()),
        scorer=fuzz.WRatio,
        score_cutoff=threshold
    )
    if result is None:
        return None, None
    matched_name, score, _ = result
    return matched_name, score

# Mask for institutions without an exact match
mask = df_institution_2019['rank'].isna()

### SAMPLE BEGIN (no need to run) ### 

# Sample for fuzzy matching (up to 1000 or all unmatched)
sample_size = min(1000, mask.sum())
sample_df = df_institution_2019[mask].sample(sample_size, random_state=42).copy()

# Apply fuzzy matching to the sample
results = sample_df['display_name_clean'].progress_apply(fuzzy_match_wrapper_score)
sample_df['fuzzy_matched_name'] = results.apply(lambda x: x[0])
sample_df['match_score'] = results.apply(lambda x: x[1])

# Summary of fuzzy matches in the sample
matched_count = sample_df['match_score'].notna().sum()
print(f"Fuzzy matches in sample: {matched_count} of {sample_size} ({(matched_count / sample_size) * 100:.2f}%)")

# Show a few example fuzzy matches
print(sample_df[sample_df['match_score'].notna()][[
    'display_name_original', 'display_name_clean', 'fuzzy_matched_name', 'match_score'
]].sample(min(10, matched_count)))


### SAMPLE END ###

# === 10. Full Fuzzy Matching (Threshold 90) ===
results = df_institution_2019.loc[mask, 'display_name_clean'].progress_apply(fuzzy_match_wrapper_score)
df_institution_2019.loc[mask, 'fuzzy_matched_name'] = results.apply(lambda x: x[0])
df_institution_2019.loc[mask, 'fuzzy_rank'] = results.apply(lambda x: x[1])

# Summary After Full Fuzzy Matching
fuzzy_matches = df_institution_2019['fuzzy_rank'].notna().sum()
print(f"Total fuzzy matches: {fuzzy_matches} of {len(df_institution_2019)} ({(fuzzy_matches / len(df_institution_2019)) * 100:.2f}%)")

# Show a few example fuzzy matches from the full dataset
sample_matches = df_institution_2019[df_institution_2019['fuzzy_rank'].notna()][[
    'display_name_original', 'display_name_clean', 'fuzzy_matched_name', 'fuzzy_rank'
]].sample(min(10, fuzzy_matches))
print(sample_matches)

# Get fuzzy matches that do not have an exact match
fuzzy_only = df_institution_2019[
    df_institution_2019['fuzzy_rank'].notna() &
    df_institution_2019['rank'].isna()
].copy()

# Sort and deduplicate fuzzy matches
fuzzy_only_sorted = fuzzy_only.sort_values(by='fuzzy_rank', ascending=False)
fuzzy_unique = fuzzy_only_sorted.drop_duplicates(subset='display_name_clean').copy()



# === 11. Manual Review of Fuzzy Matches ONLY WHEN MANUAL REVIEW IS NEEDED ===

# Export fuzzy candidates for manual checking 
fuzzy_export_path = "manual_review\fuzzy_candidates_with_manual_check.csv"
fuzzy_unique.to_csv(fuzzy_export_path, index=False)

# Read manually reviewed fuzzy matches
checked_fuzzy_df = pd.read_excel("manual_review/fuzzy_manual_checked.xlsx")
accepted_fuzzy = checked_fuzzy_df[checked_fuzzy_df["keep"] == 1].copy()
accepted_fuzzy = accepted_fuzzy[[
    'display_name_clean', 'fuzzy_matched_name', 'fuzzy_rank'
]]

# Initialize final_rank with exact match
df_institution_2019['final_rank'] = df_institution_2019['rank']

# Merge accepted fuzzy matches into main dataframe
df_institution_2019 = df_institution_2019.merge(
    accepted_fuzzy,
    on='display_name_clean',
    how='left',
    suffixes=('', '_fuzzy')
)

# Update final_rank: use accepted fuzzy if no exact match
df_institution_2019['final_rank'] = df_institution_2019['final_rank'].combine_first(
    df_institution_2019['fuzzy_rank']
)

# === 12. Secondary Ranking Statistics ===
total_rows = len(df_institution_2019)
final_ranked_rows = df_institution_2019['final_rank'].notna().sum()
print(f"Final ranked rows: {final_ranked_rows} of {total_rows} ({(final_ranked_rows / total_rows) * 100:.2f}%)")

unique_final_ranked = df_institution_2019[df_institution_2019['final_rank'].notna()]['display_name_clean'].nunique()
total_unique_institutions = df_institution_2019['display_name_clean'].nunique()
print(f"Unique institutions with final_rank: {unique_final_ranked} of {total_unique_institutions} ({(unique_final_ranked / total_unique_institutions) * 100:.2f}%)")

# === 13. Find Unmatched Ranked Institutions ONLY WHEN MANUAL REVIEW IS NEEDED ===
matched_names = set(
    df_institution_2019[df_institution_2019['final_rank'].notna()]['display_name_clean']
) | set(
    df_institution_2019[df_institution_2019['final_rank'].notna()]['fuzzy_matched_name'].dropna()
)

unmatched_ranked = df_ranking[~df_ranking['name_clean'].isin(matched_names)]
print(f"Ranked institutions NOT matched: {len(unmatched_ranked)} of {len(df_ranking)}")

# Export unmatched ranked institutions for manual review
export_umatch = unmatched_ranked[['rank', 'name_original', 'name_clean']].copy()
export_umatch.to_excel("manual_review/unmatched_ranked.xlsx", index=False)

# Export unique institution names for manual matching
unique_institutions_df = df_institution_2019[[
    'display_name_clean', 'display_name_original'
]].drop_duplicates(subset='display_name_clean')
unique_institution_export_path = "manual_review/unique_institution_names.xlsx"
unique_institutions_df.to_excel(unique_institution_export_path, index=False)

# === 14. Manual Pairing of Unmatched Institutions ===
manual_df = pd.read_excel("manual_review/unmatched_ranked_paired.xlsx")
manual_df = manual_df[manual_df['display_name_clean'].notna()]

df_institution_2019 = df_institution_2019.merge(
    manual_df[['display_name_clean', 'manual_rank']],
    on='display_name_clean',
    how='left'
)

# Update final_rank if still missing
df_institution_2019['final_rank'] = df_institution_2019['final_rank'].combine_first(
    df_institution_2019['manual_rank']
)

# === 15. Final Reporting ===
total_rows = len(df_institution_2019)
ranked_rows = df_institution_2019['final_rank'].notna().sum()
print(f"Final ranked rows: {ranked_rows} of {total_rows} ({(ranked_rows / total_rows) * 100:.2f}%)")

unique_final_ranked = df_institution_2019[df_institution_2019['final_rank'].notna()]['display_name_clean'].nunique()
total_unique_institutions = df_institution_2019['display_name_clean'].nunique()
print(f"Unique institutions with final_rank: {unique_final_ranked} of {total_unique_institutions} ({(unique_final_ranked / total_unique_institutions) * 100:.2f}%)")

# Add flag for ranked institutions
df_institution_2019['rank_flag'] = df_institution_2019['final_rank'].notna().astype(int)

# Select and export final columns
final_columns = [
    'parent_id', 'doi', 'author', 'author_position', 'institutions', 'ror',
    'display_name_original', 'display_name_clean', 'country_code', 'type',
    'homepage_url', 'final_rank', 'rank_flag'
]
final_df = df_institution_2019[final_columns].copy()
final_df.to_csv("cleaned_data/ranked_institution_2019.csv", index=False, sep="|")