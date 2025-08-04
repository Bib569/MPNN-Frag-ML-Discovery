import math
import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import Draw
from IPython.display import display
from rdkit.Chem import AllChem
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation

# Create necessary directories
os.makedirs('data', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Add this function at the beginning of your clustering analysis section
def set_sklearn_seed(estimator, random_state=42):
    """Set random_state parameter if the estimator has one"""
    if hasattr(estimator, 'random_state'):
        estimator.random_state = random_state
    return estimator

def parse(entry_list):
    entry_list = json.loads(entry_list)
    if entry_list:
        return [untag_donor(Chem.MolFromSmiles(x)) for x in entry_list]

def untag_donor(mol):
    # Handle case where mol is None (invalid SMILES)
    if mol is None:
        return None

    # Create an editable molecule from the original molecule
    editable_mol = Chem.EditableMol(mol)

    # Identify atom indices to remove
    atom_indices_to_remove = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 49]

    # Remove the atoms from the molecule (do it in reverse order to avoid indexing issues)
    for atom_idx in sorted(atom_indices_to_remove, reverse=True):
        editable_mol.RemoveAtom(atom_idx)
    untagged_mol = editable_mol.GetMol()
    return Chem.MolToSmiles(untagged_mol)

# Load and process the data
print("Loading ADA_Data_With_Tags.csv...")
df = pd.read_csv("data/ADA_Data_With_Tags_filtered.csv")

# Process and clean the data
print("Processing tagged SMILES data...")
df['Donor Core Untagged Smiles'] = df['Donor Core Smiles'].apply(lambda x: untag_donor(Chem.MolFromSmiles(x)) if pd.notna(x) else None)
df['Acceptors Untagged Smiles'] = df['Acceptors Smiles'].apply(lambda x: parse(x) if pd.notna(x) else None)
df['Acceptors Smiles'] = df['Acceptors Smiles'].apply(lambda x: json.loads(x))
df['Side Chains Untagged Smiles'] = df['Side Chains Smiles'].apply(lambda x: parse(x))
df['Side Chains Smiles'] = df['Side Chains Smiles'].apply(lambda x: json.loads(x))
df['Conjugated Extensions Untagged Smiles'] = df['Conjugated Extensions Smiles'].apply(lambda x: parse(x))
df['Conjugated Extensions Smiles'] = df['Conjugated Extensions Smiles'].apply(lambda x: json.loads(x))
df['Conjugated Extension to Donor Tags'] = df['Conjugated Extension to Donor Tags'].apply(lambda x: json.loads(x))
df['Conjugated Extension to Acceptors Tags'] = df['Conjugated Extension to Acceptors Tags'].apply(lambda x: json.loads(x))
df['Conjugated Extension to Side Chains Tags'] = df['Conjugated Extension to Side Chains Tags'].apply(lambda x: json.loads(x))

# Define property columns
property_columns = ['Diff_L-L+1', 'E_B', 'Osc. strength', 'Q20', 'EA', 'lambda', 'HL Gap']

# Process Donors
print("Processing donors...")
donor_tag_columns = ["Donor Core Smiles", "Donor to Acceptor Tags", "Donor to Side Chain Tags", "Donor to Conjugated Extension Tags"]
donor_agg_funcs = {col_name: 'first' for col_name in donor_tag_columns}
donor_agg_funcs.update({col_name: ['mean', 'std'] for col_name in property_columns})
unique_donors = df.groupby('Donor Core Untagged Smiles', as_index=False).agg(donor_agg_funcs).reset_index()

# Process Acceptors
print("Processing acceptors...")
df_acceptors = df.explode(['Acceptors Untagged Smiles', 'Acceptors Smiles'])
acceptor_agg_funcs = {"Acceptors Smiles": 'first'}
acceptor_agg_funcs.update({col_name: ['mean', 'std'] for col_name in property_columns})
unique_acceptors = df_acceptors.groupby('Acceptors Untagged Smiles', as_index=False).agg(acceptor_agg_funcs).reset_index()

# Process Side Chains
print("Processing side chains...")
df_sc = df.explode(["Side Chains Untagged Smiles", "Side Chains Smiles"])
sc_agg_funcs = {"Side Chains Smiles": 'first'}
sc_agg_funcs.update({col_name: ['mean', 'std'] for col_name in property_columns})
unique_sc = df_sc.groupby('Side Chains Untagged Smiles', as_index=False).agg(sc_agg_funcs).reset_index()

# Process Conjugated Extensions
print("Processing conjugated extensions...")
df_ce = df.explode(["Conjugated Extensions Untagged Smiles", "Conjugated Extensions Smiles",
                   "Conjugated Extension to Donor Tags", "Conjugated Extension to Acceptors Tags",
                   "Conjugated Extension to Side Chains Tags"])
ce_tag_columns = ["Conjugated Extensions Smiles", "Conjugated Extension to Donor Tags",
                 "Conjugated Extension to Acceptors Tags", "Conjugated Extension to Side Chains Tags"]
ce_agg_funcs = {col_name: 'first' for col_name in ce_tag_columns}
ce_agg_funcs.update({col_name: ['mean', 'std'] for col_name in property_columns})
unique_ce = df_ce.groupby('Conjugated Extensions Untagged Smiles', as_index=False).agg(ce_agg_funcs).reset_index()

print(f"Found {len(unique_donors)} unique donors, {len(unique_acceptors)} unique acceptors, "
      f"{len(unique_sc)} unique side chains, {len(unique_ce)} unique conjugated extensions")

# Clustering for Donors
print("\nClustering donors...")
donor_X_columns = [('Diff_L-L+1', 'mean'), ('E_B', 'mean'), ('lambda', 'mean'), ('HL Gap', 'mean')]
donor_Y_columns = [('Q20', 'mean'), ('Osc. strength', 'mean'), ('EA', 'mean')]
X = unique_donors[donor_X_columns]
Y = unique_donors[donor_Y_columns]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Y_scaled = scaler.fit_transform(Y)
data_combined = np.hstack((X_scaled, Y_scaled))

kmeans = set_sklearn_seed(KMeans(n_clusters=3, random_state=42))
clusters = kmeans.fit_predict(data_combined)
unique_donors["cluster"] = clusters

# Filter for optimal clusters
optimal_donor_clusters = [1, 2]  # Teal and Yellow cluster
optimal_donors = unique_donors[unique_donors['cluster'].isin(optimal_donor_clusters)].copy()

# Clustering for Acceptors
print("Clustering acceptors...")
acceptor_X_columns = [('Diff_L-L+1', 'mean'), ('E_B', 'mean'), ('lambda', 'mean'), ('HL Gap', 'mean')]
acceptor_Y_columns = [('Q20', 'mean'), ('Osc. strength', 'mean'), ('EA', 'mean')]
X = unique_acceptors[acceptor_X_columns]
Y = unique_acceptors[acceptor_Y_columns]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Y_scaled = scaler.fit_transform(Y)
data_combined = np.hstack((X_scaled, Y_scaled))

kmeans = set_sklearn_seed(KMeans(n_clusters=3, random_state=42))
clusters = kmeans.fit_predict(data_combined)
unique_acceptors["cluster"] = clusters

# Filter for optimal clusters
optimal_acceptor_clusters = [0, 1]  # Purple cluster with better electron-accepting capabilities and Teal Cluster
optimal_acceptors = unique_acceptors[unique_acceptors['cluster'].isin(optimal_acceptor_clusters)].copy()

# Clustering for Side Chains
print("Clustering side chains...")
sc_X_columns = [('Diff_L-L+1', 'mean'), ('E_B', 'mean'), ('lambda', 'mean'), ('HL Gap', 'mean')]
sc_Y_columns = [('Q20', 'mean'), ('Osc. strength', 'mean'), ('EA', 'mean')]
X = unique_sc[sc_X_columns]
Y = unique_sc[sc_Y_columns]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Y_scaled = scaler.fit_transform(Y)
data_combined = np.hstack((X_scaled, Y_scaled))

kmeans = set_sklearn_seed(KMeans(n_clusters=3, random_state=42))
clusters = kmeans.fit_predict(data_combined)
unique_sc["cluster"] = clusters

# Filter for optimal clusters
optimal_sc_clusters = [0, 2]  # Purple and Yellow cluster with high EA and versatility
optimal_sc = unique_sc[unique_sc['cluster'].isin(optimal_sc_clusters)].copy()

# Clustering for Conjugated Extensions
print("Clustering conjugated extensions...")
ce_X_columns = [('Diff_L-L+1', 'mean'), ('E_B', 'mean'), ('lambda', 'mean'), ('HL Gap', 'mean')]
ce_Y_columns = [('Q20', 'mean'), ('Osc. strength', 'mean'), ('EA', 'mean')]
X = unique_ce[ce_X_columns]
Y = unique_ce[ce_Y_columns]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Y_scaled = scaler.fit_transform(Y)
data_combined = np.hstack((X_scaled, Y_scaled))

kmeans = set_sklearn_seed(KMeans(n_clusters=3, random_state=42))
clusters = kmeans.fit_predict(data_combined)
unique_ce["cluster"] = clusters

# Filter for optimal clusters
optimal_ce_clusters = [0, 2]  # Purple and Yellow cluster with low HOMO-LUMO gaps
optimal_ce = unique_ce[unique_ce['cluster'].isin(optimal_ce_clusters)].copy()

# Function to format DataFrame for ver2 format
def format_dataframe_ver2_style(df, fragment_type):
    """Format DataFrame to match ver2 CSV structure exactly"""

    # Create a new DataFrame with the exact structure
    formatted_df = pd.DataFrame()

    # Add index columns
    formatted_df['Unnamed: 0'] = range(len(df))
    formatted_df['index'] = range(len(df))

    if fragment_type == 'donors':
        # Donors specific columns
        formatted_df['Donor Core Untagged Smiles'] = df['Donor Core Untagged Smiles'].values
        formatted_df['Donor Core Smiles'] = df[('Donor Core Smiles', 'first')].values
        formatted_df['Donor to Acceptor Tags'] = df[('Donor to Acceptor Tags', 'first')].values
        formatted_df['Donor to Side Chain Tags'] = df[('Donor to Side Chain Tags', 'first')].values
        formatted_df['Donor to Conjugated Extension Tags'] = df[('Donor to Conjugated Extension Tags', 'first')].values

    elif fragment_type == 'acceptors':
        # Acceptors specific columns
        formatted_df['Acceptors Untagged Smiles'] = df['Acceptors Untagged Smiles'].values
        formatted_df['Acceptors Smiles'] = df[('Acceptors Smiles', 'first')].values

    elif fragment_type == 'side_chains':
        # Side chains specific columns
        formatted_df['Side Chains Untagged Smiles'] = df['Side Chains Untagged Smiles'].values
        formatted_df['Side Chains Smiles'] = df[('Side Chains Smiles', 'first')].values

    elif fragment_type == 'conjugated_extensions':
        # Conjugated extensions specific columns
        formatted_df['Conjugated Extensions Untagged Smiles'] = df['Conjugated Extensions Untagged Smiles'].values
        formatted_df['Conjugated Extensions Smiles'] = df[('Conjugated Extensions Smiles', 'first')].values
        formatted_df['Conjugated Extension to Donor Tags'] = df[('Conjugated Extension to Donor Tags', 'first')].values
        formatted_df['Conjugated Extension to Acceptors Tags'] = df[('Conjugated Extension to Acceptors Tags', 'first')].values
        formatted_df['Conjugated Extension to Side Chains Tags'] = df[('Conjugated Extension to Side Chains Tags', 'first')].values

    # Add property columns with mean and std (using proper .1 suffix for std)
    for prop in property_columns:
        formatted_df[prop] = df[(prop, 'mean')].values
        formatted_df[f'{prop}.1'] = df[(prop, 'std')].values

    return formatted_df

# Format and save Ver1 files in Ver2 format
print("\nFormatting and saving Ver1 CSV files in Ver2 format...")

# Save Donors Ver1
D_Ver1 = format_dataframe_ver2_style(optimal_donors, 'donors')
D_Ver1.to_csv('results/D_Ver1.csv', index=False)
print(f"Saved D_Ver1.csv with {len(D_Ver1)} donors")

# Save Acceptors Ver1
A_Ver1 = format_dataframe_ver2_style(optimal_acceptors, 'acceptors')
A_Ver1.to_csv('results/A_Ver1.csv', index=False)
print(f"Saved A_Ver1.csv with {len(A_Ver1)} acceptors")

# Save Side Chains Ver1
S_Ver1 = format_dataframe_ver2_style(optimal_sc, 'side_chains')
S_Ver1.to_csv('results/S_Ver1.csv', index=False)
print(f"Saved S_Ver1.csv with {len(S_Ver1)} side chains")

# Save Conjugated Extensions Ver1
C_Ver1 = format_dataframe_ver2_style(optimal_ce, 'conjugated_extensions')
C_Ver1.to_csv('results/C_Ver1.csv', index=False)
print(f"Saved C_Ver1.csv with {len(C_Ver1)} conjugated extensions")

# Also save in the format expected by parallel_molecular_generation_5lakh.py
print("\nSaving files for parallel molecular generation...")

# Save the optimal components to CSV files (for parallel generation)
optimal_donors_simple = pd.DataFrame({
    'Donor Core Smiles': optimal_donors[('Donor Core Smiles', 'first')],
    'Donor to Acceptor Tags': optimal_donors[('Donor to Acceptor Tags', 'first')],
    'Donor to Side Chain Tags': optimal_donors[('Donor to Side Chain Tags', 'first')],
    'Donor to Conjugated Extension Tags': optimal_donors[('Donor to Conjugated Extension Tags', 'first')]
})

optimal_acceptors_simple = pd.DataFrame({
    'Acceptors Smiles': optimal_acceptors[('Acceptors Smiles', 'first')]
})

optimal_sc_simple = pd.DataFrame({
    'Side Chains Smiles': optimal_sc[('Side Chains Smiles', 'first')]
})

optimal_ce_simple = pd.DataFrame({
    'Conjugated Extensions Smiles': optimal_ce[('Conjugated Extensions Smiles', 'first')],
    'Conjugated Extension to Donor Tags': optimal_ce[('Conjugated Extension to Donor Tags', 'first')],
    'Conjugated Extension to Acceptors Tags': optimal_ce[('Conjugated Extension to Acceptors Tags', 'first')],
    'Conjugated Extension to Side Chains Tags': optimal_ce[('Conjugated Extension to Side Chains Tags', 'first')]
})

optimal_donors_simple.to_csv('results/D.csv', index=False)
optimal_acceptors_simple.to_csv('results/A.csv', index=False)
optimal_sc_simple.to_csv('results/S.csv', index=False)
optimal_ce_simple.to_csv('results/C.csv', index=False)

print(f"Saved D.csv with {len(optimal_donors_simple)} donors")
print(f"Saved A.csv with {len(optimal_acceptors_simple)} acceptors")
print(f"Saved S.csv with {len(optimal_sc_simple)} side chains")
print(f"Saved C.csv with {len(optimal_ce_simple)} conjugated extensions")

# Create Join_ADA file for the next step
print("\nCreating Join_ADA.csv...")
min_length = min(len(optimal_donors_simple), len(optimal_acceptors_simple),
                len(optimal_sc_simple), len(optimal_ce_simple))

Join_ADA = pd.DataFrame()
Join_ADA["Donor Core Smiles"] = optimal_donors_simple["Donor Core Smiles"].iloc[:min_length]
Join_ADA["Acceptors Smiles"] = optimal_acceptors_simple["Acceptors Smiles"].iloc[:min_length]
Join_ADA["Side Chains Smiles"] = optimal_sc_simple["Side Chains Smiles"].iloc[:min_length]
Join_ADA["Conjugated Extensions Smiles"] = optimal_ce_simple["Conjugated Extensions Smiles"].iloc[:min_length]
Join_ADA["Donor to Acceptor Tags"] = optimal_donors_simple["Donor to Acceptor Tags"].iloc[:min_length]
Join_ADA["Donor to Side Chain Tags"] = optimal_donors_simple['Donor to Side Chain Tags'].iloc[:min_length]
Join_ADA["Donor to Conjugated Extension Tags"] = optimal_donors_simple["Donor to Conjugated Extension Tags"].iloc[:min_length]
Join_ADA["Conjugated Extensions to Acceptors Tags"] = optimal_ce_simple["Conjugated Extension to Acceptors Tags"].iloc[:min_length]
Join_ADA["Conjugated Extensions to Side Chains Tags"] = optimal_ce_simple["Conjugated Extension to Side Chains Tags"].iloc[:min_length]
Join_ADA["Conjugated Extensions to Donor Tags"] = optimal_ce_simple["Conjugated Extension to Donor Tags"].iloc[:min_length]
Join_ADA["Donor Cluster"] = optimal_donors["cluster"].iloc[:min_length]
Join_ADA["Acceptor Cluster"] = optimal_acceptors["cluster"].iloc[:min_length]
Join_ADA["Side Chain Cluster"] = optimal_sc["cluster"].iloc[:min_length]
Join_ADA["Conjugated Extension Cluster"] = optimal_ce["cluster"].iloc[:min_length]

Join_ADA.to_csv("results/Join_ADA.csv", index=False)
print(f"Saved Join_ADA.csv with {len(Join_ADA)} entries")

print("\n✅ All files successfully processed and saved!")
print("Files created:")
print("- D_Ver1.csv (Ver2 format)")
print("- A_Ver1.csv (Ver2 format)")
print("- S_Ver1.csv (Ver2 format)")
print("- C_Ver1.csv (Ver2 format)")
print("- D.csv (for parallel generation)")
print("- A.csv (for parallel generation)")
print("- S.csv (for parallel generation)")
print("- C.csv (for parallel generation)")

print("- Join_ADA.csv (combined data)")
