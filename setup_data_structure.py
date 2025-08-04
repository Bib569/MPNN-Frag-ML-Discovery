#!/usr/bin/env python3
"""
Setup script to create the required directory structure and sample data files
for the NFA-ML-Discovery project.

Run this script before using the main pipeline:
python setup_data_structure.py
"""

import os
import pandas as pd

def create_directories():
    """Create the required directory structure."""
    directories = ['data', 'results', 'logs', 'models']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}/")

def create_sample_ewg_list():
    """Create a sample electron withdrawing groups list."""
    # Sample EWG SMILES - these are common electron-withdrawing groups used in NFAs
    sample_ewgs = [
        "C(=O)C#N",  # Dicyanomethylene
        "C(=O)O",    # Carboxylic acid
        "C#N",       # Cyano
        "C(=O)C(=O)", # Dicarbonyl
        "C(=C(C#N)C#N)", # Dicyanovinyl
        "C1=CC=C(C(=O)O)C=C1", # Benzoic acid
        "C1=CC=C(C#N)C=C1",    # Benzonitrile
    ]
    
    df_ewg = pd.DataFrame({
        'SMILES': sample_ewgs
    })
    
    df_ewg.to_csv('data/ewglist.csv', index=False)
    print(f"✓ Created sample EWG list: data/ewglist.csv ({len(sample_ewgs)} entries)")

def create_sample_data_template():
    """Create template files showing the expected data format."""
    
    # Template for the main dataset
    template_data = {
        'SMILES': ['CC1=CC=CC=C1', 'CC1=CC=C(C=C1)C=C2C=CC=CC2=O'],  # Sample SMILES
        'Diff_L-L+1': [1.5, 1.8],
        'E_b': [0.25, 0.30],
        'Osc. strength': [1.2, 1.6],
        'Q20': [10.5, 12.3],
        'A_EA': [2.1, 2.4],
        'λ(nm)': [650, 680],
        'H-L Gap (eV)': [1.8, 1.6]
    }
    
    template_df = pd.DataFrame(template_data)
    template_df.to_csv('data/Datasheet_Filtered_CorrelationPlot_5_TEMPLATE.csv', index=False)
    print("✓ Created data template: data/Datasheet_Filtered_CorrelationPlot_5_TEMPLATE.csv")
    
    # Instructions file
    instructions = """
# Data Setup Instructions

## Required Files

Place your data files in the `data/` directory with the following names:

### 1. Datasheet_Filtered_CorrelationPlot_5.csv
Your main NFA dataset with columns:
- SMILES: Molecular SMILES strings
- Diff_L-L+1: Energy difference
- E_b: Exciton binding energy
- Osc. strength: Oscillator strength
- Q20: Quadrupole moment
- A_EA: Electron affinity
- λ(nm): Absorption wavelength
- H-L Gap (eV): HOMO-LUMO gap

### 2. ewglist.csv (already created)
List of electron-withdrawing group SMILES for fragmentation

## Usage

1. Replace the template file with your actual data
2. Ensure all required columns are present
3. Run the pipeline scripts in order:
   - python Preprocessing.py
   - python Clustering.py
   - python joining.py
   - python DMPNN_pred.py

## Notes

- Template files are provided showing the expected format
- All output files will be created in results/ directory
- Log files will be saved in logs/ directory
- Trained models will be saved in models/ directory
"""
    
    with open('data/README_DATA_SETUP.txt', 'w') as f:
        f.write(instructions)
    print("✓ Created data setup instructions: data/README_DATA_SETUP.txt")

def main():
    """Main setup function."""
    print("Setting up NFA-ML-Discovery project structure...\n")
    
    # Create directories
    create_directories()
    print()
    
    # Create sample data files
    create_sample_ewg_list()
    create_sample_data_template()
    print()
    
    print("✅ Setup complete!")
    print("\nNext steps:")
    print("1. Read data/README_DATA_SETUP.txt for instructions")
    print("2. Replace template files with your actual data")
    print("3. Run the pipeline: Preprocessing.py → Clustering.py → joining.py → DMPNN_pred.py")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main()