
import pandas as pd
import random
import os
from rdkit import Chem
from rdkit.Chem import Draw
from IPython.display import display
from rdkit.Chem import AllChem
from rdkit.Chem import rdmolops

# Create necessary directories
os.makedirs('data', exist_ok=True)
os.makedirs('results', exist_ok=True)

class ADA:

    ewg_list=[]
    dfewg=pd.read_csv("data/ewglist.csv")
    for ewg in dfewg["SMILES"]:
      ewg_list.append(ewg)

    def __init__(self, molecule):
        """
        Initialize the ADA class with an RDKit molecule.

        Args:
            molecule (rdkit.Chem.Mol): An RDKit molecule object.
        """
        # Original Molecule
        self.molecule = molecule

        #Final Product
        self.acceptors = []
        self.acceptors_tags=[]
        self.donor_core = None
        self.donor_tags=[]
        self.side_chains=[]
        self.side_chains_tags=[]
        self.conjugated_extensions=[]
        self.conjugated_extensions_tags=[]
        self.donor_to_acceptor_tags=[]
        self.donor_to_side_chain_tags=[]
        self.donor_to_conjugated_tags=[]
        self.conjugated_to_acceptor_tags=[]
        self.conjugated_to_side_chain_tags=[]
        self.conjugated_to_donor_tags=[]

        #Preprocessing Variables
        self.atomic_nos = set(range(50, 119)) #For Tagging

        # Step 1: Shield EWG's and break all the other bonds attached to the donor
        self.mol1=None
        self.dont_break_bond_ids=set()

        # Step 2: Seperate donor core containing EWG from the other fragments
        self.fragments = []
        self.donor_frag = None

        # Step 3: Get True Donor from the donor fragment
        self.dont_break_atom_ids=set()
        self.donormol1=None
        self.donor_fragments=[]

        # Step 5: Seperating Donor Acceptor and Side Chain
        self.untagged_donor_core=None

        # Step 6: Seperating Conjugated Extensions and Aromatic Side Chains from Donor cores
        self.extensions=[]
        self.extensions_tags=[]

    # Step 1 Methods
    def dont_break_EWG(self):
      # print("In dont break EWG")
      for ewg in self.ewg_list:
        substructure_mol = Chem.MolFromSmiles(ewg)
        # Perform substructure search to find matches
        matches = self.molecule.GetSubstructMatches(substructure_mol)
        # Retrieve bond IDs for each match
        for match in matches:
            for bond in substructure_mol.GetBonds():
                # Map the bond indices from the substructure to the target molecule
                start_atom = match[bond.GetBeginAtomIdx()]
                end_atom = match[bond.GetEndAtomIdx()]
                bond = self.molecule.GetBondBetweenAtoms(start_atom, end_atom)
                if bond:
                    self.dont_break_bond_ids.add(bond.GetIdx())

    def break_ADA(self):
      # print("In break ADA")
      self.mol1= Chem.RWMol(self.molecule)
      atomic_no = random.choice(list(self.atomic_nos))
      self.atomic_nos.remove(atomic_no)
      for bond in self.molecule.GetBonds():
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        if bond.GetIdx() in self.dont_break_bond_ids:
          continue

        # Check if one atom is part of an aromatic ring and the other is not
        if (atom1.GetIsAromatic() or atom1.IsInRing()) != (atom2.GetIsAromatic() or atom2.IsInRing()):
            self.mol1.RemoveBond(atom1.GetIdx(),atom2.GetIdx())
            heavy_idx1 = self.mol1.AddAtom(Chem.Atom(atomic_no))
            heavy_idx2 = self.mol1.AddAtom(Chem.Atom(atomic_no))
            atomic_no = random.choice(list(self.atomic_nos))
            self.atomic_nos.remove(atomic_no)
            self.mol1.AddBond(atom1.GetIdx(), heavy_idx1, Chem.BondType.SINGLE)
            self.mol1.AddBond(atom2.GetIdx(), heavy_idx2, Chem.BondType.SINGLE)

    # Step 2 Methods
    def separate_donor_core(self,molecule,flag):
      # print("In separate donor core")
      self.fragments = rdmolops.GetMolFrags(molecule, asMols=True, sanitizeFrags=False)
      max_fused_ring=0
      max_cuts=0
      self.donor_frag=0
      for i,frag in enumerate(self.fragments):
        smiles=Chem.MolToSmiles(frag)
        mol = Chem.MolFromSmiles(smiles,sanitize=False)
        Chem.GetSymmSSSR(mol)
        ring_info = mol.GetRingInfo()
        count=0
        for ringIdx in range(len(ring_info.AtomRings())):
          if ring_info.IsRingFused(ringIdx):
            count+=1
        cuts=0
        for atom in mol.GetAtoms():
          atomic_no = atom.GetAtomicNum()
          if atomic_no > 49:
            cuts+=1
        if flag==0:
          if count>=max_fused_ring:
            if cuts>max_cuts:
              max_cuts=cuts
              max_fused_ring=count
              self.donor_frag=i
              self.donor_core=frag
          if cuts>max_cuts:
              max_cuts=cuts
              max_fused_ring=count
              self.donor_frag=i
              self.donor_core=frag
        if flag==1:
          if count>=max_fused_ring:
            max_fused_ring=count
            self.donor_frag=i
            self.donor_core=frag

    # Step 3 Methods
    def break_EWG_atoms(self):
      # print("In break EWG atoms")
      # self.dont_break_atom_ids={}
      for ewg in self.ewg_list:
        substructure_mol = Chem.MolFromSmiles(ewg)
        # Perform substructure search to find matches
        # img=Draw.MolToImage(self.fragments[self.donor_frag])
        # display(img)
        matches = self.fragments[self.donor_frag].GetSubstructMatches(substructure_mol)
        for i,match in enumerate(matches):
          # self.dont_break_atom_ids[(ewg,i)]=[]
          for atom in match:
            # self.dont_break_atom_ids[(ewg,i)].append(atom)  # Store atom indices
            self.dont_break_atom_ids.add(atom)

    def break_EWG(self):
      # print("In break EWG")
      self.donormol1 = Chem.RWMol(self.fragments[self.donor_frag])
      atomic_no = random.choice(list(self.atomic_nos))
      self.atomic_nos.remove(atomic_no)
      # print(self.dont_break_atom_ids)
      for bond in self.fragments[self.donor_frag].GetBonds():
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        if (atom1.GetIdx() in self.dont_break_atom_ids )!= (atom2.GetIdx() in self.dont_break_atom_ids):
            self.donormol1.RemoveBond(atom1.GetIdx(),atom2.GetIdx())
            heavy_idx1 = self.donormol1.AddAtom(Chem.Atom(atomic_no))
            heavy_idx2 = self.donormol1.AddAtom(Chem.Atom(atomic_no))
            atomic_no = random.choice(list(self.atomic_nos))
            self.atomic_nos.remove(atomic_no)
            self.donormol1.AddBond(atom1.GetIdx(), heavy_idx1, Chem.BondType.SINGLE)
            self.donormol1.AddBond(atom2.GetIdx(), heavy_idx2, Chem.BondType.SINGLE)
      # for atoms_attached_to_ewg in self.dont_break_atom_ids.values():
      #   for bond in self.fragments[self.donor_frag].GetBonds():
      #     atom1 = bond.GetBeginAtom()
      #     atom2 = bond.GetEndAtom()
      #     if atom1.GetIdx() in atoms_attached_to_ewg != atom2.GetIdx() in atoms_attached_to_ewg:
      #       self.donormol1.RemoveBond(atom1.GetIdx(),atom2.GetIdx())
      #       heavy_idx1 = self.donormol1.AddAtom(Chem.Atom(atomic_no))
      #       heavy_idx2 = self.donormol1.AddAtom(Chem.Atom(atomic_no))
      #       self.donormol1.AddBond(atom1.GetIdx(), heavy_idx1, Chem.BondType.SINGLE)
      #       self.donormol1.AddBond(atom2.GetIdx(), heavy_idx2, Chem.BondType.SINGLE)
      #   atomic_no = random.choice(list(self.atomic_nos))
      #   self.atomic_nos.remove(atomic_no)

    # Step 4 Methods
    def identify_donor(self):
      # print("In identify Donor")
      self.donor_fragments = rdmolops.GetMolFrags(self.donormol1, asMols=True, sanitizeFrags=False)
      max_fused_ring=0
      max_cuts=0
      # Initialize donor_core to first fragment to ensure it's never None
      if len(self.donor_fragments) > 0:
        self.donor_core = self.donor_fragments[0]

      for i,frag in enumerate(self.donor_fragments):
        smiles=Chem.MolToSmiles(frag)
        mol = Chem.MolFromSmiles(smiles,sanitize=False)
        Chem.GetSymmSSSR(mol)
        ring_info = mol.GetRingInfo()
        count=0
        for ringIdx in range(len(ring_info.AtomRings())):
          if ring_info.IsRingFused(ringIdx):
            count+=1
        cuts=0
        for atom in mol.GetAtoms():
          atomic_no = atom.GetAtomicNum()
          if atomic_no > 49:
            cuts+=1
        if count>=max_fused_ring:
          if cuts>max_cuts:
            max_cuts=cuts
            max_fused_ring=count
            self.donor_core=frag

    def untag_donor(self):
      # print("In untag Donor")
      # Create an editable molecule from the original molecule
      editable_mol = Chem.EditableMol(self.donor_core)

      # Identify atom indices to remove
      atom_indices_to_remove = [atom.GetIdx() for atom in self.donor_core.GetAtoms() if atom.GetAtomicNum() > 49]

      # Remove the atoms from the molecule (do it in reverse order to avoid indexing issues)
      for atom_idx in sorted(atom_indices_to_remove, reverse=True):
          editable_mol.RemoveAtom(atom_idx)
      self.untagged_donor_core=editable_mol.GetMol()

    def separate_acceptors_and_side_chains(self):
      # print("In sep acc and sc")

      for i,frag in enumerate(self.fragments):
        if i == self.donor_frag:
          continue
        flag=0
        for ewg in self.ewg_list:
          substructure_mol = Chem.MolFromSmiles(ewg)
          # Perform substructure search to find matches
          if frag.GetSubstructMatches(substructure_mol):
            self.acceptors.append(frag)
            flag=1
            break
        if flag==0:
          self.side_chains.append(frag)

    # Step 5 Methods
    def identify_acceptors_and_side_chains(self):
        # print("In id acc and sc")
        substructure_mol = self.untagged_donor_core
        dont_break=set()
        # Perform substructure search to find matches
        matches = self.molecule.GetSubstructMatches(substructure_mol)
        # Retrieve atom IDs for each match
        for match in matches:
          for atom in match:
            dont_break.add(atom)  # Store atom indices
        self.mol1 = Chem.RWMol(self.molecule)
        self.atomic_nos = set(range(50, 119)) #For Tagging
        atomic_no = random.choice(list(self.atomic_nos))
        self.atomic_nos.remove(atomic_no)
        for bond in self.molecule.GetBonds():
          atom1 = bond.GetBeginAtom()
          atom2 = bond.GetEndAtom()
          if (atom1.GetIdx() in dont_break )!= (atom2.GetIdx() in dont_break):
              self.mol1.RemoveBond(atom1.GetIdx(),atom2.GetIdx())
              heavy_idx1 = self.mol1.AddAtom(Chem.Atom(atomic_no))
              heavy_idx2 = self.mol1.AddAtom(Chem.Atom(atomic_no))
              atomic_no = random.choice(list(self.atomic_nos))
              self.atomic_nos.remove(atomic_no)
              self.mol1.AddBond(atom1.GetIdx(), heavy_idx1, Chem.BondType.SINGLE)
              self.mol1.AddBond(atom2.GetIdx(), heavy_idx2, Chem.BondType.SINGLE)

    # Step 6 Methods
    def is_part_of_fused_ring_system(self,atom_idx, ring_info):
      # print("In fused")
      # Get rings that the atom is part of
      atom_rings = [set(r) for r in ring_info.AtomRings() if atom_idx in r]

      # Check if the atom is in more than one ring
      if len(atom_rings) < 2:
          return False

      # Check if the rings share common atoms (fused)
      for i in range(len(atom_rings)):
          for j in range(i + 1, len(atom_rings)):
              if atom_rings[i].intersection(atom_rings[j]):
                  return True
      return False

    def separate_extensions(self):
      # print("In sep exten")
      # Get ring information
      Chem.GetSymmSSSR(self.donor_core)
      ring_info = self.donor_core.GetRingInfo()
      editable_donor= Chem.RWMol(self.donor_core)
      atomic_no = random.choice(list(self.atomic_nos))
      self.atomic_nos.remove(atomic_no)
      for bond in self.donor_core.GetBonds():
          atom1 = bond.GetBeginAtom()
          atom2 = bond.GetEndAtom()
          # Check if one atom is part of an aromatic ring and the other is not
          atom1_idx = atom1.GetIdx()
          atom2_idx = atom2.GetIdx()
          if atom1.GetAtomicNum()>49 or atom2.GetAtomicNum()>49:
            continue
          if not(self.is_part_of_fused_ring_system(atom1_idx, ring_info)) and not(self.is_part_of_fused_ring_system(atom2_idx, ring_info)):
              if not(Chem.rdchem.RingInfo.AreAtomsInSameRing(ring_info,atom1_idx,atom2_idx)):
                editable_donor.RemoveBond(atom1.GetIdx(),atom2.GetIdx())
                heavy_idx1 = editable_donor.AddAtom(Chem.Atom(atomic_no))
                heavy_idx2 = editable_donor.AddAtom(Chem.Atom(atomic_no))
                atomic_no = random.choice(list(self.atomic_nos))
                self.atomic_nos.remove(atomic_no)
                editable_donor.AddBond(atom1.GetIdx(), heavy_idx1, Chem.BondType.SINGLE)
                editable_donor.AddBond(atom2.GetIdx(), heavy_idx2, Chem.BondType.SINGLE)
      self.donor_core=editable_donor.GetMol()

    def tag_to_dict(self,molecule):
      # print("In tag dict")
      # Create a dictionary to hold atom indices and atomic numbers
      atom_dict = {}
      # Iterate over the atoms in the molecule
      for atom in molecule.GetAtoms():
          atomic_num = atom.GetAtomicNum()
          atom_index = atom.GetIdx()
          # Only add to the dictionary if atomic number is greater than 49
          if atomic_num > 49:
              atom_dict[atom_index] = atomic_num
      return atom_dict

    def store_tags(self):
      # print("In store tags")
      self.extensions=self.fragments
      for acceptor in self.acceptors:
        self.acceptors_tags.append(self.tag_to_dict(acceptor))
      for side_chain in self.side_chains:
        self.side_chains_tags.append(self.tag_to_dict(side_chain))
      self.donor_tags=self.tag_to_dict(self.donor_core)
      for frag in self.extensions:
        self.extensions_tags.append(self.tag_to_dict(frag))

    def check_tag(self,main_dict, list_of_dicts):
      # print("In check tags")
      # Extract values from the main dictionary
      main_values = set(main_dict.values())

      # Iterate over each dictionary in the list
      for i,d in enumerate(list_of_dicts):
          # Extract values from the current dictionary
          dict_values = set(d.values())
          # Check if there is an intersection between the two sets of values
          if main_values & dict_values:
              return i,True
      return -1,False

    def rejoin_bonds(self,fragment1,fragment2):
      # print("In rejoin bonds")
      # Combine the two fragments into a single editable molecule
      combined = Chem.CombineMols(fragment1, fragment2)
      editable_combined = Chem.EditableMol(combined)

      for tag in range(50,119):
        # Find the indices of the heavy atoms (Iodine in this case) in the combined molecule
        heavy_atom_indices = [atom.GetIdx() for atom in combined.GetAtoms() if atom.GetAtomicNum() == tag]

        # Check that exactly two heavy atoms were found
        if len(heavy_atom_indices) != 2:
            continue

        self.atomic_nos.add(tag)
        # Add a bond between the two heavy atoms' neighbors
        neighbor1 = [nbr.GetIdx() for nbr in combined.GetAtomWithIdx(heavy_atom_indices[0]).GetNeighbors()][0]
        neighbor2 = [nbr.GetIdx() for nbr in combined.GetAtomWithIdx(heavy_atom_indices[1]).GetNeighbors()][0]
        editable_combined.AddBond(neighbor1, neighbor2, order=Chem.rdchem.BondType.SINGLE)

        # Remove the heavy atoms from the molecule
        editable_combined.RemoveAtom(heavy_atom_indices[1])
        editable_combined.RemoveAtom(heavy_atom_indices[0])

      # Get the final molecule
      final_molecule = editable_combined.GetMol()
      return final_molecule

    def classify_extensions(self):
      # print("In classify exten")
      for i,frag in enumerate(self.extensions):
        if i==self.donor_frag:
          continue
        if self.check_tag(self.extensions_tags[i], self.acceptors_tags)[1]:
          self.conjugated_extensions.append(frag)
          continue
        side_chain_check=self.check_tag(self.extensions_tags[i],self.side_chains_tags)
        if side_chain_check[1]:
          self.side_chains[side_chain_check[0]]=self.rejoin_bonds(frag,self.side_chains[side_chain_check[0]])
          self.side_chains_tags[side_chain_check[0]]=self.tag_to_dict(self.side_chains[side_chain_check[0]])
        else:
          self.side_chains.append(frag)
          self.side_chains_tags.append(self.extensions_tags[i])

    def correct_conjugated_extensions(self):
      incorrect_side_chains=[]
      for mol in self.conjugated_extensions:
        self.conjugated_extensions_tags.append(self.tag_to_dict(mol))
      self.donor_tags=self.tag_to_dict(self.donor_core)
      for i,side_chain_tag in enumerate(self.side_chains_tags):
        donor_tags=[]
        donor_tags.append(self.donor_tags)
        conjugated_check=self.check_tag(side_chain_tag,self.conjugated_extensions_tags)
        donor_check=self.check_tag(side_chain_tag,donor_tags)
        if conjugated_check[0]!=-1 and donor_check[0]!=-1:
          self.conjugated_extensions[conjugated_check[0]]=self.rejoin_bonds(self.side_chains[i],self.conjugated_extensions[conjugated_check[0]])
          self.conjugated_extensions_tags[conjugated_check[0]]=self.tag_to_dict(self.conjugated_extensions[conjugated_check[0]])
          incorrect_side_chains.append(i)
      for i in sorted(incorrect_side_chains, reverse=True):
          del self.side_chains[i]
          del self.side_chains_tags[i]

    def untag(self,mol):
      # Create an editable molecule from the original molecule
      editable_mol = Chem.EditableMol(mol)

      # Identify atom indices to remove
      atom_indices_to_remove = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 49]

      # Remove the atoms from the molecule (do it in reverse order to avoid indexing issues)
      for atom_idx in sorted(atom_indices_to_remove, reverse=True):
          editable_mol.RemoveAtom(atom_idx)
      return Chem.MolToSmiles(editable_mol.GetMol())

    def change_acceptor_tags(self):
      tag=0
      for i,acceptor in enumerate(self.acceptors):
        if len(self.acceptors_tags[i])>=2:
          tag = random.choice(list(self.acceptors_tags[i].values()))
        else:
          continue
        for atom in self.donor_core.GetAtoms():
          if atom.GetAtomicNum() in self.acceptors_tags[i].values():
            atom.SetAtomicNum(tag)
        for extension in self.conjugated_extensions:
          for atom in extension.GetAtoms():
            if atom.GetAtomicNum() in self.acceptors_tags[i].values():
              atom.SetAtomicNum(tag)

    def store_final_tags(self):
      self.acceptors_tags=[]
      self.side_chains_tags=[]
      self.donor_tags=[]
      self.conjugated_extensions_tags=[]
      for acceptor in self.acceptors:
        self.acceptors_tags.append(self.tag_to_dict(acceptor))
      for side_chain in self.side_chains:
        self.side_chains_tags.append(self.tag_to_dict(side_chain))
      self.donor_tags=self.tag_to_dict(self.donor_core)
      for frag in self.conjugated_extensions:
        self.conjugated_extensions_tags.append(self.tag_to_dict(frag))

      untagged_side_chains=[]
      for side_chain in self.side_chains:
        untagged_side_chains.append(self.untag(side_chain))
      untagged_side_chains = list(set(untagged_side_chains))
      for smiles in untagged_side_chains:
        mol=Chem.MolFromSmiles(smiles,sanitize=False)
        tag=0
        for side_chain in self.side_chains:
          if side_chain.GetSubstructMatch(mol):
            if tag==0:
              for atom in side_chain.GetAtoms():
                if atom.GetAtomicNum()>49:
                  tag=atom.GetAtomicNum()
                  break
            else:
              for atom in side_chain.GetAtoms():
                if atom.GetAtomicNum()>49:
                  for at in self.donor_core.GetAtoms():
                    if at.GetAtomicNum()==atom.GetAtomicNum():
                      at.SetAtomicNum(tag);
                  for acceptor in self.acceptors:
                    for at in acceptor.GetAtoms():
                      if at.GetAtomicNum()==atom.GetAtomicNum():
                        at.SetAtomicNum(tag);
                  for conjugated_extension in self.conjugated_extensions:
                    for at in conjugated_extension.GetAtoms():
                      if at.GetAtomicNum()==atom.GetAtomicNum():
                        at.SetAtomicNum(tag);
                  atom.SetAtomicNum(tag)

      self.acceptors_tags=[]
      self.side_chains_tags=[]
      self.donor_tags=[]
      self.conjugated_extensions_tags=[]
      for acceptor in self.acceptors:
        self.acceptors_tags.append(self.tag_to_dict(acceptor))
      for side_chain in self.side_chains:
        self.side_chains_tags.append(self.tag_to_dict(side_chain))
      self.donor_tags=self.tag_to_dict(self.donor_core)
      for frag in self.conjugated_extensions:
        self.conjugated_extensions_tags.append(self.tag_to_dict(frag))

      for key, value in self.donor_tags.items():
        tdict={}
        tdict[key]=value
        acceptor_check=self.check_tag(tdict,self.acceptors_tags)
        if acceptor_check[1]:
          self.donor_to_acceptor_tags.append(value)
        side_chain_check=self.check_tag(tdict,self.side_chains_tags)
        if side_chain_check[1]:
          self.donor_to_side_chain_tags.append(value)
        conjugated_chain_check=self.check_tag(tdict,self.conjugated_extensions_tags)
        if conjugated_chain_check[1]:
          self.donor_to_conjugated_tags.append(value)

      for dictionary in self.conjugated_extensions_tags:
        conjugated_acceptor=[]
        conjugated_donor=[]
        conjugated_sc=[]
        for key, value in dictionary.items():
          tdict={}
          tdict[key]=value
          acceptor_check=self.check_tag(tdict,self.acceptors_tags)
          if acceptor_check[1]:
           conjugated_acceptor.append(value)
          side_chain_check=self.check_tag(tdict,self.side_chains_tags)
          if side_chain_check[1]:
            conjugated_sc.append(value)
          donor_tags=[self.donor_tags]
          conjugated_chain_check=self.check_tag(tdict,donor_tags)
          if conjugated_chain_check[1]:
            conjugated_donor.append(value)
        self.conjugated_to_acceptor_tags.append(conjugated_acceptor)
        self.conjugated_to_donor_tags.append(conjugated_donor)
        self.conjugated_to_side_chain_tags.append(conjugated_sc)
        for i,extension in enumerate(self.conjugated_to_donor_tags):
          for side_chain in self.conjugated_to_side_chain_tags:
            self.conjugated_to_donor_tags[i] = [tag for tag in extension if tag not in side_chain]

      for side_chain in self.side_chains:
        for atom in side_chain.GetAtoms():
            if atom.GetAtomicNum()>49:
              atom.SetAtomicNum(49)
      self.side_chains_tags=[]
      for side_chain in self.side_chains:
        self.side_chains_tags.append(self.tag_to_dict(side_chain))
      for acceptor in self.acceptors:
        for atom in acceptor.GetAtoms():
            if atom.GetAtomicNum()>49:
              atom.SetAtomicNum(48)
      self.acceptors_tags=[]
      for acceptor in self.acceptors:
        self.acceptors_tags.append(self.tag_to_dict(acceptor))

df=pd.read_csv("data/Datasheet-I.csv")
ADA_objects=[]
for i,smiles in enumerate(df["SMILES"]):
  if i==114:
    continue
  ADA_objects.append(ADA(Chem.MolFromSmiles(smiles,sanitize=False)))

count = 0
anamol_list=[53,55,60,62,63,91,99,100,101]
adda_list=[161,268,269,270,271,272,273,274,277]
mol_list=[]
print(f"Processing {len(ADA_objects)} ADA molecules...")
for i,ADA_object in enumerate(ADA_objects):
  if (i + 1) % 50 == 0:  # Progress indicator every 50 molecules
    print(f"Processed {i + 1}/{len(ADA_objects)} molecules...")
  # if i not in anamol_list:
  #   continue
  # if i not in adda_list:
  #   continue
  # print(i)  # Commented out for faster execution
  # print("Original")
  # img=Draw.MolToImage(ADA_object.molecule)
  # display(img)
  ADA_object.dont_break_EWG()
  ADA_object.break_ADA()
  ADA_object.separate_donor_core(ADA_object.mol1,0)
  # print("All")
  # img=Draw.MolToImage(ADA_object.mol1)
  # display(img)
  # print("Donor")
  # img=Draw.MolToImage(ADA_object.donor_core)
  # display(img)
  ADA_object.break_EWG_atoms()
  ADA_object.break_EWG()
  ADA_object.identify_donor()
  # print("Donor Core")
  # img=Draw.MolToImage(ADA_object.donor_core)
  # display(img)
  ADA_object.untag_donor()
  ADA_object.identify_acceptors_and_side_chains()
  ADA_object.separate_donor_core(ADA_object.mol1,0)
  ADA_object.separate_acceptors_and_side_chains()
  ADA_object.separate_extensions()
  # for i,donor_core in enumerate(ADA_object.donor_core):
  ADA_object.separate_donor_core(ADA_object.donor_core,1)
  # print("Donor Core")
  # img=Draw.MolToImage(ADA_object.donor_core)
  # display(img)
  ADA_object.store_tags()
  ADA_object.classify_extensions()
  ADA_object.correct_conjugated_extensions()
  ADA_object.change_acceptor_tags()
  ADA_object.store_final_tags()
  # print(ADA_object.donor_to_acceptor_tags)
  # print(ADA_object.donor_to_side_chain_tags)
  # print(ADA_object.donor_to_conjugated_tags)
  if len(ADA_object.acceptors)==2:
    count+=1
    mol_list.append(i)

# print(count)  # Commented out for faster execution
# print(mol_list)  # Commented out for faster execution


# Commented out visualization section for faster execution
# for i,ADA_object in enumerate(ADA_objects):
#   if len(ADA_object.acceptors)!=2:
#     continue
#   if i==65:
#     continue
#   # if i not in anamol_list:
#   #   continue
#   print(i)
#   # print("Original")
#   # img=Draw.MolToImage(ADA_object.molecule)
#   # display(img)
#   # print("Full Segmentation")
#   # img=Draw.MolToImage(ADA_object.mol1)
#   # display(img)
#   # # print("Donor with EWG")
#   # # img=Draw.MolToImage(ADA_object.fragments[ADA_object.donor_frag])
#   # # display(img)
#   print("Donor Core")
#   img=Draw.MolToImage(ADA_object.donor_core)
#   display(img)
#   print("Acceptors")
#   # print("Acceptors")
#   for i,acceptor in enumerate(ADA_object.acceptors):
#     img=Draw.MolToImage(acceptor)
#     display(img)
#     print(i)
#     if(len(ADA_object.acceptors_tags[i])>2):
#       img=Draw.MolToImage(acceptor)
#       display(img)
#       print(ADA_object.acceptors_tags[i])
#   print(i)
#   print(ADA_object.donor_tags)
#   print("Donor")
#   img=Draw.MolToImage(ADA_object.donor_core)
#   display(img)
#   print("Side Chains")
#   for sidechain in ADA_object.side_chains:
#     img=Draw.MolToImage(sidechain)
#     display(img)
#   print("Conjugated Extensions")
#   for conjugated_extension in ADA_object.conjugated_extensions:
#     img=Draw.MolToImage(conjugated_extension)
#     display(img)

import csv
import json

# File path
output_file = 'data/ADA_Data_With_Tags_filtered.csv'

# Write headers to CSV
def create_csv_with_headers(file_path):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["Molecule No", "Original Molecule Smiles", "Donor Core Smiles", "Acceptors Smiles", "Side Chains Smiles", "Conjugated Extensions Smiles","Donor to Acceptor Tags","Donor to Side Chain Tags","Donor to Conjugated Extension Tags","Conjugated Extension to Donor Tags","Conjugated Extension to Acceptors Tags","Conjugated Extension to Side Chains Tags","Diff_L-L+1","E_B","Osc. strength","Q20","EA","lambda","HL Gap"])

# Call this function once to create the file with headers
create_csv_with_headers(output_file)

# Function to append a single entry to the CSV file
def append_entry_to_csv(file_path, entry):
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            entry["Molecule No"],
            entry["Original Molecule Smiles"],
            entry["Donor Core Smiles"],
            json.dumps(entry["Acceptors Smiles"]),
            json.dumps(entry["Side Chains Smiles"]),
            json.dumps(entry["Conjugated Extensions Smiles"]),
            json.dumps(entry["Donor to Acceptor Tags"]),
            json.dumps(entry["Donor to Side Chain Tags"]),
            json.dumps(entry["Donor to Conjugated Extension Tags"]),
            json.dumps(entry["Conjugated Extension to Donor Tags"]),
            json.dumps(entry["Conjugated Extension to Acceptors Tags"]),
            json.dumps(entry["Conjugated Extension to Side Chain Tags"]),
            entry["Diff_L-L+1"],
            entry["E_B"],
            entry["Osc. strength"],
            entry["Q20"],
            entry["A_EA"],
            entry["\u03BB(nm)"],
            entry["H-L Gap (eV)"]
        ])
df=pd.read_csv("data/Datasheet-I.csv")
E_B=[]
Diff=[]
Osc=[]
Q20=[]
EA=[]
lamb=[]
HL=[]
for i, smiles in enumerate(df["SMILES"]):
  if i==114:
    continue
  row = df[df['SMILES'] == smiles]
  Diff.append(row["Diff_L-L+1"].values[0])
  E_B.append(row["E_b"].values[0])
  Osc.append(row["Osc. strength"].values[0])
  Q20.append(row["Q20"].values[0])
  EA.append(row["A_EA"].values[0])
  lamb.append(row["\u03BB(nm)"].values[0])
  HL.append(row["H-L Gap (eV)"].values[0])

# Example of adding entries one by one
print(f"Processing {len(ADA_objects)} molecules for CSV export...")
processed_count = 0
for i,ADA_object in enumerate(ADA_objects):
  if len(ADA_object.acceptors)!=2:
    continue
  if i==91:
    continue
  processed_count += 1
  if processed_count % 10 == 0:  # Progress indicator every 10 molecules
    print(f"Processed {processed_count} molecules...")
  new_entry={}
  new_entry["Molecule No"]=i+1
  new_entry["Original Molecule Smiles"]=Chem.MolToSmiles(ADA_object.molecule)
  new_entry["Donor Core Smiles"] = Chem.MolToSmiles(ADA_object.donor_core)
  new_entry["Acceptors Smiles"] = [Chem.MolToSmiles(mol) for mol in ADA_object.acceptors]
  new_entry["Side Chains Smiles"] = [Chem.MolToSmiles(mol) for mol in ADA_object.side_chains]
  new_entry["Conjugated Extensions Smiles"] = [Chem.MolToSmiles(mol) for mol in ADA_object.conjugated_extensions]
  new_entry["Donor to Acceptor Tags"] = ADA_object.donor_to_acceptor_tags
  new_entry["Donor to Side Chain Tags"] = ADA_object.donor_to_side_chain_tags
  new_entry["Donor to Conjugated Extension Tags"] = ADA_object.donor_to_conjugated_tags
  new_entry["Conjugated Extension to Acceptors Tags"] = ADA_object.conjugated_to_acceptor_tags
  new_entry["Conjugated Extension to Side Chain Tags"] = ADA_object.conjugated_to_side_chain_tags
  new_entry["Conjugated Extension to Donor Tags"] = ADA_object.conjugated_to_donor_tags
  new_entry["Diff_L-L+1"] = Diff[i]
  new_entry["E_B"] = E_B[i]
  new_entry["Osc. strength"] = Osc[i]
  new_entry["Q20"] = Q20[i]
  new_entry["A_EA"] = EA[i]
  new_entry["\u03BB(nm)"] = lamb[i]
  new_entry["H-L Gap (eV)"] = HL[i]

  # Append entries to the CSV file
  append_entry_to_csv(output_file, new_entry)

print(f"Successfully processed {processed_count} molecules and saved to {output_file}")
print("Tag creation completed successfully!")