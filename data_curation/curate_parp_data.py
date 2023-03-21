"""
Functions for curating PARP1/2 data from ChEMBL, GoStar and whereever else we can get it.
"""

import os
import sys
import pdb
# requires data_science/code to be in sys.path
import chembl
import urllib

import pandas as pd
import numpy as np
from rdkit import Chem
from atomsci.ddm.utils.struct_utils import mols_from_smiles, rdkit_smiles_from_smiles, base_smiles_from_smiles
from atomsci.ddm.utils.curate_data import freq_table, remove_outlier_replicates, aggregate_assay_data
from atomsci.ddm.utils import data_curation_functions as dcf
from atomsci.ddm.utils import rdkit_easy as rdk
#from atomsci.glo_spl.utils import visualize_gmd_results as vgr

import matplotlib.pyplot as plt
import seaborn as sns

parp_dir = "/usr/workspace/atom/PARP_compounds"
dset_dir = f"{parp_dir}/Datasets_and_Models"
selectivity_dir = f"{dset_dir}/PARP1_selectivity"

# ChEMBL target IDs for *human* PARP1 and PARP2. Unlike GoStar, ChEMBL only has species-specific target records.
chembl_target_id = dict(PARP1='CHEMBL3105', PARP2='CHEMBL5366', CYP2C9='CHEMBL3397')

# ---------------------------------------------------------------------------------------------------------------------------------
def get_raw_chembl_parp_activity_data(target='PARP1', activity_type='IC50', db_version='30', force_update=False):
    """
    Query ChEMBL for measurements of PARP1 or PARP2 activity of the given type.
    """
    chembl_dir = f"{dset_dir}/{target}/chembl"
    os.makedirs(chembl_dir, exist_ok=True)
    raw_file = f"{chembl_dir}/{target}_{activity_type}_chembl{db_version}.csv"
    if os.path.isfile(raw_file) and not force_update:
        print(f"Using existing raw {target} {activity_type} data from ChEMBL {db_version}")
        raw_df = pd.read_csv(raw_file)
        return raw_df

    print(f"Querying raw {target} {activity_type} data from ChEMBL {db_version}")
    with chembl.connect(version=db_version) as con:
        sql = ' '.join([
                     "SELECT DISTINCT act.activity_id, mol.chembl_id AS mol_chembl_id, cr.compound_name,",
                     "a.chembl_id AS assay_chembl_id, a.assay_type, a.assay_cell_type, a.assay_organism,",
                     "t.pref_name, a.description, act.standard_type,",
                     "act.standard_relation, act.standard_value, act.standard_units, cs.canonical_smiles,",
                     "doc.chembl_id AS doc_chembl_id, doc.title, doc.journal, doc.year, doc.volume, doc.issue,",
                     "doc.first_page, doc.last_page, doc.doi, doc.pubmed_id, doc.patent_id", 
                     "FROM assays a, activities act, compound_records cr, compound_structures cs,",
                     "target_dictionary t, molecule_dictionary mol, docs doc",
                     "WHERE a.tid = t.tid",
                     f"AND t.chembl_id = '{chembl_target_id[target]}'",
                     "AND act.assay_id = a.assay_id",
                     f"AND act.standard_type = '{activity_type}'",
                     "AND act.standard_value IS NOT NULL",
                     "AND act.standard_units IS NOT NULL",
                     "AND act.molregno = cr.molregno",
                     "AND act.molregno = mol.molregno",
                     "AND act.doc_id = doc.doc_id",
                     "AND cs.molregno = cr.molregno"])
        raw_df = pd.read_sql(sql, con=con)
        raw_df = raw_df.drop_duplicates(subset='activity_id')

        # Add a reference column formatted the same way as in GoStar to facilitate checking for duplicates
        references = []
        for row in raw_df.itertuples():
            if row.volume is None:
                volume = ''
            elif type(row.volume) == str:
                volume = f"{float(row.volume):.0f}"
            else:
                volume = f"{row.volume:.0f}"
            if row.issue is None:
                issue = ''
            elif type(row.issue) == str:
                issue = f"{float(row.issue):.0f}"
            else:
                issue = f"{row.issue:.0f}"
            if row.journal is None:
                journal = ''
            else:
                journal = row.journal.replace('.', '')
            if not (row.patent_id is None):
                references.append(row.patent_id)
            elif not (row.year is None):
                references.append(f"{journal}, {row.year:.0f}, {volume} ({issue}), {row.first_page}-{row.last_page}")
            else:
                references.append("")
        raw_df['reference'] = references

        # Add a salt-stripped RDKit SMILES column. We only do that here so we can count unique compounds.
        raw_df['base_rdkit_smiles'] = base_smiles_from_smiles(raw_df.canonical_smiles.values.tolist(), workers=16)
        raw_df = raw_df[raw_df.base_rdkit_smiles != '']

        # Map ChEMBL column names to corresponding GoStar columns, so we can combine tables later
        raw_df = raw_df.rename(columns={
                        'mol_chembl_id': 'compound_id',
                        'assay_chembl_id': 'assay_id',
                        'assay_cell_type': 'cells_cellline_organ',
                        'assay_organism': 'species',
                        'pref_name': 'standard_name',
                        'description': 'enzyme_cell_assay',
                        'standard_type': 'activity_type',
                        'standard_relation': 'activity_prefix',
                        'standard_value': 'activity_value',
                        'standard_units': 'activity_uom',
                        'canonical_smiles': 'original_smiles',
                        'doc_chembl_id': 'ref_id',
                    })

        raw_df.to_csv(raw_file, index=False)
        print(f"Wrote {target} {activity_type}s to {raw_file}")

        # Make a table of unique assays with record and compound counts
        fr_df = freq_table(raw_df, 'assay_id').rename(columns={'Count': 'assay_records'})
        fr_df['assay_cmpds'] = [len(set(raw_df.loc[raw_df.assay_id == id, 'base_rdkit_smiles'].values)) 
                                    for id in fr_df.assay_id.values]
        assay_df = raw_df.drop_duplicates(subset='assay_id')
        assay_df = assay_df.merge(fr_df, how='left', on='assay_id')
        assay_file = f"{chembl_dir}/{target}_{activity_type}_chembl{db_version}_assays.csv"
        assay_df.to_csv(assay_file, index=False)
        print(f"Wrote {target} assays to {assay_file}")

        # Do the same for documents (references)
        fr_df = freq_table(raw_df, 'ref_id').rename(columns={'Count': 'ref_records'})
        fr_df['ref_cmpds'] = [len(set(raw_df.loc[raw_df.ref_id == id, 'base_rdkit_smiles'].values)) 
                                    for id in fr_df.ref_id.values]
        ref_df = raw_df.drop_duplicates(subset='ref_id')
        ref_df = ref_df.merge(fr_df, how='left', on='ref_id')
        ref_file = f"{chembl_dir}/{target}_{activity_type}_chembl{db_version}_refs.csv"
        ref_df.to_csv(ref_file, index=False)
        print(f"Wrote {target} references to {ref_file}")
        print(f"{target}: {len(raw_df)} activities, {len(set(raw_df.base_rdkit_smiles.values))} compounds, {len(assay_df)} assays, {len(ref_df)} references")

        return raw_df

# ---------------------------------------------------------------------------------------------------------------------------------
def get_raw_gostar_parp_activity_data(target='PARP1', activity_type='IC50', db_version='2022-06-20', force_update=False):
    """
    Read data from querying specified version of GoStar for PARP1 or PARP2 activity data, allowing
    "source" (species) to be either Human or NULL. Filter the data to include only records with the
    given activity type (IC50 or Ki), convert activities to pIC50 or pKi, and write it to a file.
    """
    raw_file = f"{dset_dir}/{target}/gostar/{target}_{activity_type}_gostar_{db_version}.csv"
    if os.path.isfile(raw_file) and not force_update:
        print(f"Using existing raw {target} {activity_type} data from GoStar {db_version}")
        raw_df = pd.read_csv(raw_file)
        return raw_df

    # Read data obtained by querying against standard PARP1 and PARP2 target names
    all_types_df = pd.read_csv(f"{dset_dir}/{target}/gostar/{target}_activity_gostar_{db_version}.csv")

    # Standardize abbreviations in references to not end with periods. Leave URL references alone since dots are significant.
    all_types_df['reference'] = [ref if ref.startswith('http') else ref.replace('.', '') for ref in all_types_df.reference.values]

    # Standardize SMILES strings and exclude records with SMILES that RDKit doesn't like
    all_types_df['base_rdkit_smiles'] = base_smiles_from_smiles(all_types_df.sub_smiles.values.tolist(), workers=16)
    all_types_df = all_types_df[all_types_df.base_rdkit_smiles != '']

    # Filter the data by activity type; only include IC50/pIC50 or Ki/pKi measurements.
    act_types = dict(IC50=['IC50', 'pIC50'], Ki=['Ki', 'pKi', 'inhibition constant (Ki)'])
    sel_types = act_types[activity_type]
    sel_units = ['nM', 'nmol/L', 'uM']
    raw_df = all_types_df[all_types_df.activity_type.isin(sel_types) & 
                   (all_types_df.activity_uom.isna() | all_types_df.activity_uom.isin(sel_units))].copy()

    # Standardize some column names
    raw_df = raw_df.rename(columns={
                        'act_id': 'activity_id',
                        'gvk_id': 'compound_id',
                        'source': 'species',
                        'sub_smiles': 'original_smiles',
                        })
    # Force compound IDs to be read as strings
    raw_df['compound_id'] = [f"gvk_{id}" for id in raw_df.compound_id.values]



    print(f"GoStar {activity_type} data for {target} has {len(raw_df)} rows, {len(set(raw_df.base_rdkit_smiles.values))} compounds")

    # Write filtered and converted data to a file
    raw_df.to_csv(raw_file, index=False)
    print(f"Wrote {target} p{activity_type} data to {raw_file}")

    return raw_df

# ---------------------------------------------------------------------------------------------------------------------------------
def compare_gostar_activities_between_versions(target='PARP1', old_db_version='2022-05-06', new_db_version='2022-06-20'):
    """
    Filter raw PARP data by activity type from the specified versions of the GoStar DB and determine what records are
    new in the newer version compared to the old one.
    """
    for activity_type in ['IC50', 'Ki']:
        old_act_df = get_raw_gostar_parp_activity_data(target, activity_type, old_db_version)
        new_act_df = get_raw_gostar_parp_activity_data(target, activity_type, new_db_version)
        old_activity_ids = set(old_act_df.activity_id.values)
        new_activity_ids = set(new_act_df.activity_id.values)
        old_smiles = set(old_act_df.base_rdkit_smiles.values)
        new_smiles = set(new_act_df.base_rdkit_smiles.values)
        dropped_act_ids = old_activity_ids - new_activity_ids
        added_act_ids = new_activity_ids - old_activity_ids
        added_smiles = new_smiles - old_smiles
        if len(dropped_act_ids) > 0:
            print(f"{len(dropped_act_ids)} {activity_type} records were dropped for {target}")
        if len(added_act_ids) > 0:
            print(f"{len(added_act_ids)} {activity_type} records were added for {target} with {len(added_smiles)} new compounds")
            added_df = new_act_df[new_act_df.activity_id.isin(added_act_ids)]
            added_file = f"{dset_dir}/{target}/gostar/{target}_new_{activity_type}_records_gostar_{new_db_version}.csv"
            added_df.to_csv(added_file, index=False)
            print(f"Wrote new {activity_type} records to {added_file}")


# ---------------------------------------------------------------------------------------------------------------------------------
def curate_parp_activity_data(raw_df=None, target='PARP1', activity_type='IC50', db='gostar', db_version='2022-06-20', force_update=False):
    """
    Filter raw GoStar or ChEMBL IC50 or Ki data for PARP1 or PARP2 to exclude some phenotypic assays that don't belong
    here. Convert activities to pIC50s or pKis and standardize relational operators.
    """
    db_version = str(db_version)
    target_dir = f"{dset_dir}/{target}/{db}"
    excluded_file = f"{target_dir}/{target}_p{activity_type}_excluded_{db}_{db_version}.csv"
    filtered_file = f"{target_dir}/{target}_p{activity_type}_filtered_{db}_{db_version}.csv"
    if (not force_update) and os.path.isfile(filtered_file):
        filtered_df = pd.read_csv(filtered_file)
        return filtered_df
    if raw_df is None:
        if db == 'chembl':
            raw_df = get_raw_chembl_parp_activity_data(target=target, activity_type=activity_type, db_version=db_version,
                                                       force_update=force_update)
        elif db == 'gostar':
            raw_df = get_raw_gostar_parp_activity_data(target=target, activity_type=activity_type, db_version=db_version,
                                                       force_update=force_update)
        elif db == 'ucsf':
            _ = filter_ucsf_data(db=db, db_version=db_version)
            return pd.read_csv(filtered_file)


    # Correct GoStar data from patent WO 99/11624 A1, in which IC50 units were given for some compounds but not others. 
    raw_df.loc[raw_df.reference == 'WO 99/11624 A1', 'activity_uom'] = 'uM'

    # Correct GoStar data from patent WO 99/11624 A1, in which IC50 units were missing. By comparing against a paper by the
    # patent authors about one of the compounds, I determined the units were nM.
    raw_df.loc[raw_df.reference == 'WO 2017/223516 A1', 'activity_uom'] = 'nM'

    # Correct GoStar data from reference Bioorg Chem, 2020, 102 (), 104075, in which pIC50 values reported in supplemental
    # table S1 are calculated based on IC50s expressed in micromolar rather than molar units (i.e., they are off by 6).
    # Most of these are duplicates because the data are derived from the literature.
    raw_df.loc[(raw_df.reference == 'Bioorg Chem, 2020, 102 (), 104075') & (raw_df.activity_type == 'pIC50'), 'activity_value'] += 6

    # Correct the same issue with pIC50's from Bioorg. Med. Chem., 2005, 13 (4), 1151-1157
    raw_df.loc[(raw_df.reference == 'Bioorg Med Chem, 2005, 13 (4), 1151-1157') & (raw_df.activity_type == 'pIC50'), 'activity_value'] += 6

    # Remove data from reference Proteins (101002/prot26077), 2021, xx (xx), xxx-xxx. It duplicates data from
    # Eur J Med Chem, 2018, 145 (), 389-403, but reports it in the wrong units.
    raw_df = raw_df[raw_df.reference != 'Proteins (101002/prot26077), 2021, xx (xx), xxx-xxx']

    # Remove data from patent WO 2007/138355 A1, which is all left-censored at 5 uM; there is newer data for the same compounds
    # in Bioorg Med Chem Lett, 2010, 20 (3), 1094-1099.
    raw_df = raw_df[raw_df.reference != 'WO 2007/138355 A1']

    # Remove data from patent WO 03/014121 A1 where IC50 is reported as equal to 20 uM; these values are probably right-censored
    # since 20 uM was the highest concentration in their assay; also, the structures GoStar came up with for these are questionable.
    raw_df = raw_df[~((raw_df.reference == 'WO 03/014121 A1') & (raw_df.activity_value >= 20.0))]


    # Remove any other records for which units are blank and activity type is not pIC50 or pKi
    raw_df = raw_df[~(raw_df.activity_uom.isna() & ~raw_df.activity_type.isin(['pIC50', 'pKi']))]
    # Remove records with zero or negative IC50 or Ki values, which are not physical
    raw_df = raw_df[raw_df.activity_value > 0].copy()

    # Make copies of the original activity types, prefixes, values and units, before we overwrite the columns with negative log
    # transformed values
    raw_df['orig_activity_type'] = raw_df.activity_type.values
    raw_df['orig_activity_prefix'] = raw_df.activity_prefix.values
    raw_df['orig_activity_value'] = raw_df.activity_value.values
    raw_df['orig_activity_uom'] = raw_df.activity_uom.values

    # Convert activity values to their negative log counterparts (pIC50 and pKi), taking the activity units into account
    raw_df.loc[raw_df.activity_uom.isin(['nM', 'nmol/L']), 'activity_value'] = 9.0 - np.log10(
                                                  raw_df.loc[raw_df.activity_uom.isin(['nM', 'nmol/L']), 'activity_value'].values)
    raw_df.loc[raw_df.activity_uom == 'uM', 'activity_value'] = 6.0 - np.log10(
                                                  raw_df.loc[raw_df.activity_uom == 'uM', 'activity_value'].values)

    # Standardize the relational operators. For activity types that were already negative log values (pIC50 or pKi, 
    # indicated by null units), keep the same relationships; otherwise, invert them.
    neglog_df = raw_df[raw_df.activity_uom.isna()].copy()
    neglog_df = dcf.standardize_relations(neglog_df, rel_col='activity_prefix', output_rel_col='relation')

    needs_inv_df = raw_df[~raw_df.activity_uom.isna()].copy()
    needs_inv_df = dcf.standardize_relations(needs_inv_df, rel_col='activity_prefix', output_rel_col='relation', invert=True)
    needs_inv_df['activity_type'] = f"p{activity_type}"

    raw_df = pd.concat([neglog_df, needs_inv_df], ignore_index=True)

    # Everything is pIC50 or pKi now, so remove the units
    raw_df['activity_uom'] = np.nan

    # Add URLs for Google Scholar or Google Patents queries on the references to make it easier to run searches from Excel
    # versions of the output files (using the HYPERLINK function)
    ref_urls = []
    for ref in raw_df.reference.values:
        query = urllib.parse.quote_plus(ref)
        if ref.startswith('US') or ref.startswith('WO'):
            ref_urls.append(f"https://patents.google.com?oq={query}")
        else:
            ref_urls.append(f"https://scholar.google.com/scholar?hl=en&as_sdt=0&q={query}&btnG=")
    raw_df['ref_search_url'] = ref_urls

    # Filter on assay description and cell type: 
    #   - We exclude data from assays with 'domain' in the description because we only want data for inhibition of the 
    #     full enzyme, not just the catalytic or other domain fragment.
    #   - We exclude most data with non-blanks in the cell_cellline_organ column because most of these are measurements
    #     of effects on cellular survival or proliferation. An exception is when an insect cell line like Sf9 or Sf21
    #     is (incorrectly) listed in this column; in this case the cells are used to express the PARP proteins, not 
    #     as part of the inhibition assay. A more complicated case is HeLa cells; sometimes they are the source of
    #     the PARP enzyme, but other times the assay is conducted using HeLa cells, so we have to use keywords in the
    #     assay description that exclude the latter but not the former.
    #     There are probably other cell types that appear because they are the source of the PARP enzyme used in the assay,
    #     rather than part of the assay itself, but they only represent a small set of compounds so it's not worth the
    #     trouble to verify them.
    # As a general principle, we only want direct measurements of inhibition of the pure enzyme, not measurements of 
    # indirect phenotypic effects.
    
    assay_exclude = np.array([('domain' in assay) or ('Jurkat cell' in assay) or ('proliferation' in assay) 
                             for assay in raw_df.enzyme_cell_assay.values])
    cell_type_exclude = np.array([(type(cells) == str) and not (cells in ['Sf9', 'SF-9', 'Sf21', 'SF-21']) 
                                  for cells in raw_df.cells_cellline_organ.values])
    hela_cells = np.array([(type(cells) == str) and (cells.lower() == 'hela') for cells in raw_df.cells_cellline_organ.values])
    isolated_from = np.array([('isolated from' in assay.lower()) or 
                              ('from hela' in assay.lower()) or 
                              ('from human hela' in assay.lower()) for assay in raw_df.enzyme_cell_assay.values])
    from_hela_cells = isolated_from & hela_cells
    cell_type_exclude = cell_type_exclude & ~from_hela_cells
    filtered_df = raw_df.loc[~(assay_exclude | cell_type_exclude)]
    excluded_df = raw_df.loc[assay_exclude | cell_type_exclude]

    # Write excluded and included records to separate files
    excluded_file = f"{target_dir}/{target}_p{activity_type}_excluded_{db}_{db_version}.csv"
    excluded_df.to_csv(excluded_file, index=False)
    print(f"Wrote excluded {db} data to {excluded_file}")

    filtered_file = f"{target_dir}/{target}_p{activity_type}_filtered_{db}_{db_version}.csv"
    filtered_df.to_csv(filtered_file, index=False)
    print(f"Wrote filtered {db} data to {filtered_file}")

    return filtered_df


# ---------------------------------------------------------------------------------------------------------------------------------
def find_chembl_gostar_activity_discrepancies(target, activity_type, gostar_version='2022-06-20', chembl_version='30'):
    """
    Compare ChEMBL and GoStar versions of activity data extracted from the same references, grouped by reference and base SMILES.
    Find groups where one DB has different number of values, or same length but different values,
    and write them to a table for manual curation.
    """
    target_dir = f"{dset_dir}/{target}"
    chembl_file = f"{target_dir}/chembl/{target}_p{activity_type}_filtered_chembl_{chembl_version}.csv"
    chembl_df = pd.read_csv(chembl_file)
    chembl_df['db'] = 'chembl'
    chembl_df['db_version'] = chembl_version

    gostar_file = f"{target_dir}/gostar/{target}_p{activity_type}_filtered_gostar_{gostar_version}.csv"
    gostar_df = pd.read_csv(gostar_file)
    gostar_df['db'] = 'gostar'
    gostar_df['db_version'] = gostar_version

    # Combine ChEMBL and GoStar records, eliminating duplicates
    cmn_cols = ['activity_id', 'compound_id', 'compound_name', 'base_rdkit_smiles', 'activity_type', 'relation', 'activity_value',
                'orig_activity_type', 'orig_activity_prefix', 'orig_activity_value', 'orig_activity_uom',
                'assay_type', 'enzyme_cell_assay', 'cells_cellline_organ', 'standard_name', 'species',
                'ref_id', 'reference', 'ref_search_url', 'title', 'year', 'db', 'db_version']
    chembl_df = chembl_df[cmn_cols].copy()
    gostar_df = gostar_df[cmn_cols].copy()

    # Find references that are sources for both ChEMBL and GoStar data. Take the data from these common references and
    # write it to an intermediate file.
    cmn_refs = set(chembl_df.reference.values) & set(gostar_df.reference.values)
    cmn_ref_df = pd.concat([chembl_df[chembl_df.reference.isin(cmn_refs)], gostar_df[gostar_df.reference.isin(cmn_refs)]], 
                            ignore_index=True)

    chembl_cmn_df = chembl_df[chembl_df.reference.isin(cmn_refs)].copy()
    gostar_cmn_df = gostar_df[gostar_df.reference.isin(cmn_refs)].copy()
    def add_rec_and_cmpd_counts(df):
        df['cmpd_count'] = len(set(df.base_rdkit_smiles.values))
        df['rec_count'] = len(df)
        return df

    chembl_cmn_df = chembl_cmn_df.groupby('reference').apply(add_rec_and_cmpd_counts)
    gostar_cmn_df = gostar_cmn_df.groupby('reference').apply(add_rec_and_cmpd_counts)
    cmn_ref_df = pd.concat([chembl_cmn_df, gostar_cmn_df], ignore_index=True)

    cmn_ref_file = f"{target_dir}/{target}_gostar_{gostar_version}_chembl_{chembl_version}_common_ref_{activity_type}_data.csv"
    cmn_ref_df.to_csv(cmn_ref_file, index=False)
    print(f"Wrote data from references common to GoStar and ChEMBL to {cmn_ref_file}")

    def find_discrepancies(df):
        """
        Called on each slice of cmn_ref_df grouped by reference, base SMILES, target and activity type. If slice has records
        from both ChEMBL and GoStar, and the activities from the two DBs differ in number or values, return the discrepant
        rows. Otherwise return an empty data frame.
        """
        chembl_part = df[df.db == 'chembl']
        gostar_part = df[df.db == 'gostar']
        empty_df = df[df.db == 'foobar']
        discrep_df = df.sort_values(by=['db', 'activity_value']).copy()
        if (len(chembl_part) > 0) and (len(gostar_part) > 0):
            # Check that ChEMBL and GoStar versions have same number of distinct values and approximately the same values.
            chembl_act = np.array(sorted(set(chembl_part.activity_value.values)))
            gostar_act = np.array(sorted(set(gostar_part.activity_value.values)))
            if len(gostar_act) != len(chembl_act):
                discrep_df['discrepancy'] = 'length'
                return discrep_df
            else:
                diffs = np.abs(chembl_act - gostar_act)
                if np.any(diffs > 0.1):
                    # Check for differences that are exact integers, indicating a discrepancy in units (e.g., uM vs nM)
                    if np.all(np.abs(diffs - np.around(diffs)) < 0.01):
                        discrep_df['discrepancy'] = 'units'
                    else:
                        discrep_df['discrepancy'] = 'values'
                    return discrep_df
        return empty_df

    discrepant_df = cmn_ref_df.groupby(['reference', 'base_rdkit_smiles']).apply(find_discrepancies)
    # Set up a column to be filled in by manual editing of the discrepancy table. Initialize it with the values I
    # filled in previously when curating the PARP1 selectivity data.
    resolved_file = f"{selectivity_dir}/parp1_parp2_gostar_chembl_discrepancy_resolution.csv"
    resolved_df = pd.read_csv(resolved_file)
    resolved_df = resolved_df[['activity_id', 'remove']].fillna(0)
    discrepant_df = discrepant_df.merge(resolved_df, how='left', on='activity_id')

    # Make two copies of the discrepancy file: one to maintain a record of the initial resolutions for
    # each discrepancy, the other to be edited to resolve the remaining discrepancies.
    discrepant_file = f"{target_dir}/{target}_gostar_{gostar_version}_chembl_{chembl_version}_{activity_type}_discrepancies.csv"
    discrepant_df.to_csv(discrepant_file, index=False)
    print(f"Wrote data discrepancies between GoStar and ChEMBL to {discrepant_file}")
    resolution_file = f"{target_dir}/{target}_gostar_{gostar_version}_chembl_{chembl_version}_{activity_type}_discrepancy_resolution.csv"
    if not os.path.exists(resolution_file):
        discrepant_df.to_csv(resolution_file, index=False)
        print(f"... and to {resolution_file}")
    else:
        print(f"Discrepancy resolution file {resolution_file} already exists; will not overwrite it.")


# ---------------------------------------------------------------------------------------------------------------------------------
def combine_parp_activity_data(target, activity_type, gostar_version='2022-06-20', chembl_version='30', 
                               gostar_custom_version='2022-06-23', ucsf_version='2022-06-21'):
    """
    Combine activity data for the given target and activity type from ChEMBL and GoStar, after manually resolving
    discrepancies found by find_chembl_gostar_activity_discrepancies, to produce a table with one row per unique
    activity measurement per reference.
    """
    target_dir = f"{dset_dir}/{target}"
    chembl_file = f"{target_dir}/chembl/{target}_p{activity_type}_filtered_chembl_{chembl_version}.csv"
    chembl_df = pd.read_csv(chembl_file)
    chembl_df['db'] = 'chembl'
    chembl_df['db_version'] = chembl_version

    gostar_file = f"{target_dir}/gostar/{target}_p{activity_type}_filtered_gostar_{gostar_version}.csv"
    gostar_df = pd.read_csv(gostar_file)
    gostar_df['db'] = 'gostar'
    gostar_df['db_version'] = gostar_version

    # Combine ChEMBL and GoStar records, eliminating duplicates
    cmn_cols = ['activity_id', 'compound_id', 'compound_name', 'base_rdkit_smiles', 'activity_type', 'relation', 'activity_value',
                'orig_activity_type', 'orig_activity_prefix', 'orig_activity_value', 'orig_activity_uom',
                'assay_type', 'enzyme_cell_assay', 'cells_cellline_organ', 'standard_name', 'species',
                'ref_id', 'reference', 'ref_search_url', 'title', 'year', 'db', 'db_version']
    chembl_df = chembl_df[cmn_cols].copy()
    gostar_df = gostar_df[cmn_cols].copy()

    # Find references that are sources for both ChEMBL and GoStar data. Then split the data, one part from the common
    # references and the other from the references that are unique to ChEMBL or GoStar.
    cmn_refs = set(chembl_df.reference.values) & set(gostar_df.reference.values)
    cmn_ref_df = pd.concat([chembl_df[chembl_df.reference.isin(cmn_refs)], gostar_df[gostar_df.reference.isin(cmn_refs)]], 
                            ignore_index=True)
    uniq_ref_df = pd.concat([chembl_df[~chembl_df.reference.isin(cmn_refs)], gostar_df[~gostar_df.reference.isin(cmn_refs)]], 
                            ignore_index=True)

    # Load the discrepancy resolution file, which contains manual annotations of which discrepant records to remove
    resolution_file = f"{target_dir}/{target}_gostar_{gostar_version}_chembl_{chembl_version}_{activity_type}_discrepancy_resolution.csv"
    resolution_df = pd.read_csv(resolution_file)
    removed_ids = set(resolution_df[resolution_df.remove == 1].activity_id.values)
    cmn_ref_df = cmn_ref_df[~cmn_ref_df.activity_id.isin(removed_ids)]

    # Eliminate additional duplicates or near-duplicates between ChEMBL and GoStar versions of data from same references

    def eliminate_dups(df):
        """
        Called on each slice of cmn_ref_df grouped by reference, base SMILES, target and activity type. If slice has records
        from both ChEMBL and GoStar, return the ChEMBL rows only. Otherwise, return the whole slice.
        """
        chembl_part = df[df.db == 'chembl']
        gostar_part = df[df.db == 'gostar']
        if (len(chembl_part) > 0) and (len(gostar_part) > 0):
            return chembl_part
        else:
            return df

    undup_df = cmn_ref_df.groupby(['reference', 'base_rdkit_smiles']).apply(eliminate_dups)

    # Combine the deduplicated data with the data from references that were unique to GoStar or ChEMBL
    combined_df = pd.concat([uniq_ref_df, undup_df], ignore_index=True)

    # Merge in the custom-curated GoStar data and the data generated at UCSF. First, identify the columns common to all datasets.
    custom_file = f"{target_dir}/gostar/{target}_p{activity_type}_filtered_gostar_custom_{gostar_custom_version}.csv"
    custom_df = pd.read_csv(custom_file)
    ucsf_file = f"{target_dir}/ucsf/{target}_{activity_type}_filtered_ucsf_{ucsf_version}.csv"
    ucsf_df = pd.read_csv(ucsf_file)
    combined_cols = set(combined_df.columns.values)
    custom_cols = set(custom_df.columns.values)
    ucsf_cols = set(ucsf_df.columns.values)
    common_cols = list(combined_cols & custom_cols & ucsf_cols)

    common_df = pd.concat([combined_df[common_cols], custom_df[common_cols], ucsf_df[common_cols]], ignore_index=True)

    # Assign a unique compound_id for each base SMILES string
    smiles_id_set_map = {}
    for compound_id, smiles in zip(common_df.compound_id.values, common_df.base_rdkit_smiles.values):
        smiles_id_set_map.setdefault(smiles, set()).add(compound_id)
    smiles_id_map = {}
    for smiles, id_set in smiles_id_set_map.items():
        smiles_id_map[smiles] = sorted(id_set)[0]
    common_df['compound_id'] = [smiles_id_map[smiles] for smiles in common_df.base_rdkit_smiles.values]

    print(f"Combined ChEMBL and GoStar {activity_type} data for {target} has {len(common_df)} records, {len(set(common_df.base_rdkit_smiles.values))} unique compounds.")

    common_file = f"{target_dir}/{target}_gostar_{gostar_version}_chembl_{chembl_version}_gostar_custom_{gostar_custom_version}_UCSF_p{activity_type}_combined.csv"
    common_df.to_csv(common_file, index=False)
    print(f"Wrote combined {target} data to {common_file}")

    return common_df

    
# ---------------------------------------------------------------------------------------------------------------------------------
def average_combined_parp_activity_data(activity_df=None, target='PARP1', activity_type='pIC50', gostar_version='2022-06-20', chembl_version='30',
                                        gostar_custom_version='2022-06-23'):
    """
    Remove outlier replicates from the combined ChEMBL, GoStar, GoStar custom and UCSF activity data and average the replicates
    to produce a table with one row per compound, containing negative log activity values.
    """
    target_dir = f"{dset_dir}/{target}"
    if activity_type in ['IC50', 'Ki']:
        activity_type = f"p{activity_type}"
    if activity_df is None:
        activity_file = f"{target_dir}/{target}_gostar_{gostar_version}_chembl_{chembl_version}_gostar_custom_{gostar_custom_version}_UCSF_{activity_type}_combined.csv"
        activity_df = pd.read_csv(activity_file)
    # Remove outliers
    filt_df = remove_outlier_replicates(activity_df, response_col='activity_value', max_diff_from_median=1.0)
    # Create a table of compounds for which some activities were removed, for later analysis
    outlier_ids = set(activity_df.activity_id.values) - set(filt_df.activity_id.values)
    outlier_df = activity_df[activity_df.activity_id.isin(outlier_ids)]
    outlier_cmpds = set(outlier_df.base_rdkit_smiles.values)
    outlier_cmpd_df = activity_df[activity_df.base_rdkit_smiles.isin(outlier_cmpds)].copy()
    outlier_cmpd_df['outlier'] = [1 if id in outlier_ids else 0 for id in outlier_cmpd_df.activity_id.values]
    outlier_file = f"{target_dir}/{target}_gostar_{gostar_version}_chembl_{chembl_version}_gostar_custom_{gostar_custom_version}_UCSF_{activity_type}_outlier_cmpds.csv"
    outlier_cmpd_df.to_csv(outlier_file, index=False)
    print(f"Wrote {target} {activity_type} outliers to {outlier_file}")

    # Aggregate activity values
    agg_activity_df = aggregate_assay_data(filt_df, value_col='activity_value', label_actives=False, id_col='compound_id',
                                      smiles_col='base_rdkit_smiles', relation_col='relation')

    agg_file = f"{target_dir}/{target}_gostar_{gostar_version}_chembl_{chembl_version}_gostar_custom_{gostar_custom_version}_UCSF_{activity_type}_agg.csv"
    agg_activity_df.to_csv(agg_file, index=False)
    print(f"Wrote aggregated {target} {activity_type} values to {agg_file}")
    return agg_activity_df

# ---------------------------------------------------------------------------------------------------------------------------------
def analyze_outliers_by_reference(target='PARP1', activity_type='pIC50', gostar_version='2022-06-20', chembl_version='30',
                                        gostar_custom_version='2022-06-23', nbad=5):
    """
    Group outlier values by the references they came from and plot the distribution of their deviations from the per-compound median
    activities. Extend this comparison to the entire set of activities from each reference to see if there is a systemic error.
    """
    # Read curated activity values
    target_dir = f"{dset_dir}/{target}"
    activity_file = f"{target_dir}/{target}_gostar_{gostar_version}_chembl_{chembl_version}_gostar_custom_{gostar_custom_version}_UCSF_{activity_type}_combined.csv"
    activity_df = pd.read_csv(activity_file)

    # Compute the deviations of each activity value from the median over its compound
    gby = activity_df.groupby('compound_id')
    def compute_deviations(df):
        med = np.median(df.activity_value.values)
        df['deviation'] = df.activity_value.values - med
        df['median_activity'] = med
        # Exclude compounds with no replicates
        if len(df) == 1:
            df = df[[False]]
        return df
    dev_df = gby.apply(compute_deviations)

    # Read the file of data for all compounds that had outliers removed
    outlier_file = f"{target_dir}/{target}_gostar_{gostar_version}_chembl_{chembl_version}_gostar_custom_{gostar_custom_version}_UCSF_{activity_type}_outlier_cmpds.csv"
    outlier_cmpd_df = pd.read_csv(outlier_file)

    # Find the references that have nbad or more outliers
    outlier_df = outlier_cmpd_df[outlier_cmpd_df.outlier == 1]
    bad_refs = freq_table(outlier_df, 'reference', min_freq=nbad)['reference'].values
    nbad_refs = len(bad_refs)

    # Plot the distribution of deviations over all activities from each reference
    fig, axes = plt.subplots(nbad_refs, 1, figsize=(15, 6*nbad_refs))
    for i, ref in enumerate(bad_refs):
        ref_dev_df = dev_df[dev_df.reference == ref]
        ref_outlier_df = outlier_df[outlier_df.reference == ref]
        nout = sum(ref_outlier_df.outlier.values)
        ntot = len(ref_dev_df)
        ncmpd = len(set(ref_dev_df.compound_id.values))
        ax = sns.kdeplot(x='deviation', data=ref_dev_df, hue='orig_activity_type', ax=axes[i], bw_adjust=0.1)
        ax.set_title(f"{ref}: {nout} outliers / {ntot} activities for {ncmpd} compounds")


# ---------------------------------------------------------------------------------------------------------------------------------
def reference_outlier_details(reference, target='PARP1', activity_type='pIC50', gostar_version='2022-06-20', chembl_version='30',
                                        gostar_custom_version='2022-06-23', nbad=5):
    """
    Select outliers from the given reference, extract data from all other references for the same compounds, make a strip plot by compound,
    and return the data as a data frame.
    """
    # Read the file of data for all compounds that had outliers removed
    target_dir = f"{dset_dir}/{target}"
    outlier_file = f"{target_dir}/{target}_gostar_{gostar_version}_chembl_{chembl_version}_gostar_custom_{gostar_custom_version}_UCSF_{activity_type}_outlier_cmpds.csv"
    outlier_cmpd_df = pd.read_csv(outlier_file)
    outlier_df = outlier_cmpd_df[outlier_cmpd_df.outlier == 1]
    ref_outlier_df = outlier_df[outlier_df.reference == reference]
    ref_outlier_cmpds = set(ref_outlier_df.compound_id.values)

    # Read curated activity values
    target_dir = f"{dset_dir}/{target}"
    activity_file = f"{target_dir}/{target}_gostar_{gostar_version}_chembl_{chembl_version}_gostar_custom_{gostar_custom_version}_UCSF_{activity_type}_combined.csv"
    activity_df = pd.read_csv(activity_file)
    cmpd_act_df = activity_df[activity_df.compound_id.isin(ref_outlier_cmpds)]

    #fig, ax = plt.subplots(figsize=(20,20))
    #sns.stripplot(x='activity_value', y='compound_id', jitter=True, data=cmpd_act_df, hue='reference', orient='h')

    # Generate a table of all the compounds in the given reference, including all the values from other references for the same compounds
    ref_act_df = activity_df[activity_df.reference == reference]
    ref_cmpds = set(ref_act_df.compound_id.values)
    ref_cmpd_act_df = activity_df[activity_df.compound_id.isin(ref_cmpds)].copy()
    ref_cmpd_act_df['short_ref'] = 'Other'
    ref_cmpd_act_df.loc[ref_cmpd_act_df.reference == reference, 'short_ref'] = reference
    fig, ax = plt.subplots(figsize=(20,20))
    sns.stripplot(x='activity_value', y='compound_id', jitter=True, data=ref_cmpd_act_df, hue='short_ref', orient='h')
    ref_id = ref_outlier_df.ref_id.values[0]
    ref_cmpd_act_file = f"{target_dir}/gostar/{target}_gostar_reference_{ref_id}_compound_activities.csv"
    ref_cmpd_act_df.to_csv(ref_cmpd_act_file, index=False)
    print(f"Wrote all activities for compounds in {reference} to {ref_cmpd_act_file}")

    return ref_cmpd_act_df



# ---------------------------------------------------------------------------------------------------------------------------------
# Beginning of code for dealing with PARP1/PARP2 selectivity data
# ---------------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------------------
def get_curated_parp_selectivity_data(db, db_version=None, force_update=False):
    """
    Create a dataset of PARP1 and PARP2 activity data from references that have IC50 or Ki data for both PARP1 and PARP2.
    """
    if (db == 'chembl') and (db_version is None):
        db_version = chembl.get_current_version()
    curated_file = f"{selectivity_dir}/{db}/parp1_parp2_selectivity_curated_{db}_{db_version}.csv"
    if not force_update and os.path.isfile(curated_file):
        curated_df = pd.read_csv(curated_file)
        return curated_df

    targets = ['PARP1', 'PARP2']
    if db == 'ucsf':
        act_types = ['IC50']
    else:
        act_types = ['IC50', 'Ki']
    curated_data = []
    for activity_type in act_types:
        act_data = {}
        targ_refs = {}
        for target in targets:
            # Get prefiltered activity data for the given target and activity type
            act_df = curate_parp_activity_data(target=target, activity_type=activity_type, db=db, db_version=db_version,
                                               force_update=force_update)
            act_data[target] = act_df
            targ_refs[target] = set(act_df.reference.values)
            print(f"{target} has {activity_type} data in {db} for {len(set(act_df.base_rdkit_smiles.values))} compounds from {len(targ_refs[target])} references")
        cmn_refs = targ_refs['PARP1'] & targ_refs['PARP2']
        print(f"{len(cmn_refs)} references have {activity_type} data for both PARP1 and PARP2 in {db}")
        
        selectivity_refs = []
        parp1_data = []
        parp2_data = []
        parp1_df = act_data['PARP1']
        parp2_df = act_data['PARP2']

        for ref in sorted(cmn_refs):
            ref_parp1_df = parp1_df[parp1_df.reference == ref]
            ref_parp2_df = parp2_df[parp2_df.reference == ref]
            ref_cmn_smiles = set(ref_parp1_df.base_rdkit_smiles.values) & set(ref_parp2_df.base_rdkit_smiles.values)
            print(f"Reference {ref} has both PARP1 and PARP2 {activity_type} data in {db} for {len(ref_cmn_smiles)} compounds")
            if len(ref_cmn_smiles) > 0:
                selectivity_refs.append(ref)
                ref_parp1_df = ref_parp1_df[ref_parp1_df.base_rdkit_smiles.isin(ref_cmn_smiles)]
                ref_parp2_df = ref_parp2_df[ref_parp2_df.base_rdkit_smiles.isin(ref_cmn_smiles)]
                parp1_data.append(ref_parp1_df)
                parp2_data.append(ref_parp2_df)

        parp1_df = pd.concat(parp1_data, ignore_index=True)
        parp1_df['target'] = 'PARP1'
        parp2_df = pd.concat(parp2_data, ignore_index=True)
        parp2_df['target'] = 'PARP2'
        act_df = pd.concat([parp1_df, parp2_df], ignore_index=True)

        curated_data.append(act_df)
    

    curated_df = pd.concat(curated_data, ignore_index=True)
    curated_df.to_csv(curated_file, index=False)
    print(f"Wrote curated selectivity data to {curated_file}")

    return curated_df

# ---------------------------------------------------------------------------------------------------------------------------------
def find_chembl_gostar_selectivity_discrepancies(gostar_version='2022-06-20', chembl_version='30'):
    """
    Compare ChEMBL and GoStar versions of data extracted from the same references, grouped by reference, base SMILES,
    target and activity type. Find groups where one DB has different number of values, or same length but different values,
    and write them to a table for manual curation.
    """
    chembl_file = f"{selectivity_dir}/chembl/parp1_parp2_selectivity_curated_chembl_{chembl_version}.csv"
    chembl_df = pd.read_csv(chembl_file)
    chembl_df['db'] = 'chembl'
    chembl_df['db_version'] = chembl_version
    gostar_file = f"{selectivity_dir}/gostar/parp1_parp2_selectivity_curated_gostar_{gostar_version}.csv"
    gostar_df = pd.read_csv(gostar_file)
    gostar_df['db'] = 'gostar'
    gostar_df['db_version'] = gostar_version
    gostar_df = gostar_df.rename(columns={'act_id': 'activity_id'})

    # Combine ChEMBL and GoStar records, eliminating duplicates
    cmn_cols = ['activity_id', 'compound_id', 'compound_name', 'base_rdkit_smiles', 'activity_type', 'relation', 'activity_value',
                'assay_type', 'enzyme_cell_assay', 'cells_cellline_organ', 'standard_name', 'target', 'source',
                'ref_id', 'reference', 'title', 'year', 'db', 'db_version']
    chembl_df = chembl_df[cmn_cols].copy()
    gostar_df = gostar_df[cmn_cols].copy()
    cmn_refs = set(chembl_df.reference.values) & set(gostar_df.reference.values)
    chembl_cmn_df = chembl_df[chembl_df.reference.isin(cmn_refs)].copy()
    gostar_cmn_df = gostar_df[gostar_df.reference.isin(cmn_refs)].copy()
    def add_rec_and_cmpd_counts(df):
        df['cmpd_count'] = len(set(df.base_rdkit_smiles.values))
        df['rec_count'] = len(df)
        return df

    cmn_ref_df = pd.concat([chembl_cmn_df, gostar_cmn_df], ignore_index=True)


    cmn_ref_file = f"{selectivity_dir}/parp1_parp2_gostar_chembl_common_ref_data.csv"
    cmn_ref_df.to_csv(cmn_ref_file, index=False)
    print(f"Wrote data from references common to GoStar and ChEMBL to {cmn_ref_file}")

    
    def find_discrepancies(df):
        """
        Called on each slice of cmn_ref_df grouped by reference, base SMILES, target and activity type. If slice has records
        from both ChEMBL and GoStar, and the activities from the two DBs differ in number or values, return the discrepant
        rows. Otherwise return an empty data frame.
        """
        chembl_part = df[df.db == 'chembl']
        gostar_part = df[df.db == 'gostar']
        empty_df = df[df.db == 'foobar']
        discrep_df = df.sort_values(by=['db', 'activity_value']).copy()
        if (len(chembl_part) > 0) and (len(gostar_part) > 0):
            # Check that ChEMBL and GoStar versions have same number of values and approximately the same values.
            chembl_act = np.array(sorted(chembl_part.activity_value.values))
            gostar_act = np.array(sorted(gostar_part.activity_value.values))
            if len(gostar_act) != len(chembl_act):
                discrep_df['discrepancy'] = 'length'
                print(f"Warning: GoStar and ChEMBL slices have different sizes for {df.reference.values[0]} "
                      f"{df.compound_id.values[0]} {df.target.values[0]} {df.activity_type.values[0]}: "
                      f"{len(gostar_act)} vs {len(chembl_act)}")
                return discrep_df
            else:
                diffs = np.abs(chembl_act - gostar_act)
                if np.any(diffs > 0.1):
                    discrep_df['discrepancy'] = 'values'
                    print(f"Warning: GoStar and ChEMBL {df.activity_type.values[0]} values are different for {df.reference.values[0]} "
                        f"{df.compound_id.values[0]} {df.target.values[0]}: "
                        f"{gostar_act} vs {chembl_act}")
                    return discrep_df
        return empty_df

    discrepant_df = cmn_ref_df.groupby(['reference', 'base_rdkit_smiles', 'target', 'activity_type']).apply(find_discrepancies)
    discrepant_file = f"{selectivity_dir}/parp1_parp2_gostar_chembl_discrepancies.csv"
    discrepant_df.to_csv(discrepant_file, index=False)
    print(f"Wrote data discrepancies between GoStar and ChEMBL to {discrepant_file}")

# ---------------------------------------------------------------------------------------------------------------------------------
def combine_parp_selectivity_data(gostar_version='2022-06-20', chembl_version='30', gostar_custom_version='2022-06-23',
                                  ucsf_version='2022-06-21', force_update=False):
    """
    Combine curated ChEMBL, GoStar and UCSF selectivity data into one table with one row per compound per 
    activity type per reference, containing activity values for PARP1 and PARP2 and log selectivity values.
    """
    selectivity_file = f"{selectivity_dir}/parp1_parp2_selectivity_chembl_{chembl_version}_gostar_{gostar_version}_gostar_custom_{gostar_custom_version}_ucsf_{ucsf_version}_combined.csv"
    if not force_update and os.path.isfile(selectivity_file):
        sel_df = pd.read_csv(selectivity_file)
        return sel_df

    activity_types = ['pIC50', 'pKi']
    targets = ['PARP1', 'PARP2']

    chembl_df = get_curated_parp_selectivity_data('chembl', chembl_version, force_update=force_update)
    chembl_df['db'] = 'chembl'
    chembl_df['db_version'] = chembl_version

    gostar_df = get_curated_parp_selectivity_data('gostar', gostar_version, force_update=force_update)
    gostar_df['db'] = 'gostar'
    gostar_df['db_version'] = gostar_version

    custom_df = get_curated_parp_selectivity_data('gostar', f"custom_{gostar_custom_version}", force_update=force_update)
    custom_df['db'] = 'gostar'
    custom_df['db_version'] = f"custom_{gostar_custom_version}"

    ucsf_df = get_curated_parp_selectivity_data('ucsf', ucsf_version, force_update=force_update)
    # ucsf data already has db and version

    # Combine ChEMBL and GoStar records, eliminating duplicates
    #cmn_cols = ['activity_id', 'compound_id', 'compound_name', 'base_rdkit_smiles', 'activity_type', 'relation', 'activity_value',
    #            'assay_type', 'enzyme_cell_assay', 'cells_cellline_organ', 'target', 
    #            'ref_id', 'reference', 'ref_search_url', 'title', 'year', 'db', 'db_version']
    #chembl_df = chembl_df[cmn_cols].copy()
    #gostar_df = gostar_df[cmn_cols].copy()

    cmn_refs = set(chembl_df.reference.values) & set(gostar_df.reference.values)
    uniq_ref_df = pd.concat([chembl_df[~chembl_df.reference.isin(cmn_refs)], gostar_df[~gostar_df.reference.isin(cmn_refs)]], 
                            ignore_index=True)
    cmn_ref_df = pd.concat([chembl_df[chembl_df.reference.isin(cmn_refs)], gostar_df[gostar_df.reference.isin(cmn_refs)]], 
                            ignore_index=True)

    # Load discrepancy resolution file, use it to select duplicates to remove 

    removed_ids = set()
    for target in targets:
        target_dir = f"{dset_dir}/{target}"
        for activity_type in activity_types:
            orig_act_type = activity_type.lstrip('p')
            resolution_file = f"{target_dir}/{target}_gostar_{gostar_version}_chembl_{chembl_version}_{orig_act_type}_discrepancy_resolution.csv"
            if os.path.isfile(resolution_file):
                resolution_df = pd.read_csv(resolution_file)
                removed_ids |= set(resolution_df[resolution_df.remove == 1].activity_id.values)
    if len(removed_ids) == 0:
        raise Exception("Failed to load discrepancy resolution file(s)")
    cmn_ref_df = cmn_ref_df[~cmn_ref_df.activity_id.isin(removed_ids)]

    # Eliminate additional duplicates or near-duplicates between ChEMBL and GoStar versions of data from same references

    def eliminate_dups(df):
        """
        Called on each slice of cmn_ref_df grouped by reference, base SMILES, target and activity type. If slice has records
        from both ChEMBL and GoStar, return the ChEMBL rows only. Otherwise, return the whole slice.
        """
        chembl_part = df[df.db == 'chembl']
        gostar_part = df[df.db == 'gostar']
        if (len(chembl_part) > 0) and (len(gostar_part) > 0):
            return chembl_part
        else:
            return df

    undup_df = cmn_ref_df.groupby(['reference', 'base_rdkit_smiles', 'target', 'activity_type']).apply(eliminate_dups)

    # Combine the deduplicated data with the data from references that were unique to GoStar or ChEMBL
    combined_df = pd.concat([uniq_ref_df, undup_df], ignore_index=True)

    # Combine this data with the custom-curated GoStar data and the UCSF data
    combined_cols = set(combined_df.columns.values)
    custom_cols = set(custom_df.columns.values)
    ucsf_cols = set(ucsf_df.columns.values)
    common_cols = list(combined_cols & custom_cols & ucsf_cols)
    common_df = pd.concat([combined_df[common_cols], custom_df[common_cols], ucsf_df[common_cols]], ignore_index=True)

    # Assign a unique compound_id for each base SMILES string
    smiles_id_set_map = {}
    for compound_id, smiles in zip(common_df.compound_id.values, common_df.base_rdkit_smiles.values):
        smiles_id_set_map.setdefault(smiles, set()).add(compound_id)
    smiles_id_map = {}
    for smiles, id_set in smiles_id_set_map.items():
        smiles_id_map[smiles] = sorted(id_set)[0]
    common_df['compound_id'] = [smiles_id_map[smiles] for smiles in common_df.base_rdkit_smiles.values]
    common_file = f"{selectivity_dir}/parp1_parp2_chembl_{chembl_version}_gostar_{gostar_version}_gostar_custom_{gostar_custom_version}_ucsf_{ucsf_version}_common.csv"
    common_df.to_csv(common_file, index=False)
    print(f"Wrote combined curated GoStar/ChEMBL/UCSF data to {common_file}")

    sel_data = []
    # xxx Figure out where compounds get lost: how do we get from 1150 distinct compounds to only 835

    # Calculate average log selectivities for each unique reference/compound/activity_type. Generate a new data frame
    # with columns for PARP1 & PARP2 activities and log selectivities.
    for act_type in activity_types:
        act_df = common_df[common_df.activity_type == act_type]
        act_refs = sorted(set(act_df.reference.values))
        for ref in act_refs:
            act_ref_df = act_df[act_df.reference == ref]
            ref_smiles = set(act_ref_df.base_rdkit_smiles.values)
            targ_smiles = {}
            for target in targets:
                targ_df = act_ref_df[act_ref_df.target == target]
                targ_smiles[target] = set(targ_df.base_rdkit_smiles.values)
            cmn_smiles = targ_smiles['PARP1'] & targ_smiles['PARP2']
            if cmn_smiles != ref_smiles:
                print(f"Reference '{ref}' has {len(ref_smiles)} unique SMILES but only {len(cmn_smiles)} with data for both PARP1 and PARP2")
            cmn_act_df = act_ref_df[act_ref_df.base_rdkit_smiles.isin(cmn_smiles)]
            agg_data = {}
            for target in targets:
                targ_df = cmn_act_df[cmn_act_df.target == target]
                ref_targ_assays = sorted(set(targ_df.enzyme_cell_assay.values))
                #if len(ref_targ_assays) > 1:
                #    print(f"Warning: more than one {target} {act_type} assay for reference {ref}:")
                #    for assay_desc in ref_targ_assays:
                #        print(f"    {assay_desc}")
                nrec = len(targ_df)
                targ_df = remove_outlier_replicates(targ_df, response_col='activity_value', id_col='base_rdkit_smiles', 
                                                    max_diff_from_median=0.5)
                nrem = nrec - len(targ_df)
                if nrem > 0:
                    print(f"Removed {nrem} outliers from {ref} {target} {act_type}s")
                if len(targ_df) > 0:
                    agg_data[target] = aggregate_assay_data(targ_df, value_col='activity_value', label_actives=False, id_col='compound_id',
                                                            smiles_col='base_rdkit_smiles', relation_col='relation')
                else:
                    agg_data[target] = None
            # Combine the PARP1 and PARP2 aggregated values into one data frame with one row per compound. We may have lost some
            # compounds when removing outliers, so recalculate the set of compounds with data for both targets.
            parp1_df = agg_data['PARP1']
            parp2_df = agg_data['PARP2']
            if (parp1_df is None) or (parp2_df is None):
                continue
            cmn_smiles = set(parp1_df.base_rdkit_smiles.values) & set(parp2_df.base_rdkit_smiles.values)
            parp1_df = parp1_df[parp1_df.base_rdkit_smiles.isin(cmn_smiles)].sort_values(by='base_rdkit_smiles')
            parp2_df = parp2_df[parp2_df.base_rdkit_smiles.isin(cmn_smiles)].sort_values(by='base_rdkit_smiles')
            sel_df = parp1_df.rename(columns={
                                            'relation': 'PARP1_relation',
                                            'activity_value': 'PARP1_activity'})
            sel_df['PARP2_relation'] = parp2_df.relation.values
            sel_df['PARP2_activity'] = parp2_df.activity_value.values
            sel_df['log_selectivity'] = sel_df.PARP1_activity.values - sel_df.PARP2_activity.values
            sel_df['activity_type'] = act_type
            sel_df['reference'] = ref
            sel_data.append(sel_df)
    sel_df = pd.concat(sel_data, ignore_index=True)
    print(f"Before excluding indeterminate selectivities: combined selectivity data has {len(sel_df)} records, {len(set(sel_df.base_rdkit_smiles.values))} compounds")
    unexcluded_file = f"{selectivity_dir}/parp1_parp2_selectivity_chembl_{chembl_version}_gostar_{gostar_version}_gostar_custom_{gostar_custom_version}_ucsf_{ucsf_version}_unexcluded.csv"
    sel_df.to_csv(unexcluded_file, index=False)

    # Figure out relation of true selectivity values to reported values.
    sel_df = determine_log_selectivity_relation(sel_df)

    print(f"After excluding indeterminate selectivities: combined selectivity data has {len(sel_df)} records, {len(set(sel_df.base_rdkit_smiles.values))} compounds")
    sel_df.to_csv(selectivity_file, index=False)
    print(f"Wrote combined selectivity data to {selectivity_file}")

    return sel_df

# ---------------------------------------------------------------------------------------------------------------------------------
def determine_log_selectivity_relation(sel_df):
    """
    Figure out relation of true selectivity values to reported values. Eliminate rows where the relation can't be determined,
    i.e. when the PARP1 and PARP2 values are both left censored or both right censored.
    """
    sel_df = sel_df[~((sel_df.PARP1_relation == '<') & (sel_df.PARP2_relation == '<')) &
                    ~((sel_df.PARP1_relation == '>') & (sel_df.PARP2_relation == '>')) ].copy()
    rel_df = sel_df[['PARP1_relation', 'PARP2_relation']].copy().fillna('')
    parp1_rels = rel_df.PARP1_relation.values
    parp2_rels = rel_df.PARP2_relation.values
    inv_op = {'<': '>', '>': '<', '=': '=', '': ''}
    logsel_rels = np.array([''] * len(parp1_rels))
    logsel_rels[parp1_rels == ''] = [inv_op[op] for op in parp2_rels[parp1_rels == '']]
    logsel_rels[parp1_rels != ''] = parp1_rels[parp1_rels != '']

    sel_df['log_selectivity_relation'] = logsel_rels
    return sel_df


# ---------------------------------------------------------------------------------------------------------------------------------
def average_combined_parp_selectivity_data(sel_df=None, gostar_version='2022-06-20', chembl_version='30',
                                           gostar_custom_version='2022-06-23', ucsf_version='2022-06-21', force_update=False):
    """
    Remove outlier replicates from the combined ChEMBL and GoStar selectivity data and average the replicates
    to produce a table with one row per compound, containing log selectivity values, and one table each for PARP1 and PARP2 
    containing pIC50 values. Writes each of these to a separate file. Note that the sets of compounds will differ between
    these tables because the selectivity table will include compounds that only have pKi data. Finally, generates a combined table
    with one row per compound, with PARP1 & 2 pIC50s and log selectivity for each compound; this table will only include
    compounds that have all three types of data.
    """
    if sel_df is None:
        sel_df = combine_parp_selectivity_data(gostar_version=gostar_version, chembl_version=chembl_version, 
                                               gostar_custom_version=gostar_custom_version, ucsf_version=ucsf_version,
                                               force_update=force_update)
    # Remove outliers based on log selectivity only
    filt_df = remove_outlier_replicates(sel_df, response_col='log_selectivity', max_diff_from_median=1.0)
    # Aggregate log selectivity values
    sel_agg_df = aggregate_assay_data(filt_df, value_col='log_selectivity', label_actives=False, id_col='compound_id',
                                      smiles_col='base_rdkit_smiles', relation_col='log_selectivity_relation')

    sel_agg_file = f"{selectivity_dir}/log_selectivity_chembl_{chembl_version}_gostar_{gostar_version}_gostar_custom_{gostar_custom_version}_ucsf_{ucsf_version}_agg.csv"
    sel_agg_df.to_csv(sel_agg_file, index=False)
    print(f"Wrote aggregated log selectivity values to {sel_agg_file}")

    # Aggregate PARP1 pIC50 values
    parp1_filt_df = filt_df[filt_df.activity_type == 'pIC50'].rename(columns={'PARP1_activity': 'PARP1_pIC50'})
    parp1_agg_df = aggregate_assay_data(parp1_filt_df, value_col='PARP1_pIC50', label_actives=False, id_col='compound_id',
                                      smiles_col='base_rdkit_smiles', relation_col='PARP1_relation')
    parp1_agg_file = f"{selectivity_dir}/PARP1_pIC50_chembl_{chembl_version}_gostar_{gostar_version}_gostar_custom_{gostar_custom_version}_ucsf_{ucsf_version}_agg.csv"
    parp1_agg_df.to_csv(parp1_agg_file, index=False)
    print(f"Wrote aggregated PARP1 pIC50 values to {parp1_agg_file}")

    # Aggregate PARP2 pIC50 values
    parp2_filt_df = filt_df[filt_df.activity_type == 'pIC50'].rename(columns={'PARP2_activity': 'PARP2_pIC50'})
    parp2_agg_df = aggregate_assay_data(parp2_filt_df, value_col='PARP2_pIC50', label_actives=False, id_col='compound_id',
                                      smiles_col='base_rdkit_smiles', relation_col='PARP2_relation')
    parp2_agg_file = f"{selectivity_dir}/PARP2_pIC50_chembl_{chembl_version}_gostar_{gostar_version}_gostar_custom_{gostar_custom_version}_ucsf_{ucsf_version}_agg.csv"
    parp2_agg_df.to_csv(parp2_agg_file, index=False)
    print(f"Wrote aggregated PARP2 pIC50 values to {parp2_agg_file}")

    # Create a combined table of log selectivities, PARP1 and PARP2 pIC50s for the compounds that have all 3 types of data
    logsel_df = sel_agg_df.rename(columns={'relation': 'log_selectivity_relation'})
    parp1_df = parp1_agg_df.rename(
                    columns={'relation': 'PARP1_pIC50_relation'})[['compound_id', 'PARP1_pIC50_relation', 'PARP1_pIC50']]
    parp2_df = parp2_agg_df.rename(
                    columns={'relation': 'PARP2_pIC50_relation'})[['compound_id', 'PARP2_pIC50_relation', 'PARP2_pIC50']]

    combined_df = logsel_df.merge(parp1_df, how='inner', on='compound_id')
    combined_df = combined_df.merge(parp2_df, how='inner', on='compound_id')

    combined_file = f"{selectivity_dir}/PARP1_PARP2_pIC50_log_selectivity_chembl_{chembl_version}_gostar_{gostar_version}_gostar_custom_{gostar_custom_version}_ucsf_{ucsf_version}_agg.csv"
    combined_df.to_csv(combined_file, index=False)
    print(f"Wrote combined selectivity and pIC50 data to {combined_file}")
    return combined_df


# ---------------------------------------------------------------------------------------------------------------------------------
def draw_discrepant_compounds():
    discrep_df = pd.read_csv('/usr/workspace/atom/PARP_compounds/Datasets_and_Models/PARP1_selectivity/parp1_parp2_gostar_chembl_discrepancies.csv')
    uniq_df = discrep_df[discrep_df.db == 'chembl'].drop_duplicates(subset='base_rdkit_smiles').copy()
    uniq_df['IC50_nM'] = 10**(9-uniq_df.activity_value.values)
    uniq_df['IC50_uM'] = 10**(6-uniq_df.activity_value.values)
    vgr.display_compound_table(uniq_df, smiles_col='base_rdkit_smiles', cluster=False,
            extra_cols=['compound_name', 'activity_type', 'relation', 'activity_value', 'IC50_nM', 'IC50_uM', 'target', 'reference'])


# ---------------------------------------------------------------------------------------------------------------------------------
def show_selectivity_spread(sel_df=None, db=None, db_version=None, min_reps=2):
    """
    Find compounds with selectivity data from multiple references and plot spread of values for PARP1 & 2 pIC50 and pKi and selectivity
    """
    if sel_df is None:
        selectivity_file = f"{selectivity_dir}/{db}/parp1_parp2_selectivity_curated_{db}_{db_version}.csv"
        sel_df = pd.read_csv(selectivity_file)

    # Assign a unique compound_id for each base SMILES string
    smiles_idlist_map = {}
    for compound_id, smiles in zip(sel_df.compound_id.values, sel_df.base_rdkit_smiles.values):
        smiles_idlist_map.setdefault(smiles, []).append(compound_id)
    smiles_id_map = {}
    for smiles, idlist in smiles_idlist_map.items():
        smiles_id_map[smiles] = min(idlist)
    sel_df['compound_id'] = [f"gvk_{smiles_id_map[smiles]}" for smiles in sel_df.base_rdkit_smiles.values]

    fr_df = freq_table(sel_df, 'compound_id', min_freq=min_reps)
    rep_ids = fr_df.compound_id.values.tolist()
    rep_df = sel_df[sel_df.compound_id.isin(rep_ids)].copy()
    avg_sel = rep_df[['compound_id', 'log_selectivity']].groupby('compound_id').mean().sort_values(by='log_selectivity', ascending=False)
    avg_sel['avg_sel_rank'] = list(range(len(avg_sel)))
    rep_df['avg_log_selectivity'] = [avg_sel.log_selectivity[id] for id in rep_df.compound_id.values]
    rep_df['avg_sel_rank'] = [avg_sel.avg_sel_rank[id] for id in rep_df.compound_id.values]
    rep_df = rep_df.sort_values(by=['avg_sel_rank'])
    rep_df['compound_index'] = rep_df.avg_sel_rank.values + 0.1 * np.random.randn(len(rep_df))

    fig, axes = plt.subplots(5, 1, figsize=(30,50))
    ax = sns.scatterplot(x='compound_index', y='log_selectivity', hue='avg_sel_rank', data=rep_df, ax=axes[0], legend=False)
    ax_num = 1

    for act_type in ['pIC50', 'pKi']:
        for target in ['PARP1', 'PARP2']:
            act_col = f"{target}_{act_type}"
            act_df = rep_df[rep_df.activity_type == act_type].rename(columns={f"{target}_activity": act_col}).copy()
            act_df = act_df.sort_values(by=act_col)
            avg_act = act_df[['compound_id', act_col]].groupby('compound_id').mean().sort_values(by=act_col)
            avg_act_col = f"avg_{act_col}"
            avg_act_rank_col = f"avg_{act_col}_rank"
            avg_act[avg_act_rank_col] = list(range(len(avg_act)))
            act_df[avg_act_col] = [avg_act[act_col][id] for id in act_df.compound_id.values]
            act_df[avg_act_rank_col] = [avg_act[avg_act_rank_col][id] for id in act_df.compound_id.values]
            act_df = act_df.sort_values(by=avg_act_rank_col)
            act_df['compound_index'] = act_df[avg_act_rank_col].values + 0.1 * np.random.randn(len(act_df))

            ax = sns.scatterplot(x='compound_index', y=act_col, hue=avg_act_rank_col, data=act_df, ax=axes[ax_num], legend=False)
            ax_num += 1


# ---------------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------------------
# =-= Older functions, may no longer be relevant
# ---------------------------------------------------------------------------------------------------------------------------------
def compare_gostar_chembl_cmpds(target):
    """
    Compare compounds overlapping between GoStar and ChEMBL datasets for the given target
    """
    chembl_version = chembl.get_current_version()
    target_dir = f"{dset_dir}/{target}"
    chembl_dir = f"{dset_dir}/{target}/chembl"
    gostar_dir = f"{dset_dir}/{target}/gostar"
    gostar_file = f"{gostar_dir}/{target}_human_IC50_2021_09_30.csv"
    gostar_df = pd.read_csv(gostar_file)
    gostar_df = gostar_df[~gostar_df.sub_smiles.isna()]
    gostar_df['rdkit_smiles'] = rdkit_smiles_from_smiles(gostar_df.sub_smiles.values.tolist(), workers=16)
    gostar_df['reference'] = [ref.replace('.', '') for ref in gostar_df.reference.values]
    gostar_cmpds = set(gostar_df.rdkit_smiles.values)
    chembl_file = f"{chembl_dir}/{target}_IC50_chembl{chembl_version}.csv"
    chembl_df = pd.read_csv(chembl_file)
    chembl_df['reference'] = chembl_df.reference.fillna('')
    chembl_df['reference'] = [ref.replace('.', '') for ref in chembl_df.reference.values]
    chembl_cmpds = set(chembl_df.rdkit_smiles.values)
    cmn_cmpds = gostar_cmpds & chembl_cmpds
    gostar_cmn_df = gostar_df[gostar_df.rdkit_smiles.isin(cmn_cmpds)][['rdkit_smiles', 'reference']].drop_duplicates().rename(
                                columns={'reference': "gostar_ref"}).sort_values(by=['rdkit_smiles', 'gostar_ref'])
    chembl_cmn_df = chembl_df[chembl_df.rdkit_smiles.isin(cmn_cmpds)][['rdkit_smiles', 'reference']].drop_duplicates().rename(
                                columns={'reference': 'chembl_ref'}).sort_values(by=['rdkit_smiles', 'chembl_ref'])
    cmn_refs = set(gostar_df.reference.values) & set(chembl_df.reference.values)

    cmp_data = []
    for smiles in sorted(cmn_cmpds):
        gostar_smi_df = gostar_cmn_df[gostar_cmn_df.rdkit_smiles == smiles].copy()
        chembl_smi_df = chembl_cmn_df[chembl_cmn_df.rdkit_smiles == smiles].copy()
        diff = len(gostar_smi_df) - len(chembl_smi_df)
        if diff == 0:
            gostar_smi_df['chembl_ref'] = chembl_smi_df.chembl_ref.values
            cmp_data.append(gostar_smi_df)
        elif diff > 0:
            gostar_smi_df['chembl_ref'] = chembl_smi_df.chembl_ref.values.tolist() + ['']*diff
            cmp_data.append(gostar_smi_df)
        elif diff < 0:
            chembl_smi_df['gostar_ref'] = gostar_smi_df.gostar_ref.values.tolist() + [''] * (-diff)
            cmp_data.append(chembl_smi_df)
    cmp_df = pd.concat(cmp_data, ignore_index=True)
    print(f"{len(cmn_cmpds)} compounds in both GoStar and ChEMBL datasets")
    print(f"{len(cmn_refs)} common references between GoStar and ChEMBL")
    cmp_file = f"{target_dir}/gostar_chembl_{target}_common_cmpds.csv"
    cmp_df.to_csv(cmp_file, index=False)
    print(f"Wrote {cmp_file}")
    return cmp_df


# ---------------------------------------------------------------------------------------------------------------------------------
def extract_gostar_unique_data(target):
    """
    Identify measurements from GoStar for the given target that are not in ChEMBL, and extract
    them into a separate table.
    """
    target_dir = f"{dset_dir}/{target}"
    chembl_dir = f"{dset_dir}/{target}/chembl"
    gostar_dir = f"{dset_dir}/{target}/gostar"
    gostar_file = f"{gostar_dir}/{target}_human_IC50_2021_09_30.csv"

    gostar_df = pd.read_csv(gostar_file)
    gostar_df = gostar_df[~gostar_df.sub_smiles.isna() & ~gostar_df.activity_value.isna() & ~gostar_df.activity_uom.isna()]
    gostar_df['rdkit_smiles'] = rdkit_smiles_from_smiles(gostar_df.sub_smiles.values.tolist(), workers=16)
    gostar_df['reference'] = [ref.replace('.', '') for ref in gostar_df.reference.values]
    gostar_cmpds = set(gostar_df.rdkit_smiles.values)

    chembl_version = chembl.get_current_version()
    chembl_file = f"{chembl_dir}/{target}_IC50_chembl{chembl_version}.csv"
    chembl_df = pd.read_csv(chembl_file)
    chembl_df['reference'] = chembl_df.reference.fillna('')
    chembl_df['reference'] = [ref.replace('.', '') for ref in chembl_df.reference.values]
    chembl_cmpds = set(chembl_df.rdkit_smiles.values)
    cmn_cmpds = gostar_cmpds & chembl_cmpds
    cmn_refs = set(gostar_df.reference.values) & set(chembl_df.reference.values)
    uniq_refs = set(gostar_df.reference.values) - set(chembl_df.reference.values)

    # First pick out GoStar data from references that aren't in ChEMBL
    gostar_uniq_df = gostar_df[gostar_df.reference.isin(uniq_refs)].sort_values(
                                                                by=['reference', 'rdkit_smiles']).copy()
    gostar_uniq_df['ref_type'] = 'unique to GoStar'
    print(f"{len(uniq_refs)} references unique to GoStar contain {len(gostar_uniq_df)} records, {len(set(gostar_uniq_df.rdkit_smiles.values))} compounds")


    # Then find data from references that are in ChEMBL but are for compounds that don't have data in ChEMBL for the given reference

    gostar_cmn_df = gostar_df[gostar_df.reference.isin(cmn_refs)]
    chembl_cmn_df = chembl_df[chembl_df.reference.isin(cmn_refs)]
    uniq_cmpd_data = []
    for ref in sorted(cmn_refs):
        gostar_ref_df = gostar_cmn_df[gostar_cmn_df.reference == ref].copy()
        chembl_ref_df = chembl_cmn_df[chembl_cmn_df.reference == ref].copy()
        gostar_only_cmpds = set(gostar_ref_df.rdkit_smiles.values) - set(chembl_ref_df.rdkit_smiles.values)
        if len(gostar_only_cmpds) > 0:
            gostar_ref_uniq_df = gostar_ref_df[gostar_ref_df.rdkit_smiles.isin(gostar_only_cmpds)]
            uniq_cmpd_data.append(gostar_ref_uniq_df)
    uniq_cmpd_df = pd.concat(uniq_cmpd_data, ignore_index=True)
    uniq_cmpd_df['ref_type'] = 'common to ChEMBL'
    uniq_cmpd_refs = set(uniq_cmpd_df.reference.values)
    uniq_cmpd_cmpds = set(uniq_cmpd_df.rdkit_smiles.values)
    print(f"{len(uniq_cmpd_refs)} references common with ChEMBL contain {len(uniq_cmpd_df)} records, {len(uniq_cmpd_cmpds)} compounds unique to GoStar")
    
    novel_df = pd.concat([gostar_uniq_df, uniq_cmpd_df], ignore_index=True)
    novel_file = f"{gostar_dir}/{target}_gostar_unique_IC50_data.csv"
    novel_df.to_csv(novel_file, index=False)

    print(f"Wrote {novel_file}")

    assay_df = novel_df.drop_duplicates(subset=['ref_id', 'enzyme_cell_assay']).copy()
    # Add columns for record and compound counts
    fr_df = freq_table(novel_df, 'ref_id', min_freq=1).rename(columns={'Count': 'ref_records'})
    fr_df['ref_cmpds'] = [len(set(novel_df.loc[novel_df.ref_id == id, 'rdkit_smiles'].values)) for id in fr_df.ref_id.values]
    assay_df = assay_df.merge(fr_df, how='left', on='ref_id')
    assay_df['accept'] = -2
    assay_file = f"{gostar_dir}/{target}_gostar_unique_IC50_assays.csv"
    assay_df.to_csv(assay_file, index=False)
    print(f"Wrote {assay_file}")


# ---------------------------------------------------------------------------------------------------------------------------------
def find_parp1_parp2_common_cmpds():
    """
    Find compounds with IC50 data from either ChEMBL or GoStar for both PARP1 and PARP2
    """
    targets = ['PARP1', 'PARP2']
    dbs = ['chembl', 'gostar']
    chembl_version = chembl.get_current_version()
    input_template = dict(
        chembl = f"%s/%s_IC50_chembl{chembl_version}.csv",
        gostar = "%s/%s_gostar_unique_IC50_data.csv" )
    output_template = dict(
        chembl = f"%s/%s_IC50_chembl{chembl_version}_with_ids.csv",
        gostar = "%s/%s_IC50_gostar_with_ids.csv" )
    cmpds = {}
    dsets = {}
    smiles_id_map = {}
    next_id = 1
    for target in targets:
        dsets[target] = {}
        cmpds[target] = set()
        for db in dbs:
            db_dir = f"{dset_dir}/{target}/{db}"
            dset_file = input_template[db] % (db_dir, target)
            dset_df = pd.read_csv(dset_file)

            # Salt-strip the SMILES strings
            dset_df['base_rdkit_smiles'] = base_smiles_from_smiles(dset_df.rdkit_smiles.values.tolist(), workers=16)

            # Assign a unique ID for each SMILES
            compound_ids = []
            for smiles in dset_df.base_rdkit_smiles.values:
                try:
                    id = smiles_id_map[smiles]
                except KeyError:
                    id = next_id
                    next_id += 1
                    smiles_id_map[smiles] = id
                compound_ids.append(f"PARP_{id}")

            dset_df['compound_id'] = compound_ids
            dsets[target][db] = dset_df
            cmpds[target] |= set(dset_df.base_rdkit_smiles.values)

            # Write out a version of the file with assigned IDs so we can match the data later to MM/GBSA results
            output_file = output_template[db] % (db_dir, target)
            dset_df.to_csv(output_file, index=False)
            print(f"Wrote {len(dset_df)} rows to {output_file}")

    common_cmpds = list(cmpds['PARP1'] & cmpds['PARP2'])
    print(f"{len(common_cmpds)} compounds with both PARP1 and PARP2 data")
    print(f"{len(cmpds['PARP1'])} with PARP1 data")
    print(f"{len(cmpds['PARP2'])} with PARP2 data")

    common_df = pd.DataFrame(dict(
                    compound_id = [f"PARP_{smiles_id_map[smi]}" for smi in common_cmpds],
                    SMILES = common_cmpds)).sort_values(by='compound_id')
    common_file = f"{dset_dir}/PARP1_PARP2_common_compounds.csv"
    common_df.to_csv(common_file, index=False)
    print(f"Wrote {common_file}")

    return common_df

# ---------------------------------------------------------------------------------------------------------------------------------
def combine_chembl_gostar_data():
    """
    Load unique IC50 data from GoStar for the given target, convert to common units and salt-strip and standardize
    SMILES strings. Combine the cleaned-up data with corresponding data from ChEMBL.
    """
    chembl_version = chembl.get_current_version()
    chembl_data = {}
    gostar_data = {}
    targets = ['PARP1', 'PARP2']
    for target in targets:
        target_dir = f"{dset_dir}/{target}"
        gostar_dir = f"{dset_dir}/{target}/gostar"
        gostar_file = f"{gostar_dir}/{target}_gostar_unique_IC50_data.csv"
        gostar_df = pd.read_csv(gostar_file)
        gostar_df = gostar_df[~gostar_df.rdkit_smiles.isna()]
        before = len(gostar_df)
        gostar_df = dcf.exclude_organometallics(gostar_df)
        after = len(gostar_df)
        print(f"Excluded {before-after} organometallic molecules from GoStar {target} data")
        gostar_df['base_rdkit_smiles'] = base_smiles_from_smiles(gostar_df.rdkit_smiles.values.tolist(), workers=16)
        def scale_units(group):
            unit = group.activity_uom.values[0]
            if unit == 'M':
                group['activity_value'] = 1e9 * group.activity_value.values
            elif unit == 'uM':
                group['activity_value'] = 1000 * group.activity_value.values
            elif unit != 'nM':
                raise Exception(f"Unexpected units {group.name} in GoStar dataset")
            return group
        gostar_df = gostar_df.groupby('activity_uom').apply(scale_units)
        gostar_df = dcf.standardize_relations(gostar_df, db='GoStar')
        gostar_df['pIC50'] = 9.0 - np.log10(gostar_df.activity_value.values)
        # Invert relational operators since they are opposite for IC50 and pIC50
        inverse_relation = {'<': '>', '>': '<', '=': '='}
        gostar_df['relation'] = [inverse_relation[r] for r in gostar_df.activity_prefix.values]
        gostar_cur_df = gostar_df[['relation', 'pIC50', 'rdkit_smiles', 'base_rdkit_smiles']].copy()
        gostar_cur_df['source'] = 'GoStar'
        gostar_cur_df['compound_id'] = ''   # will generate these later
        gostar_data[target] = gostar_cur_df
    
        chembl_dir = f"{dset_dir}/{target}/chembl"
        chembl_file = f"{chembl_dir}/{target}_IC50_chembl{chembl_version}.csv"
        chembl_df = pd.read_csv(chembl_file)
        chembl_df = chembl_df[~chembl_df.rdkit_smiles.isna()]
        before = len(chembl_df)
        chembl_df = dcf.exclude_organometallics(chembl_df)
        after = len(chembl_df)
        print(f"Excluded {before-after} organometallic molecules from ChEMBL {target} data")
        chembl_df['base_rdkit_smiles'] = base_smiles_from_smiles(chembl_df.rdkit_smiles.values.tolist(), workers=16)
        chembl_df = chembl_df[chembl_df.standard_units == 'nM']
        chembl_df = dcf.standardize_relations(chembl_df, db='ChEMBL')
        chembl_df['relation'] = [inverse_relation[r] for r in chembl_df.standard_relation.values]
        chembl_df['pIC50'] = 9.0 - np.log10(chembl_df.standard_value.values)
        chembl_cur_df = chembl_df[['mol_chembl_id', 'relation', 'pIC50', 'rdkit_smiles', 'base_rdkit_smiles']].rename(
                    columns={'mol_chembl_id': 'compound_id'}).copy()
        chembl_cur_df['source'] = 'ChEMBL'
        chembl_data[target] = chembl_cur_df

    # Select or generate a unique compound ID for each base SMILES string. Use the lexicographically smallest ChEMBL
    # ID for each SMILES by default; then generate IDs for the remaining GoStar compounds that aren't also in ChEMBL
    chembl_df = pd.concat([chembl_data[target] for target in targets], ignore_index=True)
    chembl_df = chembl_df.sort_values(by='compound_id').drop_duplicates(subset=['base_rdkit_smiles'])
    smiles_id_map = dict(zip(chembl_df.base_rdkit_smiles.values, chembl_df.compound_id.values))
    id_idx = 0
    combined_data = {}
    for target in targets:
        chembl_df = chembl_data[target]
        chembl_df['compound_id'] = [smiles_id_map[smiles] for smiles in chembl_df.base_rdkit_smiles.values]
        #chembl_data[target] = chembl_df

        gostar_df = gostar_data[target]
        ids = []
        for smiles in gostar_df.base_rdkit_smiles.values:
            try:
                cmpd_id = smiles_id_map[smiles]
            except KeyError:
                cmpd_id = f"PARP_{id_idx}"
                smiles_id_map[smiles] = cmpd_id
                id_idx += 1
            ids.append(cmpd_id)
        gostar_df['compound_id'] = ids
        #gostar_data[target] = gostar_df

        combined_data[target] = combined_df = pd.concat([chembl_df, gostar_df], ignore_index=True)

    return combined_data


# ---------------------------------------------------------------------------------------------------------------------------------
def _list_null_source_refs():
    """
    Tabulate GoStar record and compound counts for PARP2 targets with source = NULL
    """
    parp2_df = pd.read_csv(f"{dset_dir}/PARP2/PARP2_activity_gostar_2022_05_06.csv")
    # restrict to source NULL
    parp2_df = parp2_df[parp2_df.source.isna()]

    parp2_data = []
    parp2_df['base_rdkit_smiles'] = base_smiles_from_smiles(parp2_df.sub_smiles.values.tolist(), workers=16)

    # Exclude records with SMILES that RDKit didn't like
    parp2_df = parp2_df[parp2_df.base_rdkit_smiles != '']

    # Count records and compounds for each reference
    refs = sorted(set(parp2_df.reference.values))
    rec_counts = []
    cmpd_counts = []
    for ref in refs:
        ref_df = parp2_df[parp2_df.reference == ref]
        rec_counts.append(len(ref_df))
        cmpd_counts.append(len(set(ref_df.base_rdkit_smiles.values)))
    res_df = pd.DataFrame(dict(reference=refs, record_count=rec_counts, cmpd_count=cmpd_counts)).sort_values(by='cmpd_count', ascending=False)
    res_file = f"{selectivity_dir}/gostar/PARP2_null_source_ref_counts.csv"
    res_df.to_csv(res_file, index=False)
    print(f"Wrote reference table to {res_file}")



# ---------------------------------------------------------------------------------------------------------------------------------
def check_excelra_curation_refs(db_version='2022-05-06'):
    """
    Check references Excelra claimed to have both PARP1 and PARP2 data. Compare to data obtained by querying GoStar directly
    for PARP data with source NULL or Human.
    """
    # First look at data from list of articles and patents that Excelra identified as having both
    # PARP1 and PARP2 data.
    article_df = pd.read_csv(f"{selectivity_dir}/gostar/PARP1_PARP2_by_articles_{db_version}.csv")
    patent_df = pd.read_csv(f"{selectivity_dir}/gostar/PARP1_PARP2_by_patents_{db_version}.csv")
    ref_df = pd.concat([article_df, patent_df], ignore_index=True)
    all_refs = set(ref_df.reference.values)
    parp1_ref_df = ref_df[ref_df.standard_name == 'Poly(ADP-ribose) polymerase 1']
    parp1_refs = set(parp1_ref_df.reference.values)
    parp2_ref_df = ref_df[ref_df.standard_name == 'Poly(ADP-ribose) polymerase 2']
    parp2_refs = set(parp2_ref_df.reference.values)
    cmn_refs = parp1_refs & parp2_refs
    print(f"Excelra reference list has {len(all_refs)} references, {len(parp1_refs)} with PARP1 data,")
    print(f" {len(parp2_refs)} with PARP2 data, {len(cmn_refs)} with data for both.")

    # Restrict this set to source Human or NULL and count again
    href_df = ref_df[ref_df.source.isna() | (ref_df.source == 'Human')]
    all_refs = set(href_df.reference.values)
    parp1_ref_df = href_df[href_df.standard_name == 'Poly(ADP-ribose) polymerase 1']
    parp1_refs = set(parp1_ref_df.reference.values)
    parp2_ref_df = href_df[href_df.standard_name == 'Poly(ADP-ribose) polymerase 2']
    parp2_refs = set(parp2_ref_df.reference.values)
    cmn_refs = parp1_refs & parp2_refs
    print(f"After restricting to human: Excelra reference list has {len(all_refs)} references, {len(parp1_refs)} with PARP1 data,")
    print(f" {len(parp2_refs)} with PARP2 data, {len(cmn_refs)} with data for both.")

    # Compare to data obtained by querying against standard PARP1 and PARP2 target names directly
    parp1_df = pd.read_csv(f"{dset_dir}/PARP1/PARP1_activity_gostar_2022_05_06.csv")
    parp2_df = pd.read_csv(f"{dset_dir}/PARP2/PARP2_activity_gostar_2022_05_06.csv")
    dparp1_refs = set(parp1_df.reference.values)
    dparp2_refs = set(parp2_df.reference.values)
    dcmn_refs = dparp1_refs & dparp2_refs
    d_only_refs = dcmn_refs - all_refs
    r_only_refs = cmn_refs - dcmn_refs
    print(f"\nReference counts from direct target queries:")
    print(f"    {len(dparp1_refs)} with PARP1 data")
    print(f"    {len(dparp2_refs)} with PARP2 data")
    print(f"    {len(dcmn_refs)} with both")
    print(f"    {len(d_only_refs)} with both not included in Excelra list")
    print(f"    {len(r_only_refs)} with both in Excelra list, not found by direct target query")

# ---------------------------------------------------------------------------------------------------------------------------------
def aggregate_selectivity_data(sel_df=None, db='gostar', db_version='2022-06-20'):
    """
    Compute a mean log selectivity value for each compound. Do this two different ways to see if separating the data by reference
    makes a difference:
     - averaging the log selectivities already computed for each reference
     - averaging all PARP1 pIC50 or pKi values for each compound and subtracting the corresponding average PARP2 values.
    """
    if sel_df is None:
        selectivity_file = f"{selectivity_dir}/{db}/parp1_parp2_selectivity_curated_{db}_{db_version}.csv"
        sel_df = pd.read_csv(selectivity_file)

    # Assign a unique compound_id for each base SMILES string
    # TODO: Is this already done at the curation step, in a DB-specific fashion? If so, delete the code below.
    smiles_idlist_map = {}
    for compound_id, smiles in zip(sel_df.compound_id.values, sel_df.base_rdkit_smiles.values):
        smiles_idlist_map.setdefault(smiles, []).append(compound_id)
    smiles_id_map = {}
    for smiles, idlist in smiles_idlist_map.items():
        smiles_id_map[smiles] = f"gvk_{min(idlist)}"
    sel_df['compound_id'] = [smiles_id_map[smiles] for smiles in sel_df.base_rdkit_smiles.values]

    # Average the per-reference log selectivity values
    #avg_sel_df = sel_df[['base_rdkit_smiles', 'log_selectivity']].groupby('base_rdkit_smiles').mean().reset_index()
    agg_sel_df = aggregate_assay_data(sel_df, value_col='log_selectivity', label_actives=False, id_col='compound_id',
                                      smiles_col='base_rdkit_smiles', relation_col='log_selectivity_relation')
    agg_sel_df = agg_sel_df.rename(columns={'relation': 'log_selectivity_relation'})

    # Average the values for each target and then subtract the PARP2 averages from the PARP1 averages
    act_types = ['pIC50', 'pKi']
    targets = ['PARP1', 'PARP2']
    for act_type in act_types:
        act_df = sel_df[sel_df.activity_type == act_type].copy()
        for target in targets:
            #targ_act_df = remove_outlier_replicates(act_df, response_col=f"{target}_activity", max_diff_from_median=1.0)
            agg_df = aggregate_assay_data(act_df, value_col=f"{target}_activity", label_actives=False, id_col='compound_id',
                                                              smiles_col='base_rdkit_smiles', relation_col=f"{target}_relation")
            agg_df = agg_df.rename(columns={'relation': f"{target}_{act_type}_relation", f"{target}_activity": f"{target}_{act_type}"})
            agg_df = agg_df.drop(columns='base_rdkit_smiles')
            #agg_data[act_type][target] = agg_df
            agg_sel_df = agg_sel_df.merge(agg_df, how='left', on='compound_id')
        agg_sel_df[f"{act_type}_diff"] = agg_sel_df[f"PARP1_{act_type}"].values - agg_sel_df[f"PARP2_{act_type}"].values
    return agg_sel_df


# ---------------------------------------------------------------------------------------------------------------------------------
def filter_ucsf_data(db="ucsf", db_version="2022-06-21"):
    """
    Filter raw PARP1 and PARP2 IC50 data for Otava compound library. This code works with the IC50 values fitted by Amanda,
    which are corrected from those in the HiTS database.
    """
    targets = ['PARP1', 'PARP2']
    ucsf_dir = dict(zip(targets, [f"{dset_dir}/{target}/ucsf" for target in targets]))
    raw_file = dict(
            PARP1="PARP1_otava_pIC50_cur.csv",
            PARP2="PARP2_otava_pIC50_cur.csv")
    
    filt_data = {}
    filt_cols = ['compound_id', 'base_rdkit_smiles', 'PAINS', 'Bruns_Watson_Demerit_Score', 'hill', 'relation', 'pIC50']
    start_activity_id = -1000
    for target in targets:
        target_dir = f"{dset_dir}/{target}"
        raw_file = f"{target_dir}/ucsf/{target}_otava_pIC50_cur.csv"
        raw_df = pd.read_csv(raw_file)
        raw_df['compound_id'] = [f"Otava_{id}" for id in raw_df.Compound.values]
        raw_df['base_rdkit_smiles'] = base_smiles_from_smiles(raw_df.SMILES.values.tolist(), workers=12)
        # The 'relation' values in this dataset are for the reported IC50s. Invert them for pIC50s.
        raw_df['orig_activity_prefix'] = raw_df.relation.values
        raw_df.loc[~raw_df.relation.isna(), 'relation'] = '<'
        raw_df.loc[raw_df.relation.isna(), 'relation'] = '='

        # Keep only records for which the logistic curve fit failed (in which case the pIC50 is left-censored), or
        # succeeded and the Hill slope is positive (i.e., the % inhibition *increases* with increasing concentration).
        filt_df = raw_df[raw_df.hill.isna() | (raw_df.hill > 0)]
        filt_data[target] = filt_df[filt_cols].copy()

        filt_file = f"{target_dir}/ucsf/{target}_Otava_library_pIC50_filtered.csv"
        filt_df.to_csv(filt_file, columns=filt_cols, index=False)
        print(f"Wrote filtered {target} data to {filt_file}")

        # Generate another table in the format we've used for curated GoStar and ChEMBL data, prior to aggregation
        cur_df = filt_df.copy().rename(columns={
                                        'pIC50': 'activity_value', 
                                        'IC50': 'orig_activity_value',
                                        'SMILES': 'original_smiles',
                                        })
        cur_df['orig_activity_type'] = 'IC50'
        cur_df['orig_activity_uom'] = 'uM'
        cur_df['activity_type'] = 'pIC50'
        cur_df['assay_type'] = 'F'
        # Add some fake activity IDs
        cur_df['activity_id'] = start_activity_id - np.array(list(range(len(cur_df))))
        start_activity_id -= 1000
        cur_df['cells_cellline_organ'] = np.nan
        cur_df['enzyme_cell_assay'] = np.nan
        cur_df['compound_name'] = np.nan
        cur_df['db'] = db
        cur_df['db_version'] = db_version
        cur_df['ref_id'] = np.nan
        cur_df['reference'] = 'unpublished UCSF data'
        cur_df['species'] = 'Human'

        cur_cols = ['activity_id', 'compound_id', 'compound_name', 'base_rdkit_smiles',
                    'activity_type', 'relation', 'activity_value',
                    'orig_activity_type', 'orig_activity_prefix', 'orig_activity_value', 'orig_activity_uom',
                    'assay_type', 'enzyme_cell_assay', 'cells_cellline_organ', 'species',
                    'ref_id', 'reference', 'db', 'db_version']

        #cur_file = f"{ucsf_dir[target]}/{target}_Otava_library_pIC50_curated.csv"
        cur_file = f"{target_dir}/ucsf/{target}_pIC50_filtered_{db}_{db_version}.csv"
        cur_df.to_csv(cur_file, columns=cur_cols, index=False)
        print(f"Wrote curated {target} data to {cur_file}")

    return filt_data

# ---------------------------------------------------------------------------------------------------------------------------------
def curate_ucsf_parp_selectivity_data():
    """
    Curate PARP1 and PARP2 data for Otava compound library, downloaded from UCSF's HiTS database. Retain records for which the
    logistic curve fit failed, treating the pIC50 values as censored; but exclude records with a negative Hill slope, indicating
    the presence of fluorescence or other artifacts. Average replicate values (which in this dataset are present only for the control
    compound AZD5305). Generate log selectivity data for the compounds that have both PARP1 and PARP2 data after the exclusions.
    Shuffle all datasets.
    """
    targets = ['PARP1', 'PARP2']
    ucsf_dir = dict(zip(targets, [f"{dset_dir}/{target}/ucsf" for target in targets]))
    ucsf_dir['PARP1_selectivity'] = f"{dset_dir}/PARP1_selectivity"
    raw_file = dict(
            PARP1="PARP1_otava_pIC50_cur.csv",
            PARP2="PARP2_otava_pIC50_cur.csv")
    
    agg_data = {}
    filt_cols = ['compound_id', 'base_rdkit_smiles', 'relation', 'pIC50']
    for target in targets:
        raw_df = pd.read_csv(f"{ucsf_dir[target]}/{raw_file[target]}")
        raw_df['compound_id'] = [f"Otava_{id}" for id in raw_df.Compound.values]
        raw_df['base_rdkit_smiles'] = base_smiles_from_smiles(raw_df.SMILES.values.tolist(), workers=12)
        raw_df['relation'] = '='

        # Keep only records for which the Hill slope is positive (i.e., the % inhibition *increases* with increasing concentration)
        censored_df = raw_df[raw_df.notes == 'Fit FAILED']
        censored_df['relation'] = '<'
        censored_df['pIC50'] = 6 - np.log10(39.8)
        uncensored_df = raw_df[(raw_df.notes != 'Fit FAILED') & (raw_df.hill > 0)]
        filt_df = pd.concat([censored_df, uncensored_df], ignore_index=True)
        agg_df = aggregate_assay_data(filt_df, value_col='pIC50', label_actives=False, id_col='compound_id',
                                                            smiles_col='base_rdkit_smiles', relation_col='relation').sample(frac=1.0)

        agg_data[target] = agg_df

        agg_file = f"{ucsf_dir[target]}/{target}_Otava_library_pIC50_agg.csv"
        agg_df.to_csv(agg_file, index=False)
        print(f"Wrote curated, averaged {target} data to {agg_file}")

    # Generate log selectivity data
    parp1_df = agg_data['PARP1'].rename(columns={'relation': 'PARP1_relation', 'pIC50': 'PARP1_pIC50'})
    parp2_df = agg_data['PARP2'].rename(columns={'relation': 'PARP2_relation', 'pIC50': 'PARP2_pIC50'}).drop(columns=['base_rdkit_smiles'])
    sel_df = parp1_df.merge(parp2_df, how='inner', on='compound_id')
    sel_df['log_selectivity'] = sel_df.PARP1_pIC50.values - sel_df.PARP2_pIC50.values
    # Figure out relation of true selectivity values to reported values.
    sel_df = determine_log_selectivity_relation(sel_df)
    agg_data['log_selectivity'] = sel_df
    sel_file = f"{ucsf_dir['PARP1_selectivity']}/Otava_library_log_selectivity_agg.csv"
    sel_df.to_csv(sel_file, index=False)
    print(f"Wrote curated, averaged log selectivity data to {sel_file}")

    return agg_data

# ---------------------------------------------------------------------------------------------------------------------------------
def combine_chembl_gostar_ucsf_data(chembl_version='30', gostar_version='2022-06-20'):
    """
    Load unique IC50 data from GoStar for the given target, convert to common units and salt-strip and standardize
    SMILES strings. Combine the cleaned-up data with corresponding data from ChEMBL.
    """
    chembl_data = {}
    gostar_data = {}
    targets = ['PARP1', 'PARP2']
    for target in targets:
        target_dir = f"{dset_dir}/{target}"
        gostar_dir = f"{dset_dir}/{target}/gostar"
        gostar_file = f"{gostar_dir}/{target}_IC50_gostar_{gostar_version}.csv"
        gostar_df = pd.read_csv(gostar_file)
        before = len(gostar_df)
        gostar_df = dcf.exclude_organometallics(gostar_df, smiles_col='base_rdkit_smiles')
        after = len(gostar_df)
        print(f"Excluded {before-after} organometallic molecules from GoStar {target} data")
        gostar_df = gostar_df.rename(columns={'gvk_id': 'compound_id', 'activity_value': 'pIC50'})
        gostar_cur_df = gostar_df[['compound_id', 'base_rdkit_smiles', 'relation', 'pIC50']].copy()
        gostar_cur_df['source'] = 'GoStar'
        gostar_data[target] = gostar_cur_df
    
        chembl_dir = f"{dset_dir}/{target}/chembl"
        chembl_file = f"{chembl_dir}/{target}_IC50_chembl{chembl_version}.csv"
        chembl_df = pd.read_csv(chembl_file)
        chembl_df = chembl_df[~chembl_df.rdkit_smiles.isna()]
        before = len(chembl_df)
        chembl_df = dcf.exclude_organometallics(chembl_df)
        after = len(chembl_df)
        print(f"Excluded {before-after} organometallic molecules from ChEMBL {target} data")
        chembl_df['base_rdkit_smiles'] = base_smiles_from_smiles(chembl_df.rdkit_smiles.values.tolist(), workers=16)
        chembl_df = chembl_df[chembl_df.standard_units == 'nM']
        chembl_df = dcf.standardize_relations(chembl_df, db='ChEMBL')
        chembl_df['relation'] = [inverse_relation[r] for r in chembl_df.standard_relation.values]
        chembl_df['pIC50'] = 9.0 - np.log10(chembl_df.standard_value.values)
        chembl_cur_df = chembl_df[['mol_chembl_id', 'relation', 'pIC50', 'rdkit_smiles', 'base_rdkit_smiles']].rename(
                    columns={'mol_chembl_id': 'compound_id'}).copy()
        chembl_cur_df['source'] = 'ChEMBL'
        chembl_data[target] = chembl_cur_df

    # Select or generate a unique compound ID for each base SMILES string. Use the lexicographically smallest ChEMBL
    # ID for each SMILES by default; then generate IDs for the remaining GoStar compounds that aren't also in ChEMBL
    chembl_df = pd.concat([chembl_data[target] for target in targets], ignore_index=True)
    chembl_df = chembl_df.sort_values(by='compound_id').drop_duplicates(subset=['base_rdkit_smiles'])
    smiles_id_map = dict(zip(chembl_df.base_rdkit_smiles.values, chembl_df.compound_id.values))
    id_idx = 0
    combined_data = {}
    for target in targets:
        chembl_df = chembl_data[target]
        chembl_df['compound_id'] = [smiles_id_map[smiles] for smiles in chembl_df.base_rdkit_smiles.values]
        #chembl_data[target] = chembl_df

        gostar_df = gostar_data[target]
        ids = []
        for smiles in gostar_df.base_rdkit_smiles.values:
            try:
                cmpd_id = smiles_id_map[smiles]
            except KeyError:
                cmpd_id = f"PARP_{id_idx}"
                smiles_id_map[smiles] = cmpd_id
                id_idx += 1
            ids.append(cmpd_id)
        gostar_df['compound_id'] = ids
        #gostar_data[target] = gostar_df

        combined_data[target] = combined_df = pd.concat([chembl_df, gostar_df], ignore_index=True)

    return combined_data


# ---------------------------------------------------------------------------------------------------------------------------------
def load_gostar_custom_parp_data():
    """
    Process raw data provided by Excelra for custom PARP selectivity curation project.
    """
    custom_dir = f"{dset_dir}/PARP1_selectivity/gostar_custom"
    excel_files = [f for f in os.listdir(custom_dir) if (f.endswith('.xls') or f.endswith('.xlsx'))]
    references = []
    assay_ids = []
    targets = []
    sources = []
    activity_types = []
    activity_units = []
    assay_descrs = []
    assay_cell_lines = []
    assay_types = []

    act_data = {}

    for fname in excel_files:
        path = f"{custom_dir}/{fname}"
        act_df = pd.read_excel(path, 0)
        reference = act_df.REFERENCE.values[0]
        act_data[reference] = act_df

        assay_df = pd.read_excel(path, 1)
        assay_df = assay_df[~assay_df.PROTEIN.isna()]
        for row in assay_df.itertuples():
            references.append(reference)
            assay_ids.append(row.ASSAY_ID)
            targets.append(row.PROTEIN)
            sources.append(row.SOURCE)
            activity_types.append(row.ACTIVITY_TYPE)
            activity_units.append(row.ACTIVITY_UOM)
            assay_descrs.append(row.ENZYME_CELL_ASSAY)
            assay_cell_lines.append(row.ARTICLE_CELL_LINE)
            assay_types.append(row.ASSAY_TYPE)

    assay_df = pd.DataFrame(dict(
                    reference=references, assay_id=assay_ids, target=targets, species=sources, 
                    activity_type=activity_types, activity_uom=activity_units, cell_line=assay_cell_lines, 
                    assay_type=assay_types, description=assay_descrs))
    # Filter assays to relevant ones for PARP1 & 2
    parp_assay_df = assay_df[assay_df.target.isin(['PARP1', 'PARP-1', 'PARP2', 'PARP-2']) &
                             assay_df.activity_type.isin(['IC50', 'pIC50', 'Ki', 'pKi'])]

    # Create a combined table of activities from all the references with relevant assays. Try to mimic the format we use
    # for the raw data queries from GoStar.
    activity_ids = []
    gvk_ids = []
    compound_names = []
    sub_smiles = []
    activity_types = []
    activity_prefixes = []
    activity_values = []
    activity_units = []
    assay_types = []
    assay_descrs = []
    assay_ids = []
    targets = []
    sources = []
    assay_cell_lines = []
    ref_ids = []
    references = []

    act_id = 0
    # Iterate over reference-specific assays
    for assay in parp_assay_df.itertuples():
        act_df = act_data[assay.reference]
        # Find the columns that have activity data and prefixes for this assay. There may be multiple columns,
        # containing replicate measurements or measurements in different units. The units are obtained by parsing
        # the activity column headers.
        act_cols = []
        prefix_cols = []
        for col in act_df.columns.values:
            if col.startswith(str(assay.assay_id)):
                if 'PREFIX' in col:
                    prefix_cols.append(col)
                elif not ('REMARKS' in col):
                    act_cols.append(col)
        if len(act_cols) == 0:
            raise Exception(f"Failed to find activity column for assay_id {assay.assay_id}")
        activities = []
        prefixes = []
        units = []
        for act_col, prefix_col in zip(act_cols, prefix_cols):
            col_fields = act_col.split('|')
            act_name = col_fields[1]
            act_units = act_name.split('_')[-1]
            activities.append(act_df[act_col])
            prefixes.append(act_df[prefix_col])
            units.append(act_units)

        # Iterate over compounds for each reference
        for i, row in enumerate(act_df.itertuples()):
            for j, act_col in enumerate(act_cols):
                if not np.isnan(activities[j][i]):
                    activity_prefixes.append(prefixes[j][i])
                    activity_values.append(activities[j][i])
                    activity_units.append(units[j])
                    # Generate fake negative activity IDs. This is OK, since they don't appear in the final curated data.
                    act_id -= 1
                    activity_ids.append(act_id)
                    gvk_ids.append(row.GVK_ID)
                    compound_names.append(row.COMPOUND_ID)
                    sub_smiles.append(row.SMILES)
                    activity_types.append(assay.activity_type)
                    assay_types.append(assay.assay_type)
                    assay_descrs.append(assay.description)
                    assay_ids.append(assay.assay_id)
                    if assay.target in ['PARP1', 'PARP-1']:
                        targets.append('PARP1')
                    elif assay.target in ['PARP2', 'PARP-2']:
                        targets.append('PARP2')
                    sources.append(assay.species)
                    assay_cell_lines.append(assay.cell_line)
                    ref_ids.append(row.REF_ID)
                    references.append(assay.reference)

    activity_df = pd.DataFrame(dict(
                    act_id=activity_ids, gvk_id=gvk_ids, compound_name=compound_names, sub_smiles=sub_smiles,
                    activity_prefix=activity_prefixes, activity_value=activity_values, activity_uom=activity_units,
                    assay_type=assay_types, cells_cellline_organ=assay_cell_lines, enzyme_cell_assay=assay_descrs,
                    target=targets, source=sources, activity_type=activity_types, ref_id=ref_ids, reference=references))

    db_version = 'custom_2022-06-23'
    activity_file = f"{custom_dir}/parp1_parp2_selectivity_raw_gostar_{db_version}.csv"
    activity_df.to_csv(activity_file, index=False)
    print(f"Wrote raw custom GoStar PARP1 and PARP2 data to {activity_file}")

    # Split data by target and write to raw data directories
    parp1_df = activity_df[activity_df.target == 'PARP1'].copy()
    parp2_df = activity_df[activity_df.target == 'PARP2'].copy()

    parp1_file = f"{dset_dir}/PARP1/gostar/PARP1_activity_gostar_{db_version}.csv"
    parp1_df.to_csv(parp1_file, index=False)
    print(f"Wrote raw custom GoStar PARP1 data to {parp1_file}")
    parp2_file = f"{dset_dir}/PARP2/gostar/PARP2_activity_gostar_{db_version}.csv"
    parp2_df.to_csv(parp2_file, index=False)
    print(f"Wrote raw custom GoStar PARP2 data to {parp2_file}")

    return activity_df

# ---------------------------------------------------------------------------------------------------------------------------------
def combine_parp_selectivity_data_old(gostar_version='2022-06-20', chembl_version='30'):
    """
    Combine curated ChEMBL and GoStar selectivity data into one table with one row per compound per 
    activity type per reference, containing activity values for PARP1 and PARP2 and log selectivity values.
    """

    chembl_file = f"{selectivity_dir}/chembl/parp1_parp2_selectivity_curated_chembl_{chembl_version}.csv"
    chembl_df = pd.read_csv(chembl_file)
    chembl_df['db'] = 'chembl'
    chembl_df['db_version'] = chembl_version
    gostar_file = f"{selectivity_dir}/gostar/parp1_parp2_selectivity_curated_gostar_{gostar_version}.csv"
    gostar_df = pd.read_csv(gostar_file)
    gostar_df['db'] = 'gostar'
    gostar_df['db_version'] = gostar_version
    gostar_df = gostar_df.rename(columns={'act_id': 'activity_id'})

    # Combine ChEMBL and GoStar records, eliminating duplicates
    cmn_cols = ['activity_id', 'compound_id', 'compound_name', 'base_rdkit_smiles', 'activity_type', 'relation', 'activity_value',
                'assay_type', 'enzyme_cell_assay', 'cells_cellline_organ', 'standard_name', 'target', 'source',
                'ref_id', 'reference', 'title', 'year', 'db', 'db_version']
    chembl_df = chembl_df[cmn_cols].copy()
    gostar_df = gostar_df[cmn_cols].copy()
    cmn_refs = set(chembl_df.reference.values) & set(gostar_df.reference.values)
    uniq_ref_df = pd.concat([chembl_df[~chembl_df.reference.isin(cmn_refs)], gostar_df[~gostar_df.reference.isin(cmn_refs)]], 
                            ignore_index=True)
    cmn_ref_df = pd.concat([chembl_df[chembl_df.reference.isin(cmn_refs)], gostar_df[gostar_df.reference.isin(cmn_refs)]], 
                            ignore_index=True)

    # Load discrepancy resolution file, use it to select duplicates to remove 
    resolution_file = f"{selectivity_dir}/parp1_parp2_gostar_chembl_discrepancy_resolution.csv"
    resolution_df = pd.read_csv(resolution_file)
    removed_ids = set(resolution_df[resolution_df.remove == 1].activity_id.values)
    cmn_ref_df = cmn_ref_df[~cmn_ref_df.activity_id.isin(removed_ids)]

    # Eliminate additional duplicates or near-duplicates between ChEMBL and GoStar versions of data from same references

    def eliminate_dups(df):
        """
        Called on each slice of cmn_ref_df grouped by reference, base SMILES, target and activity type. If slice has records
        from both ChEMBL and GoStar, return the ChEMBL rows only. Otherwise, return the whole slice.
        """
        chembl_part = df[df.db == 'chembl']
        gostar_part = df[df.db == 'gostar']
        if (len(chembl_part) > 0) and (len(gostar_part) > 0):
            return chembl_part
        else:
            return df

    undup_df = cmn_ref_df.groupby(['reference', 'base_rdkit_smiles', 'target', 'activity_type']).apply(eliminate_dups)

    # Combine the deduplicated data with the data from references that were unique to GoStar or ChEMBL
    filt_df = pd.concat([uniq_ref_df, undup_df], ignore_index=True)

    # Assign a unique compound_id for each base SMILES string
    smiles_id_set_map = {}
    for compound_id, smiles in zip(filt_df.compound_id.values, filt_df.base_rdkit_smiles.values):
        smiles_id_set_map.setdefault(smiles, set()).add(compound_id)
    smiles_id_map = {}
    for smiles, id_set in smiles_id_set_map.items():
        smiles_id_map[smiles] = sorted(id_set)[0]
    filt_df['compound_id'] = [smiles_id_map[smiles] for smiles in filt_df.base_rdkit_smiles.values]

    act_types = ['pIC50', 'pKi']
    targets = ['PARP1', 'PARP2']

    sel_data = []

    # Calculate average log selectivities for each unique reference/compound/activity_type. Generate a new data frame
    # with columns for PARP1 & PARP2 activities and log selectivities.
    for act_type in act_types:
        act_df = filt_df[filt_df.activity_type == act_type]
        act_refs = sorted(set(act_df.reference.values))
        for ref in act_refs:
            act_ref_df = act_df[act_df.reference == ref]
            ref_smiles = set(act_ref_df.base_rdkit_smiles.values)
            targ_smiles = {}
            for target in targets:
                targ_df = act_ref_df[act_ref_df.target == target]
                targ_smiles[target] = set(targ_df.base_rdkit_smiles.values)
            cmn_smiles = targ_smiles['PARP1'] & targ_smiles['PARP2']
            if cmn_smiles != ref_smiles:
                print(f"Reference '{ref}' has {len(ref_smiles)} unique SMILES but only {len(cmn_smiles)} with data for both PARP1 and PARP2")
            cmn_act_df = act_ref_df[act_ref_df.base_rdkit_smiles.isin(cmn_smiles)]
            agg_data = {}
            for target in targets:
                targ_df = cmn_act_df[cmn_act_df.target == target]
                ref_targ_assays = sorted(set(targ_df.enzyme_cell_assay.values))
                #if len(ref_targ_assays) > 1:
                #    print(f"Warning: more than one {target} {act_type} assay for reference {ref}:")
                #    for assay_desc in ref_targ_assays:
                #        print(f"    {assay_desc}")
                nrec = len(targ_df)
                targ_df = remove_outlier_replicates(targ_df, response_col='activity_value', id_col='base_rdkit_smiles', 
                                                    max_diff_from_median=0.5)
                nrem = nrec - len(targ_df)
                if nrem > 0:
                    print(f"Removed {nrem} outliers from {ref} {target} {act_type}s")
                if len(targ_df) > 0:
                    agg_data[target] = aggregate_assay_data(targ_df, value_col='activity_value', label_actives=False, id_col='compound_id',
                                                            smiles_col='base_rdkit_smiles', relation_col='relation')
                else:
                    agg_data[target] = None
            # Combine the PARP1 and PARP2 aggregated values into one data frame with one row per compound. We may have lost some
            # compounds when removing outliers, so recalculate the set of compounds with data for both targets.
            parp1_df = agg_data['PARP1']
            parp2_df = agg_data['PARP2']
            if (parp1_df is None) or (parp2_df is None):
                continue
            cmn_smiles = set(parp1_df.base_rdkit_smiles.values) & set(parp2_df.base_rdkit_smiles.values)
            parp1_df = parp1_df[parp1_df.base_rdkit_smiles.isin(cmn_smiles)].sort_values(by='base_rdkit_smiles')
            parp2_df = parp2_df[parp2_df.base_rdkit_smiles.isin(cmn_smiles)].sort_values(by='base_rdkit_smiles')
            sel_df = parp1_df.rename(columns={
                                            'relation': 'PARP1_relation',
                                            'activity_value': 'PARP1_activity'})
            sel_df['PARP2_relation'] = parp2_df.relation.values
            sel_df['PARP2_activity'] = parp2_df.activity_value.values
            sel_df['log_selectivity'] = sel_df.PARP1_activity.values - sel_df.PARP2_activity.values
            sel_df['activity_type'] = act_type
            sel_df['reference'] = ref
            sel_data.append(sel_df)

    sel_df = pd.concat(sel_data, ignore_index=True)

    # Figure out relation of true selectivity values to reported values.
    sel_df = determine_log_selectivity_relation(sel_df)

    print(f"Combined selectivity data has {len(sel_df)} records, {len(set(sel_df.base_rdkit_smiles.values))} compounds")
    selectivity_file = f"{selectivity_dir}/parp1_parp2_selectivity_combined_chembl_{chembl_version}_gostar_{gostar_version}.csv"
    sel_df.to_csv(selectivity_file, index=False)
    print(f"Wrote combined selectivity data to {selectivity_file}")

    return sel_df

# ---------------------------------------------------------------------------------------------------------------------------------
def get_raw_gostar_parp_selectivity_data(db_version='2022-06-20'):
    """
    Read data from querying latest version of GoStar for PARP1 and PARP2 activity data, allowing
    "source" (species) to be either Human or NULL. Find references that have both PARP1 and PARP2
    data, and then identify compounds with both kinds of measurements in the same reference.
    Extract the pKi and pIC50 data for these compounds.
    """
    # Read data obtained by querying against standard PARP1 and PARP2 target names
    parp1_df = pd.read_csv(f"{dset_dir}/PARP1/gostar/PARP1_activity_gostar_{db_version}.csv")
    parp2_df = pd.read_csv(f"{dset_dir}/PARP2/gostar/PARP2_activity_gostar_{db_version}.csv")

    # Standardize abbreviations in references to not end with periods. Leave URL references alone since dots are significant.
    parp1_df['reference'] = [ref if ref.startswith('http') else ref.replace('.', '') for ref in parp1_df.reference.values]
    parp2_df['reference'] = [ref if ref.startswith('http') else ref.replace('.', '') for ref in parp2_df.reference.values]

    parp1_refs = set(parp1_df.reference.values)
    parp2_refs = set(parp2_df.reference.values)
    cmn_refs = list(parp1_refs & parp2_refs)

    parp1_data = []
    parp2_data = []
    parp1_df['base_rdkit_smiles'] = base_smiles_from_smiles(parp1_df.sub_smiles.values.tolist(), workers=16)
    parp2_df['base_rdkit_smiles'] = base_smiles_from_smiles(parp2_df.sub_smiles.values.tolist(), workers=16)

    # Exclude records with SMILES that RDKit didn't like
    parp1_df = parp1_df[parp1_df.base_rdkit_smiles != '']
    parp2_df = parp2_df[parp2_df.base_rdkit_smiles != '']

    # Find compounds with both PARP1 and PARP2 data from same references
    selectivity_refs = []
    ref_activity_types = []
    ref_parp1_rec_counts = []
    ref_parp2_rec_counts = []
    ref_parp1_cmpd_counts = []
    ref_parp2_cmpd_counts = []
    for ref in cmn_refs:
        ref_parp1_df = parp1_df[parp1_df.reference == ref]
        ref_parp2_df = parp2_df[parp2_df.reference == ref]
        ref_cmn_smiles = set(ref_parp1_df.base_rdkit_smiles.values) & set(ref_parp2_df.base_rdkit_smiles.values)
        if len(ref_cmn_smiles) > 0:
            selectivity_refs.append(ref)
            ref_parp1_df = ref_parp1_df[ref_parp1_df.base_rdkit_smiles.isin(ref_cmn_smiles)]
            ref_parp1_rec_counts.append(len(ref_parp1_df))
            ref_parp1_cmpd_counts.append(len(set(ref_parp1_df.base_rdkit_smiles.values)))

            ref_parp2_df = ref_parp2_df[ref_parp2_df.base_rdkit_smiles.isin(ref_cmn_smiles)]
            ref_parp2_rec_counts.append(len(ref_parp2_df))
            ref_parp2_cmpd_counts.append(len(set(ref_parp2_df.base_rdkit_smiles.values)))
            parp1_data.append(ref_parp1_df)
            parp2_data.append(ref_parp2_df)
            act_types = '; '.join(sorted(set(ref_parp1_df.activity_type.values) & set(ref_parp2_df.activity_type.values)))
            ref_activity_types.append(act_types)

    ref_act_df = pd.DataFrame(dict(reference=selectivity_refs, activity_types=ref_activity_types,
                                   PARP1_records=ref_parp1_rec_counts, PARP1_compounds=ref_parp1_cmpd_counts,
                                   PARP2_records=ref_parp2_rec_counts, PARP2_compounds=ref_parp2_cmpd_counts
                                   ))

    ref_act_file = f"{selectivity_dir}/gostar/parp1_parp2_selectivity_ref_stats_gostar_{db_version}.csv"
    ref_act_df.to_csv(ref_act_file, index=False)
    print(f"Wrote selectivity reference stats to {ref_act_file}")

    parp1_df = pd.concat(parp1_data, ignore_index=True)
    parp1_df['target'] = 'PARP1'
    parp2_df = pd.concat(parp2_data, ignore_index=True)
    parp2_df['target'] = 'PARP2'

    n_uniq_cmpds_1 = len(set(parp1_df.base_rdkit_smiles.values))
    n_uniq_cmpds_2 = len(set(parp2_df.base_rdkit_smiles.values))
    print(f"\n{len(selectivity_refs)} references actually have PARP1 & 2 data for the same compounds.")

    print(f"    {len(parp1_df)} records of PARP1 data")
    print(f"    {len(parp2_df)} records of PARP2 data")

    print(f"    {n_uniq_cmpds_1} unique SMILES in PARP1 data")
    print(f"    {n_uniq_cmpds_2} unique SMILES in PARP2 data")

    # Save the data for all data types
    all_types_df = pd.concat([parp1_df, parp2_df], ignore_index=True)
    all_types_file = f"{selectivity_dir}/gostar/parp1_parp2_selectivity_all_types_gostar_{db_version}.csv"
    all_types_df.to_csv(all_types_file, index=False)
    print(f"Wrote combined data for all types to {all_types_file}")

    # Filter the selectivity data by activity type; only include IC50/pIC50 and Ki/pKi measurements. Convert everything
    # to pIC50 or pKi.
    sel_types = ['Ki', 'pKi', 'inhibition constant (Ki)', 'IC50', 'pIC50']
    sel_units = ['nM', 'nmol/L', 'uM']
    raw_df = all_types_df[all_types_df.activity_type.isin(sel_types) & 
                   (all_types_df.activity_uom.isna() | all_types_df.activity_uom.isin(sel_units))].copy()

    raw_df.loc[raw_df.activity_uom.isin(['nM', 'nmol/L']), 'activity_value'] = 9.0 - np.log10(
                                                  raw_df.loc[raw_df.activity_uom.isin(['nM', 'nmol/L']), 'activity_value'].values)
    raw_df.loc[raw_df.activity_uom == 'uM', 'activity_value'] = 6.0 - np.log10(
                                                  raw_df.loc[raw_df.activity_uom == 'uM', 'activity_value'].values)
    raw_df.loc[raw_df.activity_type.isin(['Ki', 'inhibition constant (Ki)']), 'activity_type'] = 'pKi'
    raw_df.loc[raw_df.activity_type == 'IC50', 'activity_type'] = 'pIC50'

    # Standardize the relational operators. For activity types that weren't already negative log values, invert them as well.
    neglog_df = raw_df[raw_df.activity_uom.isna()].copy()
    neglog_df = dcf.standardize_relations(neglog_df, db='GoStar', output_rel_col='relation')
    needs_inv_df = raw_df[~raw_df.activity_uom.isna()].copy()
    needs_inv_df = dcf.standardize_relations(needs_inv_df, db='GoStar', output_rel_col='relation', invert=True)
    raw_df = pd.concat([neglog_df, needs_inv_df], ignore_index=True)

    raw_df['activity_uom'] = np.nan

    raw_file = f"{selectivity_dir}/gostar/parp1_parp2_selectivity_raw_gostar_{db_version}.csv"
    raw_df.to_csv(raw_file, index=False)
    print(f"Wrote raw converted pIC50 and pKi data to {raw_file}")

    return raw_df

# ---------------------------------------------------------------------------------------------------------------------------------
def get_gostar_parp_assays_by_reference(db_version='2022-06-20'):
    """
    Tabulate the references and assays for PARP1 and PARP2 pIC50 and pKi data from GoStar
    """
    raw_file = f"{selectivity_dir}/gostar/parp1_parp2_selectivity_raw_gostar_{db_version}.csv"
    raw_df = pd.read_csv(raw_file)
    ref_cols = ['target', 'ref_id', 'reference', 'year', 'title', 'activity_type', 'standard_name', 'source', 'enzyme_cell_assay',
                'cells_cellline_organ', 'assay_type', 'target_id', 'base_rdkit_smiles']
    ref_df = raw_df[ref_cols].copy()

    def count_recs_and_compounds(group):
        assay_df = group.head(1).copy()
        assay_df[f"num_assay_records"] = len(group)
        assay_df[f"num_assay_cmpds"] = len(set(group.base_rdkit_smiles.values))
        return assay_df

    def count_assays(group):
        ref_assay_df = group.head(1).copy()
        ref_assay_df[f"num_ref_assays"] = len(set(group.enzyme_cell_assay.values))
        return ref_assay_df

    def summarize_by_target(targ_df):
        targ_assay_df = targ_df.groupby(['target', 'ref_id', 'enzyme_cell_assay'], as_index=False).apply(count_recs_and_compounds)
        targ_ref_df = targ_assay_df[['target', 'ref_id', 'enzyme_cell_assay']].groupby('ref_id', as_index=False).apply(count_assays).drop(
                            columns=['enzyme_cell_assay']).drop(columns=['target'])
        targ_assay_df = targ_assay_df.merge(targ_ref_df, how='left', on='ref_id')
        return targ_assay_df


    assay_summary_df = ref_df.groupby('target', as_index=False).apply(summarize_by_target).drop(
                                    columns=['base_rdkit_smiles', 'standard_name', 'target_id'])
    assay_summary_file = f"{selectivity_dir}/gostar/parp1_parp2_selectivity_assays_by_reference_gostar_{db_version}.csv"
    assay_summary_df.to_csv(assay_summary_file, index=False)
    print(f"Wrote GoStar assay summary data to {assay_summary_file}")

    return assay_summary_df


# ---------------------------------------------------------------------------------------------------------------------------------
def get_chembl_parp_assays_by_reference(db_version=None):
    """
    Tabulate the references and assays for PARP1 and PARP2 pIC50 and pKi data from GoStar
    """
    if db_version is None:
        db_version = chembl.get_current_version()

    raw_file = f"{selectivity_dir}/chembl/parp1_parp2_selectivity_raw_chembl_{db_version}.csv"
    raw_df = pd.read_csv(raw_file)
    # Map ChEMBL column names to corresponding GoStar columns, so we can combine tables later
    raw_df = raw_df.rename(columns={
                    'doc_chembl_id': 'ref_id',
                    'pref_name': 'standard_name',
                    'assay_organism': 'source',
                    'description': 'enzyme_cell_assay',
                    'assay_cell_type': 'cells_cellline_organ',
                    'assay_chembl_id': 'assay_id',
                })
    ref_cols = ['target', 'ref_id', 'reference', 'year', 'title', 'activity_type', 'standard_name', 'source', 
                'assay_id', 'enzyme_cell_assay', 'cells_cellline_organ', 'assay_type', 'base_rdkit_smiles']
    ref_df = raw_df[ref_cols].copy()

    def count_recs_and_compounds(group):
        assay_df = group.head(1).copy()
        assay_df[f"num_assay_records"] = len(group)
        assay_df[f"num_assay_cmpds"] = len(set(group.base_rdkit_smiles.values))
        return assay_df

    def count_assays(group):
        ref_assay_df = group.head(1).copy()
        ref_assay_df[f"num_ref_assays"] = len(set(group.enzyme_cell_assay.values))
        return ref_assay_df

    def summarize_by_target(targ_df):
        targ_assay_df = targ_df.groupby(['target', 'ref_id', 'assay_id'], as_index=False).apply(count_recs_and_compounds)
        targ_ref_df = targ_assay_df[['target', 'ref_id', 'assay_id', 'enzyme_cell_assay']].groupby('ref_id', as_index=False).apply(
                            count_assays).drop(columns=['target', 'assay_id', 'enzyme_cell_assay'])
        targ_assay_df = targ_assay_df.merge(targ_ref_df, how='left', on='ref_id')
        return targ_assay_df


    assay_summary_df = ref_df.groupby('target', as_index=False).apply(summarize_by_target).drop(
                                    columns=['base_rdkit_smiles', 'assay_id', 'standard_name'])
    assay_summary_file = f"{selectivity_dir}/chembl/parp1_parp2_selectivity_assays_by_reference_chembl{db_version}.csv"
    assay_summary_df.to_csv(assay_summary_file, index=False)
    print(f"Wrote ChEMBL assay summary data to {assay_summary_file}")

    return assay_summary_df

# ---------------------------------------------------------------------------------------------------------------------------------
def combine_assay_tables_by_reference():
    """
    Combine ChEMBL and GoStar selectivity assay tables into two files: one with assays from the references that are common to both
    ChEMBL and GoStar, the other with duplicates removed, giving precedence to the ChEMBL versions.
    """
    chembl_assay_df = get_chembl_parp_assays_by_reference()
    gostar_assay_df = get_gostar_parp_assays_by_reference()
    gostar_refs = set(gostar_assay_df.reference.values)
    chembl_refs = set(chembl_assay_df.reference.values)
    cmn_refs = gostar_refs & chembl_refs
    chembl_assay_df['db'] = 'chembl'
    gostar_assay_df['db'] = 'gostar'
    gostar_uniq_df = gostar_assay_df[~gostar_assay_df.reference.isin(cmn_refs)].copy()
    gostar_uniq_df['in_both_dbs'] = False
    chembl_assay_df['in_both_dbs'] = chembl_assay_df.reference.isin(cmn_refs)
    union_df = pd.concat([chembl_assay_df, gostar_uniq_df], ignore_index=True)
    union_file = f"{selectivity_dir}/parp1_parp2_selectivity_assays_gostar_chembl_union.csv"
    union_df.to_csv(union_file, index=False)
    print(f"Wrote union reference table to {union_file}")

    common_df = pd.concat([
                        gostar_assay_df[gostar_assay_df.reference.isin(cmn_refs)],
                        chembl_assay_df[chembl_assay_df.reference.isin(cmn_refs)]
                        ], ignore_index=True).sort_values(by=['reference', 'db'])
    common_file = f"{selectivity_dir}/parp1_parp2_selectivity_assays_gostar_chembl_common_refs.csv"
    common_df.to_csv(common_file, index=False)
    print(f"Wrote common reference table to {common_file}")


# ---------------------------------------------------------------------------------------------------------------------------------
def curate_parp_selectivity_data(raw_df=None, db='gostar', db_version='2022-06-20'):
    """
    Process raw GoStar or ChEMBL pIC50 and pKi data for PARP1 and PARP2 to yield a table with one row per compound per 
    activity type per reference, containing activity values for PARP1 and PARP2 and log selectivity values.
    """
    db_version = str(db_version)
    if raw_df is None:
        if db == 'chembl':
            raw_df = get_raw_chembl_parp_selectivity_data(db_version)
        else:
            raw_df = get_raw_gostar_parp_selectivity_data(db_version)

    if db.startswith('gostar'):
        raw_df['compound_id'] = [f"gvk_{id}" for id in raw_df.gvk_id.values]
    else:
        raw_df = raw_df.rename(columns={'mol_chembl_id': 'compound_id'})

    if db == 'chembl':
        # Map ChEMBL column names to corresponding GoStar columns, so we can combine tables later
        raw_df = raw_df.rename(columns={
                        'doc_chembl_id': 'ref_id',
                        'pref_name': 'standard_name',
                        'assay_organism': 'source',
                        'description': 'enzyme_cell_assay',
                        'assay_cell_type': 'cells_cellline_organ',
                        'assay_chembl_id': 'assay_id',
                    })

    # Filter on assay description and cell type. Sf9 and Sf21 cells are OK because they're used to express the PARP proteins, not as part of the inhibition assay.
    # TODO: Other cell-based assays may be OK if the endpoint is a molecular measurement of PARP function, rather than cell death or proliferation. Revisit this later.
    
    desc_exclude = np.array([('domain' in desc) or ('Jurkat cell' in desc) or ('proliferation' in desc) for desc in raw_df.enzyme_cell_assay.values])
    cell_type_exclude = np.array([(type(cells) == str) and not (cells in ['Sf9', 'SF-9', 'Sf21', 'SF-21']) for cells in raw_df.cells_cellline_organ.values])
    curated_df = raw_df.loc[~(desc_exclude | cell_type_exclude)]
    excluded_df = raw_df.loc[desc_exclude | cell_type_exclude]
    excluded_file = f"{selectivity_dir}/{db}/parp1_parp2_selectivity_excluded_{db}_{db_version}.csv"
    excluded_df.to_csv(excluded_file, index=False)
    print(f"Wrote excluded {db} data to {excluded_file}")

    curated_file = f"{selectivity_dir}/{db}/parp1_parp2_selectivity_curated_{db}_{db_version}.csv"
    curated_df.to_csv(curated_file, index=False)
    print(f"Wrote curated {db} data to {curated_file}")

    return curated_df


