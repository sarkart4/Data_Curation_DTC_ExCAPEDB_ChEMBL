"""
Functions for curation of public datasets from ChEMBL and elsewhere
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import seaborn as sns
import pdb

from atomsci.ddm.utils.struct_utils import base_smiles_from_smiles, get_rdkit_smiles
from atomsci.ddm.utils import datastore_functions as dsf
from atomsci.ddm.utils import curate_data as curate
from atomsci.ddm.utils.llnl_utils import is_lc_system

# ----------------------------------------------------------------------------------------------------------------------
# TODO:
#   - Convert values in ug.mL-1 units to nM, rather than discarding.
#   - Add code to download ChEMBL datasets directly using the chembl_webresource_client package
#   - Include ChEMBL dataset version as function parameter, rather than assuming ChEMBL 25


# set_data_root() sets global variables to point to subdirectories of your 
# public_dsets directory. This will typically be /usr/workspace/atom/public_dsets on LC, or
# /usr/local/data/public_dsets on TTB.

def set_data_root(dir):
    global data_root, data_dirs
    data_root = dir
    data_dirs = dict(ChEMBL = '%s/ChEMBL' % data_root, DTC = '%s/DTC' % data_root, 
                     Excape = '%s/Excape' % data_root, Literature = '%s/Literature' % data_root)

# Default data directory setup
if is_lc_system():
    set_data_root("/usr/workspace/atom/public_dsets")
else:
    set_data_root("/usr/local/data/public_dsets")


log_var_map = {
    'IC50': 'pIC50',
    'AC50': 'pIC50',
    'Ki': 'pKi',
    'Solubility': 'logSolubility',
    'CL': 'logCL'
}

chembl_dsets = dict(
    CYP2C9 = dict(AC50='CHEMBL25-CYP2C9_human_AC50_26Nov2019', IC50='CHEMBL25-CYP2C9_human_IC50_26Nov2019'),
    CYP2D6 = dict(AC50='CHEMBL25-CYP2D6_human_AC50_26Nov2019', IC50='CHEMBL25-CYP2D6_human_IC50_26Nov2019'),
    CYP3A4 = dict(AC50='CHEMBL25-CYP3A4_human_AC50_26Nov2019', IC50='CHEMBL25-CYP3A4_human_IC50_26Nov2019'),
    AURKA = dict(IC50="CHEMBL26-AURKA_human_IC50"),
    AURKB = dict(IC50="CHEMBL26-AURKB_human_IC50"),
    BSEP = dict(IC50="CHEMBL26-BSEP_human_IC50"),
    hERG = dict(IC50="CHEMBL26-hERG_human_IC50"),
    JAK1 = dict(IC50="CHEMBL25-JAK1_IC50_human_26Nov2019"),
    JAK2 = dict(IC50="CHEMBL25-JAK2_IC50_human_26Nov2019"),
    JAK3 = dict(IC50="CHEMBL25-JAK3_IC50_human_26Nov2019"),
    PI3Kg = dict(IC50="CHEMBL25-pI3K_p110gamma_human_IC50_26Nov2019"),
    KCNA5 = dict(IC50="CHEMBL26-KCNA5_human_IC50"),
    SCN5A = dict(IC50="CHEMBL26-SCN5A_human_IC50"),
    KCNQ1_KCNE1 = dict(IC50="CHEMBL26-KCNQ1_KCNE1_human_IC50"),
    CACNA1C = dict(IC50="CHEMBL26-CACNA1C_human_IC50"),
    CHRM1 = dict(IC50="CHEMBL26-CHRM1_human_IC50", Ki="CHEMBL26-CHRM1_human_Ki"),
    CHRM2 = dict(IC50="CHEMBL26-CHRM2_human_IC50", Ki="CHEMBL26-CHRM2_human_Ki"),
    CHRM3 = dict(IC50="CHEMBL26-CHRM3_human_IC50", Ki="CHEMBL26-CHRM3_human_Ki"),
    HRH1 = dict(IC50="CHEMBL26-HRH1_human_IC50", Ki="CHEMBL26-HRH1_human_Ki"),
    Solubility="CHEMBL25-Solubility_pH7_4_AstraZenica_26Nov2019",
    hepCL_human="CHEMBL25-hepatocyte_clearance_AZ_26Nov2019",
    hepCL_rat="CHEMBL25-hepatocyte_clearance_rat_AZ_26Nov2019",
    micCL_human="CHEMBL25-microsomal_clearance_human_AZprotocl_26Nov2019"
)
# Phospholipidosis dataset is not included above because the 'Standard Value' column is empty; it's a classification
# dataset where the classes are in the Comment column. Need to treat it specially.

# ----------------------------------------------------------------------------------------------------------------------
# Generic functions for all datasets
# ----------------------------------------------------------------------------------------------------------------------

# Note: Functions freq_table and labeled_freq_table have been moved to ddm.utils.curate_data module.

# ----------------------------------------------------------------------------------------------------------------------
def standardize_relations(dset_df, db='DTC'):
    """
    Standardize the censoring operators to =, < or >, and remove any rows whose operators
    don't map to a standard one.
    """
    relation_cols = dict(ChEMBL='Standard Relation', DTC='standard_relation')
    rel_col = relation_cols[db]

    dset_df[rel_col].fillna('=', inplace=True)
    ops = dset_df[rel_col].values
    if db == 'ChEMBL':
        # Remove annoying quotes around operators
        ops = [op.lstrip("'").rstrip("'") for op in ops]
    op_dict = {
        ">": ">",
        ">=": ">",
        "<": "<",
        "<=": "<",
        "=": "="
    }
    ops = np.array([op_dict.get(op, "@") for op in ops])
    dset_df[rel_col] = ops
    dset_df = dset_df[dset_df[rel_col] != "@"]
    return dset_df


# ----------------------------------------------------------------------------------------------------------------------
# ChEMBL-specific curation functions
# ----------------------------------------------------------------------------------------------------------------------

# Raw ChEMBL datasets as downloaded from the interactive ChEMBL web app are labeled as compressed CSV files,
# but they are actually semicolon-separated. After decompressing them, we change the extension to .txt so we
# can open them in Excel without it getting confused. These files can be placed in data_dirs['ChEMBL']/raw
# for initial processing.


# ----------------------------------------------------------------------------------------------------------------------
def filter_chembl_dset(dset_df, filter_dups=True, filter_out_of_range=True, strip_salts=True):
    """
    Filter rows from a raw dataset downloaded from the ChEMBL website. Standardize censoring relational operators.
    Add columns for the log-transformed value, the censoring relation for the log value, and the base RDKit SMILES
    string. Returns a filtered data frame.
    """

    # Filter out rows with no SMILES string or IC50 data
    dset_df = dset_df[~dset_df['Smiles'].isna()]
    dset_df = dset_df[~dset_df['Standard Value'].isna()]
    # Filter out rows flagged as likely duplicates
    if filter_dups:
        dset_df = dset_df[dset_df['Potential Duplicate'] == False]
    # Filter out rows flagged with validity concerns (e.g., value out of typical range). Note that ChEMBL
    # isn't always right about these being invalid!
    if filter_out_of_range:
        dset_df = dset_df[dset_df['Data Validity Comment'].isna()]

    # Filter out rows with nonstandard measurement types. We assume here that the type appearing
    # most frequently is the standard one.
    type_df = curate.freq_table(dset_df, 'Standard Type')
    max_type = type_df['Standard Type'].values[0]
    if type_df.shape[0] > 1:
        print('Dataset has multiple measurement types')
        print(type_df)
    dset_df = dset_df[dset_df['Standard Type'] == max_type]

    # Filter out rows with nonstandard units. Again, we assume the unit appearing most frequently
    # is the standard one.

    unit_freq_df = curate.freq_table(dset_df, 'Standard Units')
    max_unit = unit_freq_df['Standard Units'].values[0]
    if unit_freq_df.shape[0] > 1:
        print('Dataset has multiple standard units')
        print(unit_freq_df)
        # Special case: Standard unit is nM, but some values are expressed as ug/mL, so convert them to nM
        in_ug_ml = (dset_df['Standard Units'] == 'ug.mL-1')
        if (sum(in_ug_ml) > 0) and (max_unit == 'nM'):
            mwts = dset_df.loc[in_ug_ml, 'Molecular Weight'].values
            values_in_nM = dset_df.loc[in_ug_ml, 'Standard Value'].values * 1e6 / mwts
            dset_df.loc[in_ug_ml, 'Standard Value'] = values_in_nM
            dset_df.loc[in_ug_ml, 'Standard Units'] = 'nM'
            
    dset_df = dset_df[dset_df['Standard Units'] == max_unit]

    # Standardize the censoring operators to =, < or >, and remove any rows whose operators
    # don't map to a standard one.
    dset_df = standardize_relations(dset_df, db='ChEMBL')

    # Add a column for the pIC50 or log-transformed value, and a column for the associated censoring relation.
    # For pXC50 values, this will be the opposite of the original censoring relation.
    ops = dset_df['Standard Relation'].values
    log_ops = ops.copy()
    if (max_type in ['IC50', 'AC50']) and (max_unit == 'nM'):
        dset_df['pIC50'] = 9.0 - np.log10(dset_df['Standard Value'].values)
        log_ops[ops == '>'] = '<'
        log_ops[ops == '<'] = '>'
    elif (max_type == 'Ki') and (max_unit == 'nM'):
        dset_df['pKi'] = 9.0 - np.log10(dset_df['Standard Value'].values)
        log_ops[ops == '>'] = '<'
        log_ops[ops == '<'] = '>'
    elif (max_type == 'Solubility') and (max_unit == 'nM'):
        dset_df['logSolubility'] = np.log10(dset_df['Standard Value'].values) - 9.0
    elif max_type == 'CL':
        dset_df['logCL'] = np.log10(dset_df['Standard Value'].values)
    dset_df['LogVarRelation'] = log_ops

    # Add a column for the standardized base SMILES string. Remove rows with SMILES strings
    # that RDKit wasn't able to parse.
    if strip_salts:
        dset_df['rdkit_smiles'] = base_smiles_from_smiles(dset_df.Smiles.values.tolist(), workers=16)
    else:
        dset_df['rdkit_smiles'] = [get_rdkit_smiles(s) for s in dset_df.Smiles.values]
    dset_df = dset_df[dset_df.rdkit_smiles != '']

    return dset_df

# ----------------------------------------------------------------------------------------------------------------------
def filter_all_chembl_dsets(force_update=False):
    """
    Generate filtered versions of all the raw datasets present in the data_dirs['ChEMBL']/raw directory.
    Don't replace any existing filtered file unless force_update is True.
    """
    chembl_dir = data_dirs['ChEMBL']
    raw_dir = "%s/raw" % chembl_dir
    filt_dir = "%s/filtered" % chembl_dir
    os.makedirs(filt_dir, exist_ok=True)
    chembl_files = sorted(os.listdir(raw_dir))

    for fn in chembl_files:
        if fn.endswith('.txt'):
            dset_name = fn.replace('.txt', '')
            filt_path = "%s/%s_filt.csv" % (filt_dir, dset_name)
            if not os.path.exists(filt_path) or force_update:
                fpath = '%s/%s' % (raw_dir, fn)
                dset_df = pd.read_table(fpath, sep=';', index_col=False)
                print("Filtering dataset %s" % dset_name)
                dset_df = filter_chembl_dset(dset_df)
                dset_df.to_csv(filt_path, index=False)
                print("Wrote filtered data to %s" % filt_path)


# ----------------------------------------------------------------------------------------------------------------------
def summarize_chembl_dsets():
    """
    Generate a summary table describing the data in the filtered ChEMBL datasets. The function operates on all
    files in the public_dsets/ChEMBL/filtered directory.
    """

    chembl_dir = data_dirs['ChEMBL']
    filt_dir = "%s/filtered" % chembl_dir
    stats_dir = "%s/stats" % chembl_dir
    os.makedirs(stats_dir, exist_ok=True)
    chembl_files = sorted(os.listdir(filt_dir))
    dset_names = []
    databases = []
    mtype_list = []
    log_var_list = []
    units_list = []
    dset_sizes = []
    num_left = []
    num_eq = []
    num_right = []
    cmpd_counts = []
    cmpd_rep_counts = []
    max_cmpd_reps = []
    assay_counts = []
    max_assay_pts = []
    max_assay_list = []
    max_fmt_list = []

    for fn in chembl_files:
        if fn.endswith('.csv') and not fn.endswith('_no_strip.csv'):
            fpath = '%s/%s' % (filt_dir, fn)
            dset_df = pd.read_csv(fpath, index_col=False)
            dset_name = fn.replace('_filt.csv', '')
            dset_names.append(dset_name)
            print("Summarizing %s" % dset_name)
            database = dset_name.split('-')[0].replace('CHEMBL', 'ChEMBL')
            databases.append(database)
            dset_sizes.append(dset_df.shape[0])
            type_df = curate.freq_table(dset_df, 'Standard Type')
            max_type = type_df['Standard Type'].values[0]
            mtype_list.append(max_type)
            log_var = log_var_map[max_type]
            log_var_list.append(log_var)

            unit_freq_df = curate.freq_table(dset_df, 'Standard Units')
            max_unit = unit_freq_df['Standard Units'].values[0]
            units_list.append(max_unit)

            log_ops = dset_df.LogVarRelation.values
            uniq_ops, op_counts = np.unique(log_ops, return_counts=True)
            op_count = dict(zip(uniq_ops, op_counts))
            num_left.append(op_count.get('<', 0))
            num_eq.append(op_count.get('=', 0))
            num_right.append(op_count.get('>', 0))

            smiles_df = curate.freq_table(dset_df, 'rdkit_smiles')
            cmpd_counts.append(smiles_df.shape[0])
            smiles_df = smiles_df[smiles_df.Count > 1]
            cmpd_rep_counts.append(smiles_df.shape[0])
            if smiles_df.shape[0] > 0:
                max_cmpd_reps.append(smiles_df.Count.values[0])
            else:
                max_cmpd_reps.append(1)
            mean_values = []
            stds = []
            cmpd_assays = []
            for smiles in smiles_df.rdkit_smiles.values:
                sset_df = dset_df[dset_df.rdkit_smiles == smiles]
                vals = sset_df[log_var].values
                mean_values.append(np.mean(vals))
                stds.append(np.std(vals))
                cmpd_assays.append(len(set(sset_df['Assay ChEMBL ID'].values)))
            smiles_df['Mean_value'] = mean_values
            smiles_df['Std_dev'] = stds
            smiles_df['Num_assays'] = cmpd_assays
            smiles_file = "%s/%s_replicate_cmpd_stats.csv" % (stats_dir, dset_name)
            smiles_df.to_csv(smiles_file, index=False)

            assay_df = curate.labeled_freq_table(dset_df, ['Assay ChEMBL ID', 'Assay Description', 'BAO Label'])
            assay_counts.append(assay_df.shape[0])
            max_assay_pts.append(assay_df.Count.values[0])
            max_assay_list.append(assay_df['Assay Description'].values[0])
            max_fmt_list.append(assay_df['BAO Label'].values[0])
            assay_df = assay_df[assay_df.Count >= 20]
            assay_file = "%s/%s_top_assay_summary.csv" % (stats_dir, dset_name)
            assay_df.to_csv(assay_file, index=False)

    summary_df = pd.DataFrame(dict(
        Dataset=dset_names,
        Database=databases,
        MeasuredValue=mtype_list,
        LogValue=log_var_list,
        Units=units_list,
        NumPoints=dset_sizes,
        NumUncensored=num_eq,
        NumLeftCensored=num_left,
        NumRightCensored=num_right,
        NumCmpds=cmpd_counts,
        NumReplicatedCmpds=cmpd_rep_counts,
        MaxCmpdReps=max_cmpd_reps,
        NumAssays=assay_counts,
        MaxAssayPoints=max_assay_pts,
        MaxAssay=max_assay_list,
        MaxAssayFormat=max_fmt_list
    ))
    summary_file = "%s/chembl_public_dataset_summary.csv" % stats_dir
    summary_df.to_csv(summary_file, index=False, columns=['Dataset', 'Database', 'NumPoints',
                                                          'NumUncensored', 'NumLeftCensored', 'NumRightCensored',
                                                          'NumCmpds', 'NumReplicatedCmpds', 'MaxCmpdReps',
                                                          'MeasuredValue', 'LogValue', 'Units',
                                                          'NumAssays', 'MaxAssayPoints', 'MaxAssayFormat', 'MaxAssay'])
    print("Wrote summary table to %s" % summary_file)


# ----------------------------------------------------------------------------------------------------------------------
def plot_chembl_log_distrs():
    """
    Plot distributions of the log-transformed values for each of the ChEMBL datasets
    """
    chembl_dir = data_dirs['ChEMBL']
    filt_dir = "%s/filtered" % chembl_dir
    summary_file = "%s/stats/chembl_public_dataset_summary.csv" % chembl_dir
    summary_df = pd.read_csv(summary_file, index_col=False)
    dset_names = set(summary_df.Dataset.values)

    # Plot distributions for the pairs of CYP datasets together
    cyp_dsets = dict(
        CYP2C9 = dict(AC50='CHEMBL25-CYP2C9_human_AC50_26Nov2019', IC50='CHEMBL25-CYP2C9_human_IC50_26Nov2019'),
        CYP2D6 = dict(AC50='CHEMBL25-CYP2D6_human_AC50_26Nov2019', IC50='CHEMBL25-CYP2D6_human_IC50_26Nov2019'),
        CYP3A4 = dict(AC50='CHEMBL25-CYP3A4_human_AC50_26Nov2019', IC50='CHEMBL25-CYP3A4_human_IC50_26Nov2019')
    )

    cyp_dset_names = []
    for cyp in sorted(cyp_dsets.keys()):
        ds_dict = cyp_dsets[cyp]
        cyp_dset_names.append(ds_dict['AC50'])
        cyp_dset_names.append(ds_dict['IC50'])
        ac50_path = "%s/%s_filt.csv" % (filt_dir, ds_dict['AC50'])
        ic50_path = "%s/%s_filt.csv" % (filt_dir, ds_dict['IC50'])
        ac50_df = pd.read_csv(ac50_path, index_col=False)
        ic50_df = pd.read_csv(ic50_path, index_col=False)
        ac50_smiles = set(ac50_df.Smiles.values)
        ic50_smiles = set(ic50_df.Smiles.values)
        cmn_smiles = ac50_smiles & ic50_smiles
        print("For %s: %d SMILES strings in both datasets" % (cyp, len(cmn_smiles)))

        fig, ax = plt.subplots(figsize=(10,8))
        ax = sns.distplot(ac50_df.pIC50.values, hist=False, kde_kws=dict(shade=True, bw=0.05), color='b', ax=ax, label='PubChem')
        ic50_lc_df = ic50_df[ic50_df.LogVarRelation == '<']
        ic50_rc_df = ic50_df[ic50_df.LogVarRelation == '>']
        ic50_uc_df = ic50_df[ic50_df.LogVarRelation == '=']
        ax = sns.distplot(ic50_uc_df.pIC50.values, hist=False, kde_kws=dict(shade=True, bw=0.05), color='g', ax=ax, label='Uncens')
        ax = sns.distplot(ic50_lc_df.pIC50.values, hist=False, kde_kws=dict(shade=False, bw=0.05), color='r', ax=ax, label='LeftCens')
        ax = sns.distplot(ic50_rc_df.pIC50.values, hist=False, kde_kws=dict(shade=False, bw=0.05), color='m', ax=ax, label='RightCens')
        ax.set_xlabel('pIC50')
        ax.set_title('Distributions of %s dataset values' % cyp)
        plt.show()

    other_dset_names = sorted(dset_names - set(cyp_dset_names))
    for dset_name in other_dset_names:
        log_var = summary_df.LogValue.values[summary_df.Dataset == dset_name][0]
        filt_path = "%s/%s_filt.csv" % (filt_dir, dset_name)
        dset_df = pd.read_csv(filt_path, index_col=False)
        uc_df = dset_df[dset_df.LogVarRelation == '=']
        lc_df = dset_df[dset_df.LogVarRelation == '<']
        rc_df = dset_df[dset_df.LogVarRelation == '>']
        log_uc_values = uc_df[log_var].values
        log_lc_values = lc_df[log_var].values
        log_rc_values = rc_df[log_var].values
        fig, ax = plt.subplots(figsize=(10,8))
        ax = sns.distplot(log_uc_values, hist=False, kde_kws=dict(shade=True, bw=0.05), color='b', ax=ax, label='Uncens')
        ax = sns.distplot(log_lc_values, hist=False, kde_kws=dict(shade=False, bw=0.05), color='r', ax=ax, label='LeftCens')
        ax = sns.distplot(log_rc_values, hist=False, kde_kws=dict(shade=False, bw=0.05), color='m', ax=ax, label='RightCens')
        ax.set_xlabel(log_var)
        ax.set_title('Distribution of log transformed values for %s' % dset_name)
        plt.show()


# ----------------------------------------------------------------------------------------------------------------------
def curate_chembl_activity_assay(dset_df, target, endpoint, database):
    """
    Examine data from individual ChEMBL assays in the given dataset to look for suspicious patterns of activity
    values and censoring relations. Add relations where they appear to be needed, and filter out data from
    assays that seem to have only one-shot categorical data.
    """

    chembl_root = data_dirs['ChEMBL']
    assay_df = curate.freq_table(dset_df, 'Assay ChEMBL ID')
    assays = assay_df['Assay ChEMBL ID'].values
    counts = assay_df['Count'].values
    num_eq = []
    num_lt = []
    num_gt = []
    # In this function, variables referring to xc50 values may be Ki's (or any other activity value expressed
    # as a concentration).
    max_xc50s = []
    num_max_xc50 = []
    min_xc50s = []
    num_min_xc50 = []

    # For each assay ID, tabulate the number of occurrences of each relation, the min and max activity
    # and the number of values reported as the max or min activity
    for assay in assays:
        assay_dset_df = dset_df[dset_df['Assay ChEMBL ID'] == assay]

        xc50s = assay_dset_df['Standard Value'].values
        max_xc50 = max(xc50s)
        max_xc50s.append(max_xc50)
        min_xc50 = min(xc50s)
        min_xc50s.append(min_xc50)
        relations = assay_dset_df['Standard Relation'].values
        num_eq.append(sum(relations == '='))
        num_lt.append(sum(relations == '<'))
        num_gt.append(sum(relations == '>'))
        num_max_xc50.append(sum(xc50s == max_xc50))
        num_min_xc50.append(sum(xc50s == min_xc50))
    assay_df['num_eq'] = num_eq
    assay_df['num_lt'] = num_lt
    assay_df['num_gt'] = num_gt
    assay_df['max_xc50'] = max_xc50s
    assay_df['num_max_xc50'] = num_max_xc50
    assay_df['min_xc50'] = min_xc50s
    assay_df['num_min_xc50'] = num_min_xc50

    # Flag assays that appear to report one-shot screening results only (because all values are left or
    # right censored at the same threshold)
    num_eq = np.array(num_eq)
    num_lt = np.array(num_lt)
    num_gt = np.array(num_gt)
    max_xc50s = np.array(max_xc50s)
    min_xc50s = np.array(min_xc50s)
    num_max_xc50 = np.array(num_max_xc50)
    num_min_xc50 = np.array(num_min_xc50)

    one_shot =  (num_eq == 0) & (num_lt > 0) & (num_gt > 0)
    assay_df['one_shot'] = one_shot
    # Flag assays that appear not to report left-censoring correctly (because no values are censored
    # and there are multiple values at highest XC50 or Ki)
    no_left_censoring = (counts == num_eq) & (num_max_xc50 >= 5)
    assay_df['no_left_censoring'] = no_left_censoring
    # Flag assays that appear not to report right-censoring correctly (because no values are censored
    # and there are multiple values at lowest XC50)
    no_right_censoring = (counts == num_eq) & (num_min_xc50 >= 5)
    assay_df['no_right_censoring'] = no_right_censoring

    assay_file = "%s/stats/%s_%s_%s_assay_stats.csv" % (chembl_root, database, target, endpoint)
    assay_df.to_csv(assay_file, index=False)
    print("Wrote %s %s assay censoring statistics to %s" % (target, endpoint, assay_file))

    # Now generate a "curated" version of the dataset
    assay_dsets = []
    for assay, is_one_shot, has_no_left_cens, has_no_right_cens in zip(assays, one_shot, no_left_censoring,
                                                                       no_right_censoring):
        # Skip over assays that appear to contain one-shot data
        if is_one_shot:
            print("Skipping apparent one-shot data from assay %s" % assay)
        else:
            assay_dset_df = dset_df[dset_df['Assay ChEMBL ID'] == assay].copy()
            xc50s = assay_dset_df['Standard Value'].values
            max_xc50 = max(xc50s)
            min_xc50 = min(xc50s)
            # Add censoring relations for rows that seem to need them
            relations = assay_dset_df['Standard Relation'].values
            log_relations = assay_dset_df['LogVarRelation'].values
            if has_no_left_cens:
                relations[xc50s == max_xc50] = '>'
                log_relations[xc50s == max_xc50] = '<'
                print("Adding missing left-censoring relations for assay %s" % assay)
            if has_no_right_cens:
                relations[xc50s == min_xc50] = '<'
                log_relations[xc50s == min_xc50] = '>'
                print("Adding missing right-censoring relations for assay %s" % assay)
            assay_dset_df['Standard Relation'] = relations
            assay_dset_df['LogVarRelation'] = log_relations
            assay_dsets.append(assay_dset_df)
    curated_df = pd.concat(assay_dsets, ignore_index=True)
    return curated_df


# ----------------------------------------------------------------------------------------------------------------------
def curate_chembl_activity_assays(species='human', force_update=False):
    """
    Examine data from individual ChEMBL assays in each dataset to look for suspicious patterns of activity
    values and censoring relations. Add relations where they appear to be needed, and filter out data from
    assays that seem to have only one-shot categorical data.
    """
    chembl_root = data_dirs['ChEMBL']

    filtered_dir = '%s/filtered' % chembl_root
    curated_dir = '%s/curated' % chembl_root
    os.makedirs(curated_dir, exist_ok=True)

    targets = sorted(chembl_dsets.keys())
    for target in targets:
        if type(chembl_dsets[target]) == dict:
            for endpoint, dset_name in chembl_dsets[target].items():
                database = dset_name.split('-')[0].replace('CHEMBL', 'ChEMBL')
                curated_file = "%s/%s_%s_%s_%s_curated.csv" % (curated_dir, database, target, endpoint, species)
                if os.path.exists(curated_file) and not force_update:
                    print("\nCurated dataset %s already exists, skipping" % curated_file)
                    continue
                print("\n\nCurating %s data for %s" % (endpoint, target))
                dset_file = "%s/%s_filt.csv" % (filtered_dir, dset_name)
                dset_df = pd.read_csv(dset_file, index_col=False)
                curated_df = curate_chembl_activity_assay(dset_df, target, endpoint, database)
                curated_df.to_csv(curated_file, index=False)
                print("Wrote %s" % curated_file)

# ----------------------------------------------------------------------------------------------------------------------
def upload_chembl_raw_data(dset_name, endpoint, title, description, tags,
                           assay_category, functional_area, target_type, 
                           target='', database='ChEMBL25', activity='inhibition', species='human',
                           force_update=False):
    """
    Upload a raw dataset to the datastore from the given data frame. 
    Returns the datastore OID of the uploaded dataset.
    """
    raw_dir = '%s/raw' % data_dirs['ChEMBL']
    raw_path = "%s/%s.txt" % (raw_dir, dset_name)
    dset_df = pd.read_table(raw_path, sep=';', index_col=False)

    bucket = 'public'
    filename = '%s.csv' % dset_name
    dataset_key = 'dskey_' + filename

    kv = {
          'activity': activity,
          'assay_category': assay_category,
          'assay_endpoint': endpoint,
          'target_type': target_type,
          'functional_area': functional_area,
          'data_origin': database,
          'species': species,
          'file_category': 'experimental',
          'curation_level': 'raw',
          'matrix': 'in vitro',
          'sample_type': 'in_vitro',
          'id_col': 'Molecule ChEMBL ID',
          'smiles_col': 'Smiles',
          'response_col': 'Standard Value'} 

    if target != '':
        kv['target'] = target

    ds_client = dsf.config_client()
    if force_update or not dsf.dataset_key_exists(dataset_key, bucket, ds_client):
        raw_meta = dsf.upload_df_to_DS(dset_df, bucket, filename=filename, title=title,
                                       description=description,
                                       tags=tags, key_values=kv, client=None, dataset_key=dataset_key,
                                       override_check=True, return_metadata=True)
        print("Uploaded raw dataset with key %s" % dataset_key)
    else:
        raw_meta = dsf.retrieve_dataset_by_datasetkey(dataset_key, bucket, ds_client, return_metadata=True)
        print("Raw dataset %s is already in datastore, skipping upload." % dataset_key)
    raw_dset_oid = raw_meta['dataset_oid']
    return raw_dset_oid

# ----------------------------------------------------------------------------------------------------------------------
def upload_chembl_curated_data(dset_name, endpoint, title, description, tags,
                           assay_category, functional_area, target_type, 
                           target='', database='ChEMBL25', activity='inhibition', species='human',
                           raw_dset_oid=None, strip_salts=True, force_update=False):
    """
    Upload a curated dataset to the datastore. Returns the datastore OID of the uploaded dataset.
    """
    curated_dir = '%s/curated' % data_dirs['ChEMBL']
    filtered_dir = '%s/filtered' % data_dirs['ChEMBL']

    if strip_salts:
        strip_suffix = ''
    else:
        strip_suffix = '_no_strip'
    if target == '':
        # This is a PK dataset, for which curation consists only of the initial filtering
        filename = '%s_curated%s.csv' % (dset_name, strip_suffix)
        curated_file = "%s/%s_filt%s.csv" % (filtered_dir, dset_name, strip_suffix)
    else:
        # This is a bioactivity dataset
        filename = "%s_%s_%s_%s_curated%s.csv" % (database, target, endpoint, species, strip_suffix)
        curated_file = "%s/%s" % (curated_dir, filename)

    dset_df = pd.read_csv(curated_file, index_col=False)
    bucket = 'public'
    dataset_key = 'dskey_' + filename

    kv = {
          'activity': activity,
          'assay_category': assay_category,
          'assay_endpoint': endpoint,
          'target_type': target_type,
          'functional_area': functional_area,
          'data_origin': database,
          'species': species,
          'file_category': 'experimental',
          'curation_level': 'curated',
          'matrix': 'in vitro',
          'ignore_salts': str(strip_salts),
          'sample_type': 'in_vitro',
          'id_col': 'Molecule ChEMBL ID',
          'smiles_col': 'rdkit_smiles',
          'response_col': log_var_map[endpoint] } 
    if target != '':
        kv['target'] = target
    if raw_dset_oid is not None:
        kv['source_file_id'] = raw_dset_oid

    ds_client = dsf.config_client()
    if force_update or not dsf.dataset_key_exists(dataset_key, bucket, ds_client):
        curated_meta = dsf.upload_df_to_DS(dset_df, bucket, filename=filename, title=title,
                                       description=description,
                                       tags=tags, key_values=kv, client=None, dataset_key=dataset_key,
                                       override_check=True, return_metadata=True)
        print("Uploaded curated dataset with key %s" % dataset_key)
    else:
        print("Curated dataset %s is already in datastore, skipping upload." % dataset_key)
        curated_meta = dsf.retrieve_dataset_by_datasetkey(dataset_key, bucket, ds_client, return_metadata=True)
    curated_oid = curated_meta['dataset_oid']
    return curated_oid


# ----------------------------------------------------------------------------------------------------------------------
def create_ml_ready_chembl_dataset(dset_name, endpoint, target='', species='human', active_thresh=None,
                                   strip_salts=True, force_update=False):
    """
    Average replicate values from the curated version of the given dataset to give one value
    per unique compound. Select and rename columns to include only the ones we need for building
    ML models. Save the resulting dataset to disk.

    endpoint is the measured value, IC50, AC50, Ki, CL or Solubility.
    """
    curated_dir = '%s/curated' % data_dirs['ChEMBL']
    filtered_dir = '%s/filtered' % data_dirs['ChEMBL']
    ml_ready_dir = '%s/ml_ready' % data_dirs['ChEMBL']
    os.makedirs(ml_ready_dir, exist_ok=True)
    database = dset_name.split('-')[0].replace('CHEMBL', 'ChEMBL')

    if strip_salts:
        strip_suffix = ''
    else:
        strip_suffix = '_no_strip'
    if target == '':
        # This is a PK dataset, for which curation consists only of the initial filtering
        curated_file = "%s/%s_filt%s.csv" % (filtered_dir, dset_name, strip_suffix)
        ml_ready_file = "%s/%s_ml_ready%s.csv" % (ml_ready_dir, dset_name, strip_suffix)
    else:
        # This is a bioactivity dataset
        curated_file = "%s/%s_%s_%s_%s_curated%s.csv" % (curated_dir, database, target, endpoint, species, strip_suffix)
        ml_ready_file = "%s/%s_%s_%s_%s_ml_ready%s.csv" % (ml_ready_dir, database, target, endpoint, species, strip_suffix)

    if os.path.exists(ml_ready_file) and not force_update:
        return

    dset_df = pd.read_csv(curated_file, index_col=False)

    # Rename and select the columns we want from the curated dataset
    param = log_var_map[endpoint]
    agg_cols = ['compound_id', 'rdkit_smiles', 'relation', param]
    colmap = {
        'Molecule ChEMBL ID': 'compound_id',
        'LogVarRelation': 'relation'
    }
    assay_df = dset_df.rename(columns=colmap)[agg_cols]
    # Compute a single value and relational flag for each compound
    ml_ready_df = curate.aggregate_assay_data(assay_df, value_col=param, active_thresh=active_thresh,
                                              id_col='compound_id', smiles_col='rdkit_smiles',
                                              relation_col='relation')

    ml_ready_df.to_csv(ml_ready_file, index=False)
    print("Wrote ML-ready data to %s" % ml_ready_file)
    return ml_ready_df

# ----------------------------------------------------------------------------------------------------------------------
def create_ml_ready_chembl_activity_datasets(species='human', strip_salts=True, force_update=False):
    """
    Create ML-ready datasets for all the ChEMBL activity assays that have curated datasets.
    """
    chembl_root = data_dirs['ChEMBL']

    filtered_dir = '%s/filtered' % chembl_root
    curated_dir = '%s/curated' % chembl_root
    ml_ready_dir = '%s/ml_ready' % chembl_root
    os.makedirs(ml_ready_dir, exist_ok=True)

    targets = sorted(chembl_dsets.keys())
    for target in targets:
        if type(chembl_dsets[target]) == dict:
            for endpoint, dset_name in chembl_dsets[target].items():
                database = dset_name.split('-')[0].replace('CHEMBL', 'ChEMBL')
                curated_file = "%s/%s_%s_%s_%s_curated.csv" % (curated_dir, database, target, endpoint, species)
                ml_ready_file = "%s/%s_%s_%s_%s_ml_ready.csv" % (ml_ready_dir, database, target, endpoint, species)
                if os.path.exists(ml_ready_file) and not force_update:
                    print("\nML-ready dataset %s already exists, skipping" % ml_ready_file)
                    continue
                print("\n\nCreating ML-ready %s data for %s" % (endpoint, target))
                #curated_df = pd.read_csv(curated_file, index_col=False)
                ml_ready_df = create_ml_ready_chembl_dataset(dset_name, endpoint, target, species=species,
                                                             strip_salts=strip_salts, force_update=force_update)

# ----------------------------------------------------------------------------------------------------------------------
def upload_chembl_ml_ready_data(dset_name, endpoint, title, description, tags,
                           assay_category, functional_area, target_type, 
                           target='', database='ChEMBL25', activity='inhibition', species='human',
                           curated_dset_oid=None, strip_salts=True, force_update=False):
    """
    Upload a ML-ready dataset to the datastore, previously created by create_ml_ready_chembl_dataset. 
    Returns the datastore OID of the uploaded dataset.
    """
    ml_ready_dir = '%s/ml_ready' % data_dirs['ChEMBL']

    if strip_salts:
        strip_suffix = ''
    else:
        strip_suffix = '_no_strip'
    if target == '':
        # This is a PK dataset
        filename = '%s_ml_ready%s.csv' % (dset_name, strip_suffix)
    else:
        # This is a bioactivity dataset
        filename = "%s_%s_%s_%s_ml_ready%s.csv" % (database, target, endpoint, species, strip_suffix)
    ml_ready_file = "%s/%s" % (ml_ready_dir, filename)

    dset_df = pd.read_csv(ml_ready_file, index_col=False)
    bucket = 'public'
    dataset_key = 'dskey_' + filename

    kv = {
          'activity': activity,
          'assay_category': assay_category,
          'assay_endpoint': endpoint,
          'target_type': target_type,
          'functional_area': functional_area,
          'data_origin': database,
          'species': species,
          'file_category': 'experimental',
          'curation_level': 'ml_ready',
          'matrix': 'in vitro',
          'ignore_salts': str(strip_salts),
          'sample_type': 'in_vitro',
          'id_col': 'compound_id',
          'smiles_col': 'base_rdkit_smiles',
          'response_col': log_var_map[endpoint]
          } 
    if target != '':
        kv['target'] = target
    if curated_dset_oid is not None:
        kv['source_file_id'] = curated_dset_oid

    ds_client = dsf.config_client()
    if force_update or not dsf.dataset_key_exists(dataset_key, bucket, ds_client):
        ml_ready_meta = dsf.upload_df_to_DS(dset_df, bucket, filename=filename, title=title,
                                       description=description,
                                       tags=tags, key_values=kv, client=None, dataset_key=dataset_key,
                                       override_check=True, return_metadata=True)
        print("Uploaded ML-ready dataset with key %s" % dataset_key)
    else:
        print("ML-ready dataset %s is already in datastore, skipping upload." % dataset_key)
        ml_ready_meta = dsf.retrieve_dataset_by_datasetkey(dataset_key, bucket, ds_client, return_metadata=True)
    ml_ready_oid = ml_ready_meta['dataset_oid']
    return ml_ready_oid

# ----------------------------------------------------------------------------------------------------------------------
def chembl_dataset_curation_pipeline(dset_table_file, force_update=False):
    """
    Run a series of ChEMBL datasets through the process of filtering, curation, and aggregation for use in
    building machine learning models. Upload the raw, curated and ML-ready datasets to the datastore.
    The datasets are described in the CSV file dset_table_file, which tabulates the attributes of each dataset
    and the metadata to be included with the uploaded datasets.
    """
    chembl_dir = data_dirs['ChEMBL']
    raw_dir = "%s/raw" % chembl_dir
    filt_dir = "%s/filtered" % chembl_dir
    curated_dir = "%s/curated" % chembl_dir
    ml_ready_dir = "%s/ml_ready" % chembl_dir

    os.makedirs(filt_dir, exist_ok=True)
    table_df = pd.read_csv(dset_table_file, index_col=False)
    table_df = table_df.fillna('')

    for i, dset_name in enumerate(table_df.Dataset.values):
        endpoint = table_df.endpoint.values[i]
        target = table_df.target.values[i]
        assay_category = table_df.assay_category.values[i]
        functional_area = table_df.functional_area.values[i]
        target_type = table_df.target_type.values[i]
        activity = table_df.activity.values[i]
        species = table_df.species.values[i]
        database = table_df.database.values[i]
        title = table_df.title.values[i]
        description = table_df.description.values[i]
        tags = ['public', 'raw']

        # Upload the raw dataset as-is
        raw_dset_oid = upload_chembl_raw_data(dset_name, endpoint, title, description, tags,
                           assay_category, functional_area, target_type, 
                           target=target, database=database, activity=activity, species=species,
                           force_update=force_update)

        # First curation step: Filter dataset to remove rows with missing IDs, SMILES, values, etc.
        raw_path = "%s/%s.txt" % (raw_dir, dset_name)
        filt_path = "%s/%s_filt.csv" % (filt_dir, dset_name)
        if not os.path.exists(filt_path) or force_update:
            dset_df = pd.read_table(raw_path, sep=';', index_col=False)
            print("Filtering dataset %s" % dset_name)
            filt_df = filter_chembl_dset(dset_df)
            filt_df.to_csv(filt_path, index=False)
        else:
            filt_df = pd.read_csv(filt_path, index_col=False)
            print("Filtered dataset file %s already exists" % filt_path)

        # Second curation step: Fix or remove anomalous data. Currently this is only done for 
        # bioactivity data.
        if target != '':
            curated_file = "%s/%s_%s_%s_%s_curated.csv" % (curated_dir, database, target, endpoint, species)
            if not os.path.exists(curated_file) or force_update:
                print("Curating %s data for %s" % (endpoint, target))
                curated_df = curate_chembl_activity_assay(filt_df, target, endpoint, database)
                curated_df.to_csv(curated_file, index=False)
            else:
                curated_df = pd.read_csv(curated_file, index_col=False)
                print("Curated %s dataset file for %s already exists" % (endpoint, target))
            description += "\nCurated using public_data_curation functions filter_chembl_dset and curate_chembl_activity_assay."
        else:
            curated_df = filt_df
            description += "\nCurated using public_data_curation function filter_chembl_dset."
        title = title.replace('Raw', 'Curated')
        tags = ['public', 'curated']

        # Upload curated data to datastore
        curated_dset_oid = upload_chembl_curated_data(dset_name, endpoint, title, description, tags,
                           assay_category, functional_area, target_type, 
                           target=target, database=database, activity=activity, species=species,
                           raw_dset_oid=raw_dset_oid, force_update=force_update)

        # Prepare ML-ready dataset
        if target == '':
            ml_ready_file = "%s/%s_ml_ready.csv" % (ml_ready_dir, dset_name)
        else:
            ml_ready_file = "%s/%s_%s_%s_%s_ml_ready.csv" % (ml_ready_dir, database, target, endpoint, species)
        if not os.path.exists(ml_ready_file) or force_update:
            print("Creating ML-ready dataset file %s" % ml_ready_file)
            ml_ready_df = create_ml_ready_chembl_dataset(dset_name, endpoint, target=target, species=species, active_thresh=None,
                                                         force_update=force_update)
        else:
            ml_ready_df = pd.read_csv(ml_ready_file, index_col=False)
            print("ML-ready dataset file %s already exists" % ml_ready_file)
        title = title.replace('Curated', 'ML-ready')
        description += "\nAveraged for ML model building using public_data_curation.create_ml_ready_dataset."
        tags = ['public', 'ML-ready']

        # Upload ML-ready data to the datastore
        ml_ready_dset_oid = upload_chembl_ml_ready_data(dset_name, endpoint, title, description, tags,
                           assay_category, functional_area, target_type, 
                           target=target, database=database, activity=activity, species=species,
                           curated_dset_oid=curated_dset_oid, force_update=force_update)
        print("Done with dataset %s\n" % dset_name)       

# ----------------------------------------------------------------------------------------------------------------------
def chembl_replicate_variation(dset_df, value_col='pIC50', dset_label='', min_freq=3, num_assays=4):
    """
    Plot the variation among measurements in a ChEMBL dataset for compounds with multiple measurements
    from the same or different ChEMBL assay IDs.
    """
    rep_df = curate.freq_table(dset_df, 'rdkit_smiles', min_freq=min_freq)
    rep_df['mean_value'] = [np.mean(dset_df[dset_df.rdkit_smiles == s][value_col].values)
                            for s in rep_df.rdkit_smiles.values]
    rep_df['std_value'] = [np.std(dset_df[dset_df.rdkit_smiles == s][value_col].values)
                           for s in rep_df.rdkit_smiles.values]
    rep_df = rep_df.sort_values(by='mean_value')
    nrep = rep_df.shape[0]
    rep_df['cmpd_id'] = ['C%05d' % i for i in range(nrep)]

    rep_dset_df = dset_df[dset_df.rdkit_smiles.isin(rep_df.rdkit_smiles.values)].merge(
        rep_df, how='left', on='rdkit_smiles')

    # Label the records coming from the num_assays most common assays with the first part of
    # their assay descriptions; label the others as 'Other'.
    assay_df = curate.freq_table(rep_dset_df, 'Assay ChEMBL ID')
    other_ids = assay_df['Assay ChEMBL ID'].values[num_assays:]
    assay_labels = np.array([desc[:30]+'...' for desc in rep_dset_df['Assay Description'].values])
    assay_labels[rep_dset_df['Assay ChEMBL ID'].isin(other_ids)] = 'Other'
    rep_dset_df['Assay'] = assay_labels

    fig, ax = plt.subplots(figsize=(10,15))
    sns.stripplot(x=value_col, y='cmpd_id', hue='Assay', data=rep_dset_df,
                  order=rep_df.cmpd_id.values)
    ax.set_title(dset_label)
    return rep_df


# ----------------------------------------------------------------------------------------------------------------------
# Filename templates for curated bioactivity datasets, with a %s field to plug in the target or property name. Probably
# we should just rename the files from all data sources to follow the standard template: 
# (database)_(target)_(endpoint)_(species)_curated.csv.

curated_dset_file_templ = dict(
    ChEMBL="ChEMBL25_%s_IC50_human_curated.csv",
    DTC="%s_DTC_curated.csv",
    Excape="%s_Excape_curated.csv"
)

# ----------------------------------------------------------------------------------------------------------------------
def curate_with_salts(dset_table_file, force_update=False):
    """
    Recurate one or more ChEMBL datasets without stripping salts from the SMILES strings. Generates new curated
    and ML-ready datasets; the raw dataset is assumed to have already been uploaded to the datastore.
    """
    chembl_dir = data_dirs['ChEMBL']
    raw_dir = "%s/raw" % chembl_dir
    filt_dir = "%s/filtered" % chembl_dir
    curated_dir = "%s/curated" % chembl_dir
    ml_ready_dir = "%s/ml_ready" % chembl_dir

    os.makedirs(filt_dir, exist_ok=True)
    table_df = pd.read_csv(dset_table_file, index_col=False)
    table_df = table_df.fillna('')

    ds_client = dsf.config_client()

    for i, dset_name in enumerate(table_df.Dataset.values):
        endpoint = table_df.endpoint.values[i]
        target = table_df.target.values[i]
        assay_category = table_df.assay_category.values[i]
        functional_area = table_df.functional_area.values[i]
        target_type = table_df.target_type.values[i]
        activity = table_df.activity.values[i]
        species = table_df.species.values[i]
        database = table_df.database.values[i]
        title = table_df.title.values[i]
        description = table_df.description.values[i]
        tags = ['public', 'raw']

        bucket = 'public'
        filename = '%s.csv' % dset_name
        dataset_key = 'dskey_' + filename

        raw_meta = dsf.retrieve_dataset_by_datasetkey(dataset_key, bucket, ds_client, return_metadata=True)
        raw_dset_oid = raw_meta['dataset_oid']

        # First curation step: Filter dataset to remove rows with missing IDs, SMILES, values, etc.
        raw_path = "%s/%s.txt" % (raw_dir, dset_name)
        filt_path = "%s/%s_filt_no_strip.csv" % (filt_dir, dset_name)
        if not os.path.exists(filt_path) or force_update:
            dset_df = pd.read_table(raw_path, sep=';', index_col=False)
            print("Filtering dataset %s without stripping salts." % dset_name)
            filt_df = filter_chembl_dset(dset_df, strip_salts=False)
            filt_df.to_csv(filt_path, index=False)
        else:
            filt_df = pd.read_csv(filt_path, index_col=False)
            print("Filtered dataset file %s already exists" % filt_path)


        # Second curation step: Fix or remove anomalous data. Currently this is only done for 
        # bioactivity data.
        if target != '':
            curated_file = "%s/%s_%s_%s_%s_curated_no_strip.csv" % (curated_dir, database, target, endpoint, species)
            if not os.path.exists(curated_file) or force_update:
                print("Curating %s data for %s" % (endpoint, target))
                curated_df = curate_chembl_activity_assay(filt_df, target, endpoint, database)
                curated_df.to_csv(curated_file, index=False)
            else:
                curated_df = pd.read_csv(curated_file, index_col=False)
                print("Curated %s dataset file for %s already exists" % (endpoint, target))
            description += ("\nCurated using public_data_curation functions filter_chembl_dset and curate_chembl_activity_assay," + 
                            "\nwithout stripping salts.")
        else:
            curated_df = filt_df
            description += "\nCurated using public_data_curation function filter_chembl_dset without stripping salts."
        title = title.replace('Raw', 'Curated')
        tags = ['public', 'curated']

        # Upload curated data to datastore
        curated_dset_oid = upload_chembl_curated_data(dset_name, endpoint, title, description, tags,
                           assay_category, functional_area, target_type, 
                           target=target, database=database, activity=activity, species=species,
                           raw_dset_oid=raw_dset_oid, strip_salts=False, force_update=force_update)

        # Prepare ML-ready dataset
        if target == '':
            ml_ready_file = "%s/%s_ml_ready_no_strip.csv" % (ml_ready_dir, dset_name)
        else:
            ml_ready_file = "%s/%s_%s_%s_%s_ml_ready_no_strip.csv" % (ml_ready_dir, database, target, endpoint, species)
        if not os.path.exists(ml_ready_file) or force_update:
            print("Creating ML-ready dataset file %s" % ml_ready_file)
            ml_ready_df = create_ml_ready_chembl_dataset(dset_name, endpoint, target=target, species=species, active_thresh=None,
                                                         strip_salts=False, force_update=force_update)
        else:
            ml_ready_df = pd.read_csv(ml_ready_file, index_col=False)
            print("ML-ready dataset file %s already exists" % ml_ready_file)
        title = title.replace('Curated', 'ML-ready')
        description += "\nAveraged for ML model building using public_data_curation.create_ml_ready_dataset."
        tags = ['public', 'ML-ready']

        # Upload ML-ready data to the datastore
        ml_ready_dset_oid = upload_chembl_ml_ready_data(dset_name, endpoint, title, description, tags,
                           assay_category, functional_area, target_type, 
                           target=target, database=database, activity=activity, species=species,
                           curated_dset_oid=curated_dset_oid, strip_salts=False, force_update=force_update)
        print("Done with dataset %s\n" % dset_name)       

# ----------------------------------------------------------------------------------------------------------------------
def chembl_jak_replicate_variation(min_freq=2, num_assays=4):
    """
    Plot variation among replicate measurements for compounds in the JAK datasets
    """
    jak_genes = ['JAK1', 'JAK2', 'JAK3']
    db = 'ChEMBL'
    dsets = {}
    for gene in jak_genes:
        dset_file = "%s/curated/%s" % (data_dirs[db], curated_dset_file_templ[db] % gene)
        dset_df = pd.read_csv(dset_file, index_col=False)
        dsets[gene] = chembl_replicate_variation(dset_df, value_col='pIC50', min_freq=min_freq,
                                                 dset_label=gene,
                                                 num_assays=num_assays)
    return dsets


# ----------------------------------------------------------------------------------------------------------------------
def chembl_clearance_data():
    """
    """
    cl_path = "/usr/local/data/public_dsets/ChEMBL/raw/CHEMBL25-all_microsomal_clearance.csv"
    cl_df = pd.read_csv(cl_path, index=False)
    # Filter non-human data
    cl_df = cl_df[cl_df['Assay Organism'] == 'Homo sapiens']
    # Filter out data from tissues other than liver. Blank tissue type is assumed to be liver.
    cl_df['Assay Tissue Name'] = cl_df['Assay Tissue Name'].fillna('')
    cl_df = cl_df[cl_df['Assay Tissue Name'].isin(['Liver', ''])]

    assay_df = curate.labeled_freq_table(cl_df, ['Assay ChEMBL ID', 'Assay Description'], min_freq=2)
    return assay_df

# ----------------------------------------------------------------------------------------------------------------------
def chembl_assay_bias(target, endpoint, database='ChEMBL25', species='human', min_cmp_assays=5, min_cmp_cmpds=10):
    """
    Investigate systematic biases among assays for target, by selecting data for compounds with data from
    multiple assays and computing deviations from mean for each compound; then reporting mean deviation for
    each assay.
    """
    curated_dir = '%s/curated' % data_dirs['ChEMBL']
    curated_file = "%s/%s_%s_%s_%s_curated.csv" % (curated_dir, database, target, endpoint, species)
    dset_df = pd.read_csv(curated_file, index_col=False)
    assay_df = curate.labeled_freq_table(dset_df, ['Assay ChEMBL ID', 'Assay Description', 'BAO Label'], min_freq=2)
    assays = assay_df['Assay ChEMBL ID'].values
    print("\nChecking bias for ChEMBL %s %s dataset:" % (target, endpoint))
    if assay_df.shape[0] == 1:
        print("Dataset %s has data for one assay only; skipping." % curated_file)
        return None
    # Restrict to data from assays with at least 2 rows of data
    dset_df = dset_df[dset_df['Assay ChEMBL ID'].isin(assay_df['Assay ChEMBL ID'].values.tolist())]

    # Tabulate overall mean and SD and compound count for each assay
    log_var = log_var_map[dset_df['Standard Type'].values[0]]
    mean_values = [np.mean(dset_df[dset_df['Assay ChEMBL ID'] == assay][log_var].values) for assay in assays]
    stds = [np.std(dset_df[dset_df['Assay ChEMBL ID'] == assay][log_var].values) for assay in assays]
    ncmpds = [len(set(dset_df[dset_df['Assay ChEMBL ID'] == assay]['rdkit_smiles'].values)) for assay in assays]
    assay_df['num_cmpds'] = ncmpds
    assay_df['mean_%s' % log_var] = mean_values
    assay_df['std_%s' % log_var] = stds
    assay_df = assay_df.rename(columns={'Count': 'num_rows'})

    # Select compounds with data from multiple assays. Compute mean values for each compound.
    # Then compute deviations from mean for each assay for each compound.
    assay_devs = {assay : [] for assay in assays}
    cmp_assays = {assay : set() for assay in assays}
    cmp_cmpds = {assay : 0 for assay in assays}
    rep_df = curate.freq_table(dset_df, 'rdkit_smiles', min_freq=2)
    for smiles in rep_df.rdkit_smiles.values:
        sset_df = dset_df[dset_df.rdkit_smiles == smiles]
        sset_assays = sset_df['Assay ChEMBL ID'].values
        sset_assay_set = set(sset_assays)
        num_assays = len(set(sset_assays))
        if num_assays > 1:
            vals = sset_df[log_var].values
            mean_val = np.mean(vals)
            deviations = vals - mean_val
            for assay, dev in zip(sset_assays, deviations):
                assay_devs[assay].append(dev)
                cmp_assays[assay] |= (sset_assay_set - set([assay]))
                cmp_cmpds[assay] += 1
    assay_df['num_cmp_assays'] = [len(cmp_assays[assay]) for assay in assays]
    assay_df['num_cmp_cmpds'] = [cmp_cmpds[assay] for assay in assays]
    mean_deviations = [np.mean(assay_devs[assay]) for assay in assays]
    assay_df['mean_deviation'] = mean_deviations
    assay_df = assay_df.sort_values(by='mean_deviation', ascending=False)
    assay_file = "%s/stats/%s_%s_assay_bias.csv" % (data_dirs['ChEMBL'], target, endpoint)
    assay_df.to_csv(assay_file, index=False)
    # Flag assays compared against at least min_cmp_assays other assays over min_cmp_cmpds compounds
    flag_df = assay_df[(assay_df.num_cmp_assays >= min_cmp_assays) & (assay_df.num_cmp_cmpds >= min_cmp_cmpds)]
    print("For %s %s data: %d assays with robust bias data:" % (target, endpoint, flag_df.shape[0]))
    if flag_df.shape[0] > 0:
        print(flag_df)
    print("Wrote assay bias statistics to %s" % assay_file)
    return assay_df

# ----------------------------------------------------------------------------------------------------------------------
def chembl_activity_assay_bias():
    """
    Tabulate systematic biases for all the ChEMBL XC50 datasets
    """
    targets = sorted(chembl_dsets.keys())
    for target in targets:
        if type(chembl_dsets[target]) == dict:
            for endpoint in chembl_dsets[target].keys():
                bias_df = chembl_assay_bias(target, endpoint)

# ----------------------------------------------------------------------------------------------------------------------
# DTC-specific curation functions
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
def get_dtc_jak_smiles():
    """
    Use PubChem REST API to download SMILES strings for InChi strings in DTC JAK123 data table
    """
    jak_file = "%s/jak123_dtc.csv" % data_dirs['DTC']
    dset_df = pd.read_csv(jak_file, index_col=False)
    jak_dtc_df = jak_dtc_df[~jak_dtc_df.standard_inchi_key.isna()]
    inchi_keys = sorted(set(jak_dtc_df.standard_inchi_key.values))
    smiles_df, fail_list, discard_list = pu.download_smiles(inchi_keys)
    smiles_df.to_csv('%s/jak123_inchi_smiles.csv' % data_dirs['DTC'], index=False)

# ----------------------------------------------------------------------------------------------------------------------
def curate_dtc_jak_datasets():
    """
    Extract JAK1, 2 and 3 datasets from Drug Target Commons database, filtered for data usability.
    """
    # filter criteria:
    #   gene_names == JAK1 | JAK2 | JAK3
    #   InChi key not missing
    #   standard_type IC50
    #   units NM
    #   standard_relation mappable to =, < or >
    #   wildtype_or_mutant != 'mutated'
    #   valid SMILES
    #   maps to valid RDKit base SMILES
    #   standard_value not missing
    #   pIC50 > 3
    jak_file = "%s/jak123_dtc.csv" % data_dirs['DTC']
    dset_df = pd.read_csv(jak_file, index_col=False)
    # Get rid of nans in string columns we want to filter on
    for colname in ['standard_inchi_key', 'gene_names', 'standard_type', 'standard_units', 'standard_relation',
                    'wildtype_or_mutant']:
        dset_df[colname] = dset_df[colname].fillna('')
    jak_genes = ['JAK1', 'JAK2', 'JAK3']
    # Filter dataset on existing columns
    dset_df = dset_df[dset_df.gene_names.isin(jak_genes) &
                      ~(dset_df.standard_inchi_key == '') &
                      (dset_df.standard_type == 'IC50') &
                      (dset_df.standard_units == 'NM') &
                      ~dset_df.standard_value.isna() &
                      (dset_df.wildtype_or_mutant != 'mutated') ]
    # Standardize the relational operators
    dset_df = standardize_relations(dset_df, 'DTC')

    # Map the InChi keys to SMILES strings. Remove rows that don't map.
    smiles_file = "%s/jak123_inchi_smiles.csv" % data_dirs['DTC']
    smiles_df = pd.read_csv(smiles_file, index_col=False)[['standard_inchi_key', 'smiles']]
    smiles_df['smiles'] = [s.lstrip('"').rstrip('"') for s in smiles_df.smiles.values]
    dset_df = dset_df.merge(smiles_df, how='left', on='standard_inchi_key')
    dset_df = dset_df[~dset_df.smiles.isna()]

    # Add standardized desalted RDKit SMILES strings
    dset_df['rdkit_smiles'] = [base_smiles_from_smiles(s) for s in dset_df.smiles.values]
    dset_df = dset_df[dset_df.rdkit_smiles != '']

    # Add pIC50 values and filter on them
    dset_df['pIC50'] = 9.0 - np.log10(dset_df.standard_value.values)
    dset_df = dset_df[dset_df.pIC50 >= 3.0]

    # Add censoring relations for pIC50 values
    rels = dset_df['standard_relation'].values
    log_rels = rels.copy()
    log_rels[rels == '<'] = '>'
    log_rels[rels == '>'] = '<'
    dset_df['pIC50_relation'] = log_rels

    # Split into separate datasets by gene name
    curated_dir = "%s/curated" % data_dirs['DTC']
    os.makedirs(curated_dir, exist_ok=True)
    for gene in jak_genes:
        gene_dset_df = dset_df[dset_df.gene_names == gene]
        gene_dset_file = "%s/%s_DTC_curated.csv" % (curated_dir, gene)
        gene_dset_df.to_csv(gene_dset_file, index=False)
        print("Wrote file %s" % gene_dset_file)


# ----------------------------------------------------------------------------------------------------------------------
# Excape-specific curation functions
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def curate_excape_jak_datasets():
    """
    Extract JAK1, 2 and 3 datasets from Excape database, filtered for data usability.
    """

    # Filter criteria:
    #   pXC50 not missing
    #   rdkit_smiles not blank
    #   pXC50 > 3

    jak_file = "%s/jak123_excape_smiles.csv" % data_dirs['Excape']
    dset_df = pd.read_csv(jak_file, index_col=False)
    dset_df = dset_df[ ~dset_df.pXC50.isna() & ~dset_df.rdkit_smiles.isna() ]
    dset_df = dset_df[dset_df.pXC50 >= 3.0]
    jak_genes = ['JAK1', 'JAK2', 'JAK3']

    # Split into separate datasets by gene name
    curated_dir = "%s/curated" % data_dirs['Excape']
    os.makedirs(curated_dir, exist_ok=True)
    for gene in jak_genes:
        gene_dset_df = dset_df[dset_df.Gene_Symbol == gene]
        gene_dset_file = "%s/%s_Excape_curated.csv" % (curated_dir, gene)
        gene_dset_df.to_csv(gene_dset_file, index=False)
        print("Wrote file %s" % gene_dset_file)


# ----------------------------------------------------------------------------------------------------------------------
# Functions for comparing datasets from different sources
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
def compare_jak_dset_compounds():
    """
    Plot Venn diagrams for each set of public JAK datasets showing the numbers of compounds in common
    between them
    """
    jak_genes = ['JAK1', 'JAK2', 'JAK3']
    dbs = sorted(data_dirs.keys())
    for gene in jak_genes:
        dset_smiles = []
        for db in dbs:
            dset_file = "%s/curated/%s" % (data_dirs[db], curated_dset_file_templ[db] % gene)
            dset_df = pd.read_csv(dset_file, index_col=False)
            dset_smiles.append(set(dset_df.rdkit_smiles.values))
        fig, ax = plt.subplots(figsize=(8,8))
        venn3(dset_smiles, dbs)
        plt.title(gene)
        plt.show()

# ----------------------------------------------------------------------------------------------------------------------
def find_jak_dset_duplicates():
    """
    Check for potential duplication of records within and between datasets. A record is a potential
    duplicate if it has the same base SMILES string, IC50 value and standard relation.
    """
    colmap = dict(
        ChEMBL={'pIC50': 'pIC50',
                'LogVarRelation': 'Relation'
                },
        DTC={'pIC50': 'pIC50',
                'pIC50_relation': 'Relation'
                },
        Excape={'pXC50': 'pIC50'
             } )
    jak_genes = ['JAK1', 'JAK2', 'JAK3']
    dbs = sorted(data_dirs.keys())
    for gene in jak_genes:
        dedup = {}
        smiles_set = {}
        for db in dbs:
            dset_file = "%s/curated/%s" % (data_dirs[db], curated_dset_file_templ[db] % gene)
            dset_df = pd.read_csv(dset_file, index_col=False)
            dset_df = dset_df.rename(columns=colmap[db])
            if db == 'Excape':
                dset_df['Relation'] = "="
            dset_df = dset_df[['Relation', 'pIC50', 'rdkit_smiles']]
            is_dup = dset_df.duplicated().values
            print("Within %s %s dataset, %d/%d rows are potential duplicates" % (db, gene, sum(is_dup), dset_df.shape[0]))
            dedup[db] = dset_df.drop_duplicates()
            smiles_set[db] = set(dset_df.rdkit_smiles.values)
        print('\n')
        for i, db1 in enumerate(dbs[:2]):
            for db2 in dbs[i+1:]:
                combined_df = pd.concat([dedup[db1], dedup[db2]], ignore_index=True)
                is_dup = combined_df.duplicated().values
                n_cmn_smiles = len(smiles_set[db1] & smiles_set[db2])
                print("Between %s and %s %s datasets, %d common SMILES, %d identical responses" % (db1, db2, gene,
                                                                                                   n_cmn_smiles,
                                                                                                   sum(is_dup)))
        print('\n')


