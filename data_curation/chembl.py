"""
Simple module for connecting to local sqlite CHEMBL database
and running queries. Also includes some functions that execute example
queries.
"""

import os
import sys
#import pymysql
import sqlite3 as sql
import pandas as pd
import glob
##from atomsci.ddm.utils import get_env ##** commented by Titli

chembl_dir = '/p/vast1/atom/chembl'

def connect(version=None):
  """
  Connect to the specified version of the CHEMBL DB, defaulting to the most recent downloaded version
  """
  if version is None:
    version = get_current_version()
  chembl_path = f"{chembl_dir}/chembl_{version}.db"
  return sql.connect(chembl_path)

def get_current_version():
  """
  Get the latest downloaded version of the ChEMBL database
  """
  db_files = glob.glob(f"{chembl_dir}/chembl_*.db")
  return max([int(os.path.basename(f).rstrip('.db').lstrip('chembl_')) for f in db_files])

def query(con, sql):
  """
  Execute the specified query and return the results in a data frame.
  """

  try:
    res_df = pd.read_sql(sql, con=con)
    print("Query returned %d rows" % res_df.shape[0])
    return res_df
  except Exception as e:
    print(e.message, file=sys.stderr)
    raise

def quit(con):
  """
  Quit the sqlite session
  """
  con.close()


def get_human_vdss():
    """
    Query for steady state volume of distribution in humans
    """
    with connect() as con:
        sql = ' '.join([
                     "SELECT DISTINCT act.activity_id, mol.chembl_id AS mol_chembl_id, cr.compound_name, a.assay_type, a.description, act.standard_type,",
                     "act.standard_relation, act.standard_value, act.standard_units, cs.canonical_smiles, doc.chembl_id AS doc_chembl_id, doc.journal, doc.title, doc.year", 
                     "FROM assays a, activities act, compound_records cr, compound_structures cs, target_dictionary t, molecule_dictionary mol, docs doc",
                     "WHERE a.tid = t.tid",
                     "AND (a.assay_organism = 'Homo sapiens' OR (a.assay_organism IS NULL AND a.description LIKE '%in man%'))",
                     "AND NOT (a.description LIKE '%cancer%' OR a.description LIKE '%tumor%')",
                     "AND t.pref_name = 'ADMET'",
                     "AND act.assay_id = a.assay_id",
                     "AND act.standard_type = 'Vdss'",
                     "AND act.molregno = cr.molregno",
                     "AND act.molregno = mol.molregno",
                     "AND act.doc_id = doc.doc_id",
                     "AND cs.molregno = cr.molregno"])
        res_df = pd.read_sql(sql, con=con)
    res_df = res_df.drop_duplicates(subset='activity_id')
    return res_df
    
def get_human_cl():
    """
    Query for clearance in humans
    """
    with connect() as con:
        sql = ' '.join([
                     "SELECT DISTINCT act.activity_id, mol.chembl_id AS mol_chembl_id, cr.compound_name, a.assay_type, a.description, act.standard_type,",
                     "act.standard_relation, act.standard_value, act.standard_units, cs.canonical_smiles, doc.chembl_id AS doc_chembl_id, doc.journal, doc.title, doc.year", 
                     "FROM assays a, activities act, compound_records cr, compound_structures cs, target_dictionary t, molecule_dictionary mol, docs doc",
                     "WHERE a.tid = t.tid",
                     "AND (a.assay_organism = 'Homo sapiens' OR (a.assay_organism IS NULL AND a.description LIKE '%in man%'))",
                     "AND NOT (a.description LIKE '%cancer%' OR a.description LIKE '%tumor%')",
                     "AND t.pref_name = 'ADMET'",
                     "AND act.assay_id = a.assay_id",
                     "AND act.standard_type = 'CL'",
                     "AND act.molregno = cr.molregno",
                     "AND act.molregno = mol.molregno",
                     "AND act.doc_id = doc.doc_id",
                     "AND cs.molregno = cr.molregno"])
        res_df = pd.read_sql(sql, con=con)
    res_df = res_df.drop_duplicates(subset='activity_id')
    return res_df
    
def get_human_fup():
    """
    Query for fraction unbound in plasma in humans
    """
    with connect() as con:
        sql = ' '.join([
                     "SELECT DISTINCT act.activity_id, mol.chembl_id AS mol_chembl_id, cr.compound_name, a.assay_type, a.description, act.standard_type,",
                     "act.standard_relation, act.standard_value, act.standard_units, cs.canonical_smiles, doc.chembl_id AS doc_chembl_id, doc.journal, doc.title, doc.year", 
                     "FROM assays a, activities act, compound_records cr, compound_structures cs, target_dictionary t, molecule_dictionary mol, docs doc",
                     "WHERE a.tid = t.tid",
                     "AND (a.assay_organism = 'Homo sapiens' OR (a.assay_organism IS NULL AND a.description LIKE '%in man%'))",
                     "AND NOT (a.description LIKE '%cancer%' OR a.description LIKE '%tumor%')",
                     "AND t.pref_name = 'ADMET'",
                     "AND act.assay_id = a.assay_id",
                     "AND (act.standard_type = 'Fu' OR act.standard_type = 'PPB')",
                     "AND act.molregno = cr.molregno",
                     "AND act.molregno = mol.molregno",
                     "AND act.doc_id = doc.doc_id",
                     "AND cs.molregno = cr.molregno"])
        res_df = pd.read_sql(sql, con=con)
    res_df = res_df.drop_duplicates(subset='activity_id')
    return res_df
    
def get_admet_types():
    """
    Query for standard types of data with target ADMET
    """
    with connect() as con:
        sql = ' '.join([
                     "SELECT act.standard_type, COUNT(DISTINCT act.activity_id) AS num_records",
                     "FROM activities act, assays a, target_dictionary t",
                     "WHERE a.tid = t.tid AND act.assay_id = a.assay_id",
                     "AND t.pref_name = 'ADMET'",
                     "GROUP BY act.standard_type" ])
        res_df = pd.read_sql(sql, con=con)
    return res_df
    

def get_ar_assays(con):
  """
  Find all results for assays targeting the human androgen receptor; returns them in a data frame.
  """

  sql = ' '.join([
                 "SELECT cr.compound_name, a.assay_type, a.description, v.standard_type,",
                 "v.standard_relation, v.standard_value, v.standard_units, cs.canonical_smiles", 
                 "FROM assays a, activities v, compound_records cr, compound_structures cs, target_dictionary t",
                 "WHERE a.tid = t.tid AND t.organism = 'Homo sapiens' AND t.pref_name = 'Androgen Receptor'",
                 "AND v.assay_id = a.assay_id",
                 "AND v.molregno = cr.molregno",
                 "AND cs.molregno = cr.molregno"])
  res_df = pd.read_sql(sql, con=con)
  return res_df

def get_tox21_assays(con):
  """
  Return information about assays used for the Tox21 project
  """
  sql = ' '.join([
                  "SELECT a.assay_id, t.pref_name, a.assay_type, a.description, a.assay_test_type, a.assay_category",
                  "FROM assays a, target_dictionary t",
                  "WHERE a.description LIKE '%Tox21%'",
                  "AND a.tid = t.tid",
                  "ORDER BY t.pref_name, a.description"
                  ])
  with con.cursor() as cursor:
    res_df = pd.read_sql(sql, con=con)
  print("Query returned %d rows" % res_df.shape[0])
  res_file = os.path.join(chembl_dir, 'extracts/tox_21_assays.xlsx')
  res_df.to_excel(res_file, index=False)
  return res_df


def dump_chembl():
  """
  Dump SMILES strings with associated names and IDs to a table file.
  """
  con = connect()
  print("Connected to chembl DB")
  sql = ' '.join([
                    "SELECT cs.molregno, cr.compound_name, cs.canonical_smiles",
                    "FROM compound_records cr, compound_structures cs",
                    "WHERE cs.molregno = cr.molregno",
                    "AND cs.canonical_smiles IS NOT NULL" ])
  res_df = query(con, sql)
  out_file = os.path.join(chembl_dir, 'extracts/chembl_smiles_strs.txt')
  res_df.to_csv(out_file, sep='\t', index=False)
  print(("Wrote file %s" % out_file))
  quit(con)

def get_generic_names():
  """
  Create two lists of generic drug names: one combining FDA and WHO/INN
  names with no other information, the other matching FDA names to
  molregnos.
  """
  con = connect()
  print("Connected to CHEMBL DB")
  sql = ' '.join([
                    "SELECT DISTINCT who_name",
                    "FROM atc_classification" ])
  who_names_df = query(con, sql)
  who_names = who_names_df.who_name.values.tolist()
  print(("Got %d unique WHO drug names" % len(who_names)))
  sql = ' '.join([
                    "SELECT DISTINCT molregno, ingredient",
                    "FROM formulations" ])
  fda_names_df = query(con, sql)
  quit(con)

  fda_names = [name.lower() for name in fda_names_df.ingredient.values]
  fda_names_df['fda_name'] = fda_names
  uniq_fda_names = set(fda_names)
  uniq_molregnos = set(fda_names_df.molregno.values.tolist())
  print(("Got %d rows, %d unique FDA drug names, %d unique molregnos" 
    % (fda_names_df.shape[0], len(uniq_fda_names), len(uniq_molregnos))))

  # List molregnos that map to more than one FDA name
  mol_dict = {}
  for i in range(fda_names_df.shape[0]):
    molregno = fda_names_df.molregno.iloc[i]
    fda_name = fda_names_df.fda_name.iloc[i]
    mol_dict.setdefault(molregno, []).append(fda_name)
  for molregno, name_list in mol_dict.items():
    if len(name_list) > 1:
      print(("molregno %d -> [%s]" % (molregno, ', '.join(name_list))))

  # Write FDA names to a table
  fda_file = os.path.join(chembl_dir, 'extracts/chembl_fda_generic_names.txt')
  fda_names_df.to_csv(fda_file, sep='\t', index=False,
    columns=['molregno', 'fda_name'])
  print(("Wrote file %s" % fda_file))

  # Write combined WHO and FDA names to a file
  generic_names = sorted(uniq_fda_names | set(who_names))
  generic_df = pd.DataFrame({'name' : generic_names} )
  generic_file = os.path.join(chembl_dir, 
    'extracts/chembl_combined_generic_names.txt')
  generic_df.to_csv(generic_file, header=False, index=False)
  print(("Wrote file %s" % generic_file))


def generic_name_chembl_ids():
  """
  Create table matching FDA and WHO/INN generic drug names to ChEMBL compound IDs.
  """
  con = connect()
  print("Connected to CHEMBL DB")
  # First query for WHO/INN names
  sql = ' '.join([
                    "SELECT DISTINCT atc.who_name, chid.chembl_id",
                    "FROM atc_classification atc, chembl_id_lookup chid, molecule_atc_classification mac",
                    "WHERE atc.level5 = mac.level5",
                    "AND mac.molregno = chid.entity_id",
                    "AND chid.entity_type = 'COMPOUND'" ])
  who_names_df = query(con, sql)
  who_names_df['source'] = 'WHO/INN'
  who_names = who_names_df.who_name.values.tolist()
  print(("Got %d unique WHO drug names" % len(who_names)))

  # Then query for FDA ingredient names
  sql = ' '.join([
                    "SELECT DISTINCT frm.ingredient, chid.chembl_id",
                    "FROM formulations frm, chembl_id_lookup chid",
                    "WHERE frm.molregno = chid.entity_id",
                    "AND chid.entity_type = 'COMPOUND'" ])
  fda_names_df = query(con, sql)
  quit(con)
  fda_names_df['source'] = 'FDA'
  #generic_df = pd.concat([fda_names_df

  # Write FDA names to a table
  fda_file = os.path.join(chembl_dir, 'extracts/fda_generic_name_chembl_ids.csv')
  fda_names_df.to_csv(fda_file, index=False)
  print(("Wrote file %s" % fda_file))

  # Write WHO names to a file
  who_file = os.path.join(chembl_dir, 'extracts/who_inn_generic_name_chembl_ids.csv')
  who_names_df.to_csv(who_file, index=False)
  print(("Wrote file %s" % who_file))

def get_smiles_for_chembl_ids(id_list):
  """
  Look up SMILES strings for a list of compounds identified by ChEMBL IDs
  """
  con = connect()
  sql = ' '.join([
                  "SELECT DISTINCT chid.chembl_id, cs.canonical_smiles",
                  "FROM chembl_id_lookup chid, compound_structures cs",
                  "WHERE chid.entity_id = cs.molregno",
                  "AND chid.entity_type = 'COMPOUND'",
                  "AND chid.chembl_id = '%s'"])
  smiles_df_list = []
  for chembl_id in id_list:
    id_smiles_df = query(con, sql % chembl_id)
    smiles_df_list.append(id_smiles_df)
  smiles_df = pd.concat(smiles_df_list, ignore_index=True)
  return smiles_df

def get_molecule_types():
    """
    Get distinct values of molecule_dictionary.molecule_type
    """
    with connect() as con:
        sql = ' '.join([
                     "SELECT DISTINCT mol.molecule_type",
                     "FROM molecule_dictionary mol",
                     "ORDER BY mol.molecule_type"])
        res_df = pd.read_sql(sql, con=con)
    return res_df


def get_large_molecules(min_mw=1000.0, version=get_current_version()):
    """
    Query for all molecule records with molecular weight >= min_mw
    """
    with connect(version) as con:
        sql = ' '.join([
                     "SELECT mol.chembl_id AS mol_chembl_id, mol.pref_name, cr.compound_name, mol.molecule_type, mol.natural_product, mol.structure_type,"
                     "cs.canonical_smiles, cs.standard_inchi, cp.mw_freebase, cp.full_mwt, cp.heavy_atoms",
                     "FROM molecule_dictionary mol, compound_records cr, compound_structures cs, compound_properties cp",
                     "WHERE",
                     "cs.molregno = mol.molregno",
                     "AND cr.molregno = mol.molregno",
                     "AND cp.molregno = mol.molregno",
                     f"AND cp.full_mwt > {min_mw}"
                     ])
        res_df = pd.read_sql(sql, con=con)
    res_df = res_df.drop_duplicates(subset='mol_chembl_id')
    return res_df
