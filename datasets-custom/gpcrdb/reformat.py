"""
Reformat csv file downloaded from https://gpcrdb.org/ligand/
"""
import os, sys
import pandas as pd
import click
import glob


# Settings
col_names = [
    "MOLNAME",
    "GPCR ID",
    "Type",
    "Reference Ligand",
    "Fold selectivity",
    "# Tested GPCR",
    "Species",
    "# of Vendors",
    "Assay Type",
    "# of Records",
    "Min",
    "Average",
    "Max",
    "Value",
    "Source",
    "MW",
    "RotBonds",
    "Hdon",
    "Hacc",
    "LogP",
    "SMILES"
]



def report(df):
    n_data = len(df)
    assay_type = list(df['Assay Type'].unique())
    species = list(df['Species'].unique())
    value = list(df['Value'].unique())
    vmin = df['Average'].min()
    vmax = df['Average'].max()
    print(f"Number of data: {n_data}")
    print(f"Unique species: {species}")
    print(f"Unique assay type: {assay_type}")
    print(f"Unique assay value: {value}")
    print(f"Value range: {vmin:.2f}-{vmax:.2f}")
    

def run(kwargs):
    input_prefix = kwargs['input_prefix']
    min_datasize = kwargs['min_datasize']

    files = glob.glob(f"{input_prefix}/*.csv")
    for file in files:
        filename_noext = os.path.splitext(os.path.basename(file))[0]
        print(f"{filename_noext}")
        print("----------------------")
        df = pd.read_csv(file, sep=';', header=None, skiprows=1)
        df.columns = col_names

        # Drop entries with SMILES=None
        df.drop(df[df['SMILES'] == 'None'].index, inplace=True)

        # Full report
        print(f"# Full data")
        report(df)
        
        # Drop assay values with 4, 5, and 6 to avoid assay measurement uncertainty.
        n_remove = len(df[df['Average'].isin([4,5,6])])
        print(f"Found {n_remove} assay measurements with values 4, 5, or 6. Remove from data.")
        df = df[df['Average'].isin([4,5,6]) == False]

        # Select columns
        assay_values = list(df['Value'].unique())
        for assay_value in assay_values:            
            _df = df[df['Value'] == assay_value]
            if len(_df) > min_datasize:
                print(f"# {assay_value}")
                report(_df)
                df_sel = _df[['SMILES', 'MOLNAME', 'Average']]
                #df_sel.rename({"Average": "VALUE"}, axis='columns')
                # Rename all molecule names that is not written in alphabets and number
                for idx, row in df_sel.iterrows():
                    if not row['MOLNAME'].isalnum():
                        #print(f"{idx}, {row}")
                        df_sel.at[idx, 'MOLNAME'] = 'CPD' + str(idx)
                df_sel_drop = df_sel.drop_duplicates(subset=['SMILES', 'MOLNAME'], keep=False)
                if len(df_sel) != len(df_sel_drop):
                    print(f"<WARNING> Removed {len(df_sel)-len(df_sel_drop)} duplicate SMILES/MOLNAME from data set.")
                df_sel_drop.to_csv(os.path.join(input_prefix, filename_noext + "_" + assay_value.lower() + ".smi"), sep=',', index=False, header=False)
        print("\n")


@click.command()
@click.option("--input_prefix", default="./gpcrdb", required=True, help="input path to your file")
@click.option("--min_datasize", default=100, help="minimal number of dataset to save")
def cli(**kwargs):
    run(kwargs)


if __name__ == '__main__':
    cli()

