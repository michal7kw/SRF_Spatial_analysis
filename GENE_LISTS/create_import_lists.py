import pandas as pd
import itertools

# --- Configuration ---

# The main gene list exported from the MERSCOPE Vizualizer.
main_gene_file = 'export.csv'

# A list of the new files containing your multi-column gene lists.
gene_group_files = [
    'Hippocampus.csv',
    'Hippocampus_dev.csv'
]

# The final output file that you will import back into the Vizualizer.
formatted_output_file = 'import_ready.csv'

# A list of visually distinct colors for the new groups.
distinct_colors = [
    '#E6194B', '#3CB44B', '#FFE119', '#4363D8', '#F58231', '#911EB4', '#42D4F4',
    '#F032E6', '#BFEF45', '#FABEBE', '#469990', '#E6BEFF', '#9A6324', '#800000',
    '#AAFFC3', '#808000', '#FFD8B1', '#000075', '#A9A9A9'
]
color_cycler = itertools.cycle(distinct_colors)


# --- Main Script ---

print("🚀 Starting the gene group formatting process...")

try:
    # Load the CSV without a header and assign the correct column names.
    main_df = pd.read_csv(
        main_gene_file,
        header=None,
        names=['gene group', 'gene name', 'gene description', 'gene color']
    )
    print(f"✅ Successfully loaded '{main_gene_file}' and assigned correct headers.")
    # Use the 'gene name' column as the index for efficient updating
    main_df.set_index('gene name', inplace=True)
except FileNotFoundError:
    print(f"\n❌ Error: The main export file '{main_gene_file}' was not found.")
    print("Please make sure it's in the same folder as the script.")
    exit()


# Process each of the gene list files
for file in gene_group_files:
    try:
        print(f"\nProcessing file: '{file}'...")
        group_df = pd.read_csv(file)

        # Iterate over each column in the current file.
        for group_name in group_df.columns:
            # Get the list of genes, dropping NaNs and duplicates.
            gene_list = group_df[group_name].dropna().unique().tolist()
            
            # Assign the next available color from the list
            group_color = next(color_cycler)
            
            print(f"  - Assigning group '{group_name}' with color {group_color}...")
            
            genes_found_count = 0
            for gene in gene_list:
                if gene in main_df.index:
                    # Update the 'gene group' and 'gene color' for the matching gene
                    main_df.loc[gene, 'gene group'] = group_name
                    main_df.loc[gene, 'gene color'] = group_color
                    genes_found_count += 1
                else:
                    print(f"    - Warning: Gene '{gene}' from group '{group_name}' was not found in '{main_gene_file}' and will be skipped.")
            
            print(f"    - Assigned {genes_found_count} of {len(gene_list)} genes to this group.")

    except FileNotFoundError:
        print(f"\n❌ Warning: The file '{file}' was not found and will be skipped.")


# Convert the index back to a column
main_df.reset_index(inplace=True)

# --- NEW FIX ---
# Define the correct column order required by the Vizualizer
correct_column_order = ['gene group', 'gene name', 'gene description', 'gene color']
# Reorder the DataFrame columns to match the required format
main_df = main_df[correct_column_order]
print("\n✅ Columns reordered to match original format.")

# Save the final, updated DataFrame.
# "header=False" ensures no column names are written to the file, matching your example.
main_df.to_csv(formatted_output_file, index=False, header=False)

print(f"\n✅ Success! The file '{formatted_output_file}' has been created with the correct format.")