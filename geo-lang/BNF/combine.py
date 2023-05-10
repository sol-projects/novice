import glob

def combine_bnf_files():
    output_file = "language.bnf"
    file_list = glob.glob("*.bnf")

    with open(output_file, 'w') as outfile:
        for filename in file_list:
            with open(filename, 'r') as infile:
                for line in infile:
                    if not line.startswith('@'):
                        outfile.write(line)

    print(f"Combined BNF files into: {output_file}")

combine_bnf_files()
