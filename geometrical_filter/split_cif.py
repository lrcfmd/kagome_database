import argparse
import pathlib
import re
from collections import defaultdict
import os
from concurrent.futures import ThreadPoolExecutor


def group_lines_by_element(lines):
    groups = defaultdict(list)
    for line in lines:
        i = re.search(r'[0-9]', line).start()
        el_symbol = line[:i]
        groups[el_symbol].append(line)
    return groups
    
def extract_section(file_path, section_heading="_atom_site_occupancy"):
    with file_path.open(encoding="utf8") as f:
        section = 0
        start_str, relevant_lines, end_str = '', [], ''
        while line := f.readline():
            if section == 1 and (line.startswith("_") or line.startswith("loop")) or line.startswith("#End"):
                # finished the relevant section on this line
                section = 2      

            if section == 0: start_str += line
            if section == 1: relevant_lines.append(line)
            if section == 2: end_str += line

            if line.startswith(section_heading):
                # start of the relevant section on next line
                section = 1

    return start_str, relevant_lines, end_str

def split_cif(file_path):
    start_str, relevant_lines, end_str = extract_section(file_path)
    el_groups = group_lines_by_element(relevant_lines)
    new_cifs = {}
    for el, el_lines in el_groups.items():
        new_file_name = file_path.stem+'_'+el+'.cif'
        new_cifs[new_file_name] = start_str + ''.join(el_lines) + end_str
    #print(file_path)
    return new_cifs


def write_out(new_cifs, out_folder):
    for new_cif_path, new_cif_content in new_cifs.items():
        out_file = pathlib.Path(out_folder, new_cif_path)
        with out_file.open('w', encoding="utf8") as f:
            f.write(new_cif_content)

def main(cif_path, verbose=False):
    def process_batch(batch_id, batch_data):
        if verbose:
            print(f'processing batch {batch_id}')
        out_folder = pathlib.Path(cif_path, f'extracted_cifs_{batch_id}')
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        new_cifs = {}            
        for file in batch_data:
            new_cifs.update(split_cif(file))
        write_out(new_cifs, out_folder)  
        if verbose:
            print(f'completed batch {batch_id}.')         

    batch_size = 500 # batched to avoid creating enormous directories which are then very slow to open
    if cif_path.is_dir():
        all_files = [f for f in cif_path.iterdir() if f.is_file()]
        batched_files = [all_files[i:i+batch_size] for i in range(0, len(all_files), batch_size)]
        process_batch(0, batched_files[0])
       # print(len(all_files))
        with ThreadPoolExecutor() as exec:
            exec.map(process_batch, range(len(batched_files)), batched_files)

    elif cif_path.is_file(): 
        out_folder = pathlib.Path(cif_path.parent, 'extracted_cifs')
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        file = cif_path
        new_cifs = split_cif(file)
        write_out(new_cifs, out_folder)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create single element cifs.')
    parser.add_argument('path', type=pathlib.Path)
    parser.add_argument('-v', '--v', action='store_true')

    args = parser.parse_args()
    cif_path = args.path
    verbose = args.v
    print(cif_path.is_dir())
    
    from time import time
    start = time()
    main(cif_path, verbose)
    print(time() - start)

