import torch

import csv
import os


def get_causal_mask(seq_length):
    attn_mask = torch.triu(torch.ones((seq_length, seq_length), dtype=torch.bool), diagonal=1)
    return attn_mask

def log_to_csv(filename, data):
    """
    Logs data to a CSV file.
    
    Args:
        filename (str): The name of the CSV file.
        data (dict): A dictionary containing the data to log. 
                     Keys will be the header, values will be the row.
    """
    try:
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=data.keys())
            
            # Write header only if file is empty
            if csvfile.tell() == 0:
              writer.writeheader()
            writer.writerow(data)
    except Exception as e:
        print(f"An error occurred: {e}")


def create_training_directory(base_dir: str) -> str:
    """
    Creates a new directory named `base_dir`. If it already exists,
    it appends an incremental suffix (_1, _2, ...) until a non-existing
    directory is found, then creates it.

    Returns:
        str: The final path of the newly created directory.
    """
    dir_path = base_dir
    counter = 1
    while os.path.exists(dir_path):
        dir_path = f"{base_dir}_{counter}"
        counter += 1

    os.makedirs(dir_path)
    return dir_path, counter