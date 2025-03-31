"""
Python Glacier Evolution Model (PyGEM)

copyright © 2024 Brandon Tober <btober@cmu.edu> David Rounce <drounce@cmu.edu>

Distrubted under the MIT lisence

initialization script (ensure config.yaml and get sample datasets)
"""
import requests
import zipfile
import os,sys
import shutil
from pygem.setup.config import ConfigManager
# instantiate ConfigManager
config_manager = ConfigManager(overwrite=True)
# read the config
pygem_prms = config_manager.read_config()

def print_file_tree(start_path, indent=""):
    # Loop through all files and directories in the current directory
    for item in os.listdir(start_path):
        path = os.path.join(start_path, item)
        
        # Print the current item with indentation
        print(indent + "|-- " + item)
        
        # Recursively call this function if the item is a directory
        if os.path.isdir(path):
            print_file_tree(path, indent + "    ")
            
def get_confirm_token(response):
    """Extract confirmation token for Google Drive large file download."""
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None

def save_response_content(response, destination):
    """Save the response content to a file."""
    chunk_size = 32768
    with open(destination, "wb") as file:
        for chunk in response.iter_content(chunk_size):
            if chunk:  # Filter out keep-alive chunks
                file.write(chunk)

def get_unique_folder_name(dir):
    """Generate a unique folder name by appending a suffix if the folder already exists."""
    counter = 1
    unique_dir = dir
    while os.path.exists(unique_dir):
        unique_dir = f"{dir}_{counter}"
        counter += 1
    return unique_dir

def download_and_unzip_from_google_drive(file_id, output_dir):
    """
    Download a ZIP file from Google Drive and extract its contents.
    
    Args:
        file_id (str): The Google Drive file ID.
        output_dir (str): The directory to save and extract the contents of the ZIP file.
    
    Returns:
        int: 1 if the ZIP file was successfully downloaded and extracted, 0 otherwise.
    """
    # Google Drive URL template
    base_url = "https://drive.google.com/uc?export=download"

    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Path to save the downloaded file
    zip_path = os.path.join(output_dir, "tmp_download.zip")

    try:
        # Start the download process
        with requests.Session() as session:
            response = session.get(base_url, params={"id": file_id}, stream=True)
            token = get_confirm_token(response)
            if token:
                response = session.get(base_url, params={"id": file_id, "confirm": token}, stream=True)
            save_response_content(response, zip_path)

        # Unzip the file
        tmppath = os.path.join(output_dir, 'tmp')
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmppath)

        # get root dir name of zipped files
        dir = [item for item in os.listdir(tmppath) if os.path.isdir(os.path.join(tmppath, item))][0]
        unzip_dir = os.path.join(tmppath, dir)
        # get unique name if root dir name already exists in output_dir
        output_dir = get_unique_folder_name(os.path.join(output_dir, dir))
        # move data and cleanup
        shutil.move(unzip_dir, output_dir)
        shutil.rmtree(tmppath)
        os.remove(zip_path)
        return output_dir  # Success

    except (requests.RequestException, zipfile.BadZipFile, Exception) as e:
        return None  # Failure
    
def main():
    # Define the base directory
    basedir = os.path.join(os.path.expanduser('~'), 'PyGEM')
    # Google Drive file id for sample dataset
    file_id = "1Wu4ZqpOKxnc4EYhcRHQbwGq95FoOxMfZ"
    # download and unzip
    out = download_and_unzip_from_google_drive(file_id, basedir)

    if out:
        print(f"Downloaded PyGEM sample dataset:")
        print(os.path.abspath(out))
        try:
            print_file_tree(out)
        except:
            pass

    else:
        print(f'Error downloading PyGEM sample dataset.')

    # update root path in config.yaml
    try:
        config_manager.update_config(updates={'root':f'{out}/sample_data'})
    except:
        pass
    
if __name__ == "__main__":
    main()