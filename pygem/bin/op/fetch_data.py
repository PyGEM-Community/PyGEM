import os
import zipfile
import requests
from urllib.parse import urlparse
from pygem.setup.config import ConfigManager
# instantiate ConfigManager
config_manager = ConfigManager()
# read the config
pygem_prms = config_manager.read_config()

# Check if "sample_data" is in the root directory and set default directory
if "sample_data" not in pygem_prms["root"]:
    DEFAULT_DIR = pygem_prms["root"]
else:
    DEFAULT_DIR = os.path.expanduser('~/PyGEM/pygem_data/')

# URL of the zip file
ZIP_FILE_URL = "https://cmu.box.com/shared/static/ret0765ek2meef8kl4riaxn5bgwy4ifi.zip"

def download_and_unzip(zip_file_url, target_dir):
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Download the zip file
    print(f"Downloading zip file from {zip_file_url}...")
    response = requests.get(zip_file_url)

    # Check if the response is successful and if it's a zip file
    if response.status_code == 200:
        # Check the content type
        content_type = response.headers.get('Content-Type')
        if 'zip' not in content_type:
            print(f"Error: The file downloaded is not a zip file. Content-Type: {content_type}")
            return

        # Get the filename from the URL
        zip_filename = os.path.basename(urlparse(zip_file_url).path)
        zip_file_path = os.path.join(target_dir, zip_filename)

        # Save the downloaded zip file
        with open(zip_file_path, 'wb') as f:
            f.write(response.content)

        # Unzip the file, keeping the nested structure intact
        print(f"Unzipping {zip_filename} to {target_dir}...")
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)  # This keeps the directory structure
            print(f"Unzipped {zip_filename} successfully.")
        except zipfile.BadZipFile:
            print(f"Error: The downloaded file is not a valid zip file.")
            return

        # Remove the zip file after extraction
        os.remove(zip_file_path)
    else:
        print(f"Failed to download {zip_file_url}. Status Code: {response.status_code}")

def main():
    # Prompt user for custom directory location with input validation
    print(f"Downloading and unzipping the file to:\t{DEFAULT_DIR}")
    while True:
        user_input = input("Would you like to download and unzip to the default location? (y/n): ").strip().lower()
        if user_input in ["n", "no"]:
            custom_dir = input("Enter the full path for the directory: ")
            target_dir = os.path.expanduser(custom_dir)
            break
        elif user_input in ['y', 'yes']:
            target_dir = DEFAULT_DIR
            break
        else:
            continue  # This continues the loop without reprinting the prompt

    # Download and unzip the file
    download_and_unzip(ZIP_FILE_URL, target_dir)

    print(f"The zip file has been downloaded, unzipped, and stored at {target_dir}.")

if __name__ == "__main__":
    main()
