import os

def check_file_exists(folder_path, file_name):
    # Construct the full path of the file
    file_path = os.path.join(folder_path, file_name)
    
    # Check if the file exists
    if os.path.isfile(file_path):
        print(f"File '{file_name}' exists in the folder '{folder_path}'.")
        return True
    else:
        print(f"File '{file_name}' does not exist in the folder '{folder_path}'.")
        return False

