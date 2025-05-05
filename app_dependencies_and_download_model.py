#!/usr/bin/env python3
"""
A script that checks for required dependencies, offers to install 
missing ones, and downloads a specified model.
"""

import os
import sys
import subprocess
import importlib.util


def check_dependency(package_name):
    is_installed = importlib.util.find_spec(package_name) is not None
    return is_installed


def install_package(package_name):
    print(f"Installing {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"Successfully installed {package_name}.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package_name}. Error: {e}")
        return False


def check_dependencies():
    # check if all required dependencies are installed
    required_packages = ["transformers", "anthropic", "torch"]
    missing_packages = []

    print("Checking dependencies...")

    for package in required_packages:
        if check_dependency(package):
            print(f"{package} is available.")
        else:
            missing_packages.append(package)
            print(f"{package} is missing.")

    return missing_packages


def request_installation(missing_packages):
    # prompt to install missing packages
    if not missing_packages:
        return []
    
    print("The following packages are missing:")
    for package in missing_packages:
        print(f"  - {package}")

    response = input("Would you like to install these missing packages? (y/n): ").strip().lower()
    
    if response == 'y' or response == 'yes':
        failed_packages = []
        for package in missing_packages:
            if not install_package(package):
                failed_packages.append(package)
        return failed_packages
    else:
        return missing_packages


def download_model(model_id, local_dir):
    # download model a local directory
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        print(f"Creating directory '{local_dir}' if it doesn't exist...")
        os.makedirs(local_dir, exist_ok=True)
        
        print(f"Downloading model '{model_id}'...")
        
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        print("Saving tokenizer...")
        tokenizer.save_pretrained(local_dir)
        
        print("Downloading model...")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        
        print("Saving model...")
        model.save_pretrained(local_dir)
        
        print(f"Model successfully downloaded to {local_dir}")
        return True
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        return False


def main():
    print("This is a script to install dependencies and download the model for the app.py file".center(60))
    print("//"*60 + "\n")
    
    # dependencies
    missing_packages = check_dependencies()
    
    # request install
    failed_packages = request_installation(missing_packages)
    
    if failed_packages:
        print("The following packages could not be installed:")
        for package in failed_packages:
            print(f"  - {package}")
        print("You may need to install them manually and/or run this script again.")
        sys.exit(1)
    
    # download model
    model_id = "sieg2011/codet5-base-sql-create-context"
    local_dir = "finetuned/codet5-base-sql-create-context"
    
    success = download_model(model_id, local_dir)
    
    if success:
        print("\n" + "//"*60)
        print("Download completed successfully!".center(60))
        print("//"*60)
        print(f"\nModel saved to: {os.path.abspath(local_dir)}\n")
    else:
        print("\n" + "//"*60)
        print("Download failed.".center(60))
        print("//"*60)
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        sys.exit(1)