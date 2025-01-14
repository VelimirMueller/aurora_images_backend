import requests
import os

# Function to download an image
def image_downloader(url, target_directory):
    response = requests.get(url)
    if response.status_code == 200:
        
        image_path = url.split("/")[-1] # Extract the image name from the URL
        
        if not os.path.exists(target_directory): # Create the folder if it doesn't exist
            os.makedirs(target_directory)
        
        with open(os.path.join(target_directory, image_path), 'wb') as f: # Save the image to the specified folder
            f.write(response.content)
        print(f"Downloaded: {image_path}")
    else:
        print(f"Failed to retrieve image from {url}")
