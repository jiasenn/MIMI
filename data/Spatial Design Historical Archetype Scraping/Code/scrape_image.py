import requests
from bs4 import BeautifulSoup
import os
import csv
from PIL import Image
from io import BytesIO

# Define the URL of the webpage you want to scrape
URL = 'https://www.roots.gov.sg/en/stories-landing/stories/destination-nanyang/story'

# Fetch the content of the webpage
response = requests.get(URL)
soup = BeautifulSoup(response.content, 'html.parser')

# Find all image tags
img_tags = soup.find_all('img')

# Create a directory to save the images
if not os.path.exists('downloaded_images'):
    os.makedirs('downloaded_images')

# List to store image details
image_details = []

# Loop through the image tags and download each image
for img in img_tags:
    img_url = img['src']
    img_name = os.path.basename(img_url)
    alt_text = img.get('alt', '')  # Alt text of the image, if available
    
    # Downloading the image
    img_data = requests.get(img_url).content
    with open(f'downloaded_images/{img_name}', 'wb') as handler:
        handler.write(img_data)

    # Getting image dimensions
    image = Image.open(BytesIO(img_data))
    width, height = image.size

    # Append details to the list
    image_details.append({
        'Name': img_name,
        'URL': img_url,
        'Alt Text': alt_text,
        'Width': width,
        'Height': height,
    })

# Save the details to a CSV file
with open('image_details.csv', 'w', newline='') as csvfile:
    fieldnames = ['Name', 'URL', 'Alt Text', 'Width', 'Height']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for detail in image_details:
        writer.writerow(detail)

print("Images downloaded and details saved to 'image_details.csv' successfully!")
