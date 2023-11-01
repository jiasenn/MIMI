from bs4 import BeautifulSoup
import requests
import csv
import re

def clean_text(text):
    text = re.sub(r'\[\d+\]', '', text)  # Remove references like [1], [2], ...
    text = re.sub(r'\[citation needed\]', '', text)  # Remove [citation needed]
    text = ' '.join(text.split())  # Remove excessive whitespace
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    return text

url = 'https://www.roots.gov.sg/stories-landing/stories/tan-kah-kee/story'
response = requests.get(url)

soup = BeautifulSoup(response.content, 'html.parser')
title = clean_text(soup.find('title').text)

# Extracting and cleaning all paragraph texts
paragraphs = soup.find_all('p')
data = [["Title/Paragraph", "Content"]]  # Header for the CSV
data.append(["Title", title])
for p in paragraphs:
    cleaned_paragraph = clean_text(p.text)
    data.append(["Paragraph", cleaned_paragraph])

# Save to .csv file
with open("root_tankahkee", "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

