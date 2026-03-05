import gdown
import zipfile
import os

url = "https://drive.google.com/uc?export=download&id=1W5R9vuxAMBAIv-N1Y-W6SRUR9D9WJuqq"
zip_file = "ai_model_assets.zip"

if not os.path.exists(zip_file):
    print("Downloading ai_model_assets.zip...")
    gdown.download(url, zip_file, quiet=False)
else:
    print("ai_model_assets.zip already exists.")

with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(".")
    print("Files extracted.")