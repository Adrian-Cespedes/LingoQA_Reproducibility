import gdown

# File ID of the file you want to download (replace with your file ID)
zip_file_id = '1U5tBTZjw0rH9ID1JuuWsOmIp2HNvQvQo'
parquet_file_id = '1I8u6uYysQUstoVYZapyRQkXmOwr-AG3d'

gdown.download(f'https://drive.google.com/uc?export=download&id={zip_file_id}', quiet=False)
gdown.download(f'https://drive.google.com/uc?export=download&id={parquet_file_id}', quiet=False)