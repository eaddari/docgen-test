from azure.storage.blob import BlobServiceClient
import os
from dotenv import load_dotenv

load_dotenv()
azure_storage_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME")


def upload_file_to_blob(file_path, blob_name):
    blob_service_client = BlobServiceClient.from_connection_string(azure_storage_connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    with open(file_path, "rb") as data:
        container_client.upload_blob(name=blob_name, data=data, overwrite=True)
    print(f"Uploaded {file_path} to blob storage as {blob_name}")


def upload_folder(folder_path, parent_path=""):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        blob_name = os.path.join(parent_path, filename) if parent_path else filename
        if os.path.isfile(file_path):
            upload_file_to_blob(file_path, blob_name)
        elif os.path.isdir(file_path):
            upload_folder(file_path, blob_name)


def download_blob_container(azure_storage_connection_string, container_name, output_folder, blob_name=None):
    blob_service_client = BlobServiceClient.from_connection_string(azure_storage_connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    os.makedirs(output_folder, exist_ok=True)
    if blob_name:
        blobs = container_client.list_blobs(name_starts_with=blob_name + "/")
        for blob in blobs:
            blob_client = container_client.get_blob_client(blob.name)
            relative_path = os.path.relpath(blob.name, blob_name + "/")
            blob_path = os.path.join(output_folder, relative_path)
            os.makedirs(os.path.dirname(blob_path), exist_ok=True)
            with open(blob_path, "wb") as file:
                data = blob_client.download_blob()
                file.write(data.readall())
            print(f"Downloaded {blob.name} to {blob_path}")
    else:
        for blob in container_client.list_blobs():
            blob_client = container_client.get_blob_client(blob.name)
            blob_path = os.path.join(output_folder, blob.name)
            os.makedirs(os.path.dirname(blob_path), exist_ok=True)
            with open(blob_path, "wb") as file:
                data = blob_client.download_blob()
                file.write(data.readall())
            print(f"Downloaded {blob.name} to {blob_path}")


if __name__ == "__main__":

    # Upload repo su blob storage per backup dati
    upload_folder("cloned_data_repo", "cloned_data_repo")

    #Creazione della cartella di output per i blob scaricati, se non esiste già
    output_folder = "downloaded_data"
    os.makedirs(output_folder, exist_ok=True)

    # Download dei blob aggiornati all'ultima versione
    download_blob_container(azure_storage_connection_string=azure_storage_connection_string, container_name=container_name, output_folder=output_folder, blob_name="cloned_data_repo")