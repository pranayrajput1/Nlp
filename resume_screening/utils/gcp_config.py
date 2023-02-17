"""
All the values we need to access the Cloud Storage and BigQuery services.
"""
import logging
import os

from google.cloud.exceptions import GoogleCloudError
from google.oauth2 import service_account
from google.cloud import storage
from google.cloud import bigquery
from google.auth import compute_engine
from resume_screening.utils.constant_config import PROJ_ID, BUCKET_NAME, DATASET_ID


def get_credentials():
    """Get the key path to the JSON file if we're running locally
    :return: service account credentials
    """

    try:
        creds = _get_credentials_from_file()
        if creds is not None:
            return creds
        else:  # Maybe we're running on a GCP VM?
            return compute_engine.Credentials()
    except GoogleCloudError as e:
        message = f'Unable to retrieve service account_credentials. The error was {e}.'
        logging.error(message)


def _get_credentials_from_file():
    path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", None)
    if path is None:  # try legacy name for this env var
        path = os.getenv("SERVICE_ACCOUNT_PATH_DEV", None)

    if path is None:
        return None

    return service_account.Credentials.from_service_account_file(
        path, scopes=["https://www.googleapis.com/auth/cloud-platform"])


def get_bigquery_config():
    """Create a dict of BigQuery configurations by building
    them from component names.

    :return: dict of BigQuery configurations
    """
    return {
        'bq_client': init_client('bigquery'),
        'dataset_id': f'{PROJ_ID}.{DATASET_ID}',
    }


def init_client(service: str = 'storage'):
    if service == 'storage':
        client_class = storage.Client
    elif service == 'bigquery':
        client_class = bigquery.Client
    else:
        raise ValueError(f"Unrecognized or unsupported GCP service: {service}")
    creds = get_credentials()
    return client_class(project=PROJ_ID, credentials=creds)


def get_cloud_storage_config():
    """Create a dict of Cloud Storage configurations by building
    them from component names.

    :return: dict of Cloud Storage configurations
    """
    return {
        'project_id': PROJ_ID,
        'cs_client': init_client('storage'),
        'bucket_name': BUCKET_NAME,
    }
