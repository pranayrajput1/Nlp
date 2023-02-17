"""Constant values"""

# docker image name
# TODO: to be updated
IMAGE_QUALIFIER = ''

# tag of docker image
IMAGE_TAG = '0.0.0'


# TODO: Update the Project id
PROJ_ID = ''

# Docker Image
# TODO: update the location of container registry
BASE_IMAGE = f'gcr.io/{PROJ_ID}/forms_classifier_{IMAGE_QUALIFIER}:{IMAGE_TAG}'

# Name of the pipeline
PIPELINE_NAME = 'resume-screening-train'

# Small description of the pipeline
PIPELINE_DESCRIPTION = "Resume Screening model training pipeline"

# TODO: Update the GCS location of the kubeflow pipeline
PIPELINE_ROOT_GCS = "gs://d-ulti-ml-ds-dev-9561-kubeflowpipelines-default"

# TODO: Update the pipeline service account
PIPELINE_SERVICE_ACCOUNT = ""

# TODO: Update the location/region
LOCATION = ""

# Cloud Storage
# TODO: updated the bucket name where data will be stored
BUCKET_NAME = ''

# Big Query
# TODO: update the dataset id if required
DATASET_ID = ''
