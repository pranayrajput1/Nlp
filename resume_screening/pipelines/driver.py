""" Vertex AI Pipeline """

from google.cloud import aiplatform
from kfp import dsl
from kfp.v2 import compiler

from resume_screening.utils.constant_config import (
    LOCATION,
    PIPELINE_DESCRIPTION,
    PIPELINE_NAME,
    PIPELINE_ROOT_GCS,
    PIPELINE_SERVICE_ACCOUNT,
    PROJ_ID,
)


def single_resume_screen_pipeline():
    """
    This function is used as a wrapper for all the steps for training and evaludating the model performances
    """


@dsl.pipeline(
    name=PIPELINE_NAME,
    description=PIPELINE_DESCRIPTION,
    pipeline_root=PIPELINE_ROOT_GCS,
)
def pipeline():
    """
    In this function we are calling the wrapper function for model creation
    """
    single_resume_screen_pipeline()


if __name__ == "__main__":
    pipeline_json_fname = './pipeline.json'
    compiler.Compiler().compile(pipeline_func=pipeline, package_path=pipeline_json_fname)

    job = aiplatform.PipelineJob(
        display_name=PIPELINE_NAME,
        template_path=pipeline_json_fname,
        project=PROJ_ID,
        location=LOCATION,
        enable_caching=False,  # May want to set to False when doing dev work on pipeline
    )
    job.submit(service_account=PIPELINE_SERVICE_ACCOUNT)
