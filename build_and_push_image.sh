if [ -z "$2" ]; then
  echo "usage: $0 <pipeline_qualifier> <version_tag>"
  echo "example: $0 csm 0.0.3rc4"
  exit 1
fi

PIPELINE_QUALIFIER="$1"
VERSION_TAG="$2"
IMAGE="gcr.io/{project_id}/{repo_name}_${PIPELINE_QUALIFIER}:${VERSION_TAG}"

echo "Building and pushing image: ${IMAGE}"
docker build -t "$IMAGE" .
docker push "$IMAGE"