#!/usr/bin/env bash

set -eou pipefail

repo_name="sagemaker-custom-images"
image_name="minimal_custom_image"

if [[ $1 == "--profile" ]]; then
    profile=$2
fi

if [[ -z "${profile}" ]]; then
    echo "Please provide a profile name using --profile"
    exit 1
fi;

account=$(aws sts get-caller-identity --query Account --output text --profile $profile)

echo "Your account ID is: $account"

aws ecr get-login-password --region us-east-1 --profile $profile | docker login --username AWS --password-stdin "${account}.dkr.ecr.us-east-1.amazonaws.com"

docker buildx build --platform linux/amd64 -t ${image_name}.latest .

docker tag ${image_name}.latest "${account}.dkr.ecr.us-east-1.amazonaws.com/${repo_name}:${image_name}"

docker push "${account}.dkr.ecr.us-east-1.amazonaws.com/${repo_name}:${image_name}"