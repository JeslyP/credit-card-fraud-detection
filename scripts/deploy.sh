#!/usr/bin/env bash
# Build the image, push it to ECR and deploy with Terraform.
# Needs: AWS CLI (logged in), Docker, Terraform, and a trained model in models/.
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f models/model.joblib ] || { echo "No model found. Run: python -m fraudservice.train --data creditcard.csv --out models"; exit 1; }
REGION="${AWS_REGION:-eu-west-2}"
TAG="$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M%S)"

echo "1/4 Creating the container registry"
scripts/state_bucket.sh "$REGION"
terraform -chdir=infra apply -input=false -auto-approve -var "region=$REGION" -target=aws_ecr_repository.service
REPO="$(terraform -chdir=infra output -raw ecr_repository_url)"

echo "2/4 Building the arm64 image"
docker buildx build --platform linux/arm64 --provenance=false -f Dockerfile.lambda -t "$REPO:$TAG" --load .

echo "3/4 Pushing to ECR"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "${REPO%%/*}"
docker push "$REPO:$TAG"

echo "4/4 Deploying Lambda, API Gateway and alarms"
terraform -chdir=infra apply -input=false -auto-approve -var "region=$REGION" -var "image_tag=$TAG"

URL="$(terraform -chdir=infra output -raw api_url)"
echo; echo "Deployed. Health check:"
curl -s "$URL/health"; echo
echo "API: $URL   (tear down with: terraform -chdir=infra destroy -var region=$REGION)"
