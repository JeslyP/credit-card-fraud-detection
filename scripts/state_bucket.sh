#!/usr/bin/env bash
# Create (if needed) a private, versioned, encrypted S3 bucket for Terraform state,
# then run terraform init against it. Usage: scripts/state_bucket.sh <region>
set -euo pipefail
REGION="${1:-eu-west-2}"
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
BUCKET="fraud-scoring-tfstate-${ACCOUNT}-${REGION}"

if ! aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  echo "Creating Terraform state bucket $BUCKET"
  if [ "$REGION" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "$BUCKET" --region "$REGION"
  else
    aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" --create-bucket-configuration LocationConstraint="$REGION"
  fi
  aws s3api put-bucket-versioning --bucket "$BUCKET" --versioning-configuration Status=Enabled
  aws s3api put-bucket-encryption --bucket "$BUCKET" --server-side-encryption-configuration '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
  aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
fi

terraform -chdir=infra init -input=false -reconfigure \
  -backend-config="bucket=$BUCKET" -backend-config="key=fraud-scoring/terraform.tfstate" \
  -backend-config="region=$REGION" -backend-config="encrypt=true"
