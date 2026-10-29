terraform {
  required_version = ">= 1.6"

  # State lives in S3 so deploys from GitHub Actions and your laptop share it.
  # The bucket is created by scripts/state_bucket.sh and passed in at init time.
  backend "s3" {}
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = {
      Project   = var.project
      ManagedBy = "terraform"
    }
  }
}
