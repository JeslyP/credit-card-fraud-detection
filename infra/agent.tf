# ---------- LLM fraud analyst agent (Claude) ----------
# The Anthropic API key lives in an encrypted SSM SecureString parameter that the deploy workflow writes from a
# GitHub secret. It is never stored in Terraform state, the container image or Lambda environment variables.
data "aws_caller_identity" "current" {}

locals {
  anthropic_key_param = "/${var.project}/anthropic-api-key"
}

data "aws_iam_policy_document" "read_anthropic_key" {
  statement {
    actions   = ["ssm:GetParameter"]
    resources = ["arn:aws:ssm:${var.region}:${data.aws_caller_identity.current.account_id}:parameter${local.anthropic_key_param}"]
  }
}

resource "aws_iam_role_policy" "read_anthropic_key" {
  name   = "read-anthropic-key"
  role   = aws_iam_role.lambda.id
  policy = data.aws_iam_policy_document.read_anthropic_key.json
}
