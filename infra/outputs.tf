output "api_url" {
  description = "Base URL of the fraud scoring API"
  value       = aws_apigatewayv2_api.http.api_endpoint
}

output "ecr_repository_url" {
  description = "Where to push the container image"
  value       = aws_ecr_repository.service.repository_url
}

output "lambda_function_name" {
  value = aws_lambda_function.service.function_name
}
