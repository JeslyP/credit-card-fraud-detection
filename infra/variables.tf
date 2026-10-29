variable "project" {
  description = "Name prefix for every resource"
  type        = string
  default     = "fraud-scoring"
}

variable "region" {
  description = "AWS region to deploy to"
  type        = string
  default     = "eu-west-2" # London
}

variable "image_tag" {
  description = "Tag of the container image in ECR to deploy"
  type        = string
  default     = "latest"
}

variable "memory_mb" {
  description = "Lambda memory; more memory also means more CPU"
  type        = number
  default     = 1024
}

variable "throttle_rate" {
  description = "Steady-state requests per second allowed through API Gateway (keeps a public demo cheap)"
  type        = number
  default     = 5
}

variable "throttle_burst" {
  description = "Short burst of requests allowed through API Gateway"
  type        = number
  default     = 10
}

variable "alarm_email" {
  description = "Optional email to notify when the error alarm fires (leave empty to skip)"
  type        = string
  default     = ""
}
