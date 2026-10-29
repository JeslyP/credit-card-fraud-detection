# ---------- Event-driven scoring: SQS -> Lambda -> SNS fraud alerts ----------
resource "aws_sqs_queue" "dlq" {
  name                      = "${var.project}-transactions-dlq"
  message_retention_seconds = 1209600 # 14 days to investigate failures
}

resource "aws_sqs_queue" "transactions" {
  name                       = "${var.project}-transactions"
  visibility_timeout_seconds = 180 # at least 6x the consumer timeout
  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.dlq.arn
    maxReceiveCount     = 3
  })
}

resource "aws_sns_topic" "fraud_alerts" {
  name = "${var.project}-fraud-alerts"
}

resource "aws_sns_topic_subscription" "fraud_email" {
  count     = var.alarm_email == "" ? 0 : 1
  topic_arn = aws_sns_topic.fraud_alerts.arn
  protocol  = "email"
  endpoint  = var.alarm_email
}

resource "aws_iam_role" "consumer" {
  name               = "${var.project}-consumer"
  assume_role_policy = data.aws_iam_policy_document.assume_lambda.json
}

resource "aws_iam_role_policy_attachment" "consumer_logs" {
  role       = aws_iam_role.consumer.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Least privilege: read this one queue, publish to this one topic
data "aws_iam_policy_document" "consumer" {
  statement {
    actions   = ["sqs:ReceiveMessage", "sqs:DeleteMessage", "sqs:GetQueueAttributes", "sqs:ChangeMessageVisibility"]
    resources = [aws_sqs_queue.transactions.arn]
  }
  statement {
    actions   = ["sns:Publish"]
    resources = [aws_sns_topic.fraud_alerts.arn]
  }
}

resource "aws_iam_role_policy" "consumer" {
  name   = "${var.project}-consumer"
  role   = aws_iam_role.consumer.id
  policy = data.aws_iam_policy_document.consumer.json
}

resource "aws_cloudwatch_log_group" "consumer" {
  name              = "/aws/lambda/${var.project}-consumer"
  retention_in_days = 14
}

resource "aws_lambda_function" "consumer" {
  function_name = "${var.project}-consumer"
  role          = aws_iam_role.consumer.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.service.repository_url}:${var.image_tag}"
  architectures = ["arm64"]
  memory_size   = var.memory_mb
  timeout       = 30

  image_config {
    command = ["fraudservice.queue_handler.handler"]
  }

  environment {
    variables = {
      MODEL_DIR       = "/var/task/models"
      ALERT_TOPIC_ARN = aws_sns_topic.fraud_alerts.arn
    }
  }

  depends_on = [aws_iam_role_policy_attachment.consumer_logs, aws_cloudwatch_log_group.consumer]
}

resource "aws_lambda_event_source_mapping" "transactions" {
  event_source_arn                   = aws_sqs_queue.transactions.arn
  function_name                      = aws_lambda_function.consumer.arn
  batch_size                         = 10
  maximum_batching_window_in_seconds = 5
  function_response_types            = ["ReportBatchItemFailures"]
  depends_on                         = [aws_iam_role_policy.consumer]
}

resource "aws_cloudwatch_metric_alarm" "dlq_not_empty" {
  alarm_name          = "${var.project}-dlq-not-empty"
  alarm_description   = "Transactions failed scoring 3 times and landed in the dead-letter queue"
  namespace           = "AWS/SQS"
  metric_name         = "ApproximateNumberOfMessagesVisible"
  dimensions          = { QueueName = aws_sqs_queue.dlq.name }
  statistic           = "Maximum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

output "transactions_queue_url" {
  description = "Send transactions here to have them scored asynchronously"
  value       = aws_sqs_queue.transactions.url
}

output "fraud_alerts_topic_arn" {
  value = aws_sns_topic.fraud_alerts.arn
}
