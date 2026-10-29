"""Send a few transactions from creditcard.csv to the SQS queue to try the event-driven path.

Usage: python scripts/send_test_events.py --csv creditcard.csv --queue-url <transactions_queue_url> --n 20
"""
import argparse
import json

import boto3
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--csv", required=True)
parser.add_argument("--queue-url", required=True)
parser.add_argument("--n", type=int, default=20, help="Genuine transactions to send (all frauds in the sample are added)")
args = parser.parse_args()

df = pd.read_csv(args.csv)
sample = pd.concat([df[df.Class == 1].head(5), df[df.Class == 0].sample(args.n, random_state=1)])
sqs = boto3.client("sqs")
for i, row in sample.reset_index(drop=True).iterrows():
    body = row.drop("Class").to_dict()
    body["transaction_id"] = f"test-{i}-{'fraud' if row.Class == 1 else 'genuine'}"
    sqs.send_message(QueueUrl=args.queue_url, MessageBody=json.dumps(body))
print(f"Sent {len(sample)} transactions ({int(sample.Class.sum())} known frauds). Check your fraud-alerts email or CloudWatch logs.")
