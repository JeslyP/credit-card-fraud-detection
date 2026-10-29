"""AWS Lambda entry point: runs the same FastAPI app behind API Gateway using Mangum."""
from mangum import Mangum

from .api import create_app

handler = Mangum(create_app(), lifespan="off")
