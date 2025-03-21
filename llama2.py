import boto3
import json
import os

from dotenv import load_dotenv
load_dotenv()

prompt_data="""
Act as a Shakespeare and write a poem on Genertaive AI
"""

bedrock=boto3.client(service_name="bedrock-runtime",
                     region_name="us-east-1",
                     aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                     aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
                    aws_session_token=os.environ['AWS_SESSION_TOKEN'])



payload={
    "prompt": prompt_data ,
    "max_gen_len":512,
    "temperature":0.5,
    "top_p":0.9
}
body=json.dumps(payload)


model_id="meta.llama3-8b-instruct-v1:0"
accept = "application/json"
contentType = "application/json"

#chain.invoke (llm | inputs )
response=bedrock.invoke_model(
 modelId = model_id,
 contentType= contentType,
 accept = accept,
 body = body
)
response_body=json.loads(response.get("body").read())
repsonse_text=response_body['generation']
print(repsonse_text)


