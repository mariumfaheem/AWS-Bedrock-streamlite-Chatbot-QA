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
    "inputText": prompt_data ,
     "textGenerationConfig":
         {"maxTokenCount":3072,
          "temperature":0.5,
          "topP":0.9,
        "stopSequences":[]
          }
}
body=json.dumps(payload)


model_id="amazon.titan-text-premier-v1:0"
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
repsonse_text=response_body['results'][0]['outputText']
print(repsonse_text)



