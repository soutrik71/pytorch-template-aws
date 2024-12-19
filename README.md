## Serverless Deployment with AWS Lambda and API Gateway
### create a docker image and run
```bash
docker build -t fastapi-aws-catdog:latest .
docker run --name fastapi-aws-catdog-container -p 8000:8000 --env-file .env fastapi-aws-catdog:latest
docker stop fastapi-aws-catdog-container
docker rm -f fastapi-aws-catdog-container
docker run -it --rm -p 8000:8000 --env-file .env fastapi-aws-catdog:latest bash # to run bash in the container to check the env variables
```
### deploy to AWS Lambda and cdk
```bash
cdk bootstrap -v
cdk deploy -v --logs
cdk destroy -vf
```