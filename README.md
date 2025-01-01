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
### Deploy the code on minikube with fastapi and Fasthtml UI with kubernetes
```bash
# test locally
uvicorn main:app --host 0.0.0.0 --port 8000
# build the docker image
docker build -t catdog-classifier -f./Dockerfile . --no-cache
docker run -it --rm -p 8000:8000 catdog-classifier:latest

# deploy to minikube
eval $(minikube docker-env )
# build the image
docker build -t catdog-classifier -f./Dockerfile . --no-cache
docker ps -a # check the container
eval $(minikube docker-env -u) # exit from minikube docker env
# apply the deployment and service
kubectl apply -f catdog-classifier.yaml
kubectl get all # to get all the resources and check the deployment and service
minikube service catdog-classifer-service # Expose the Service using MiniKube Service Proxy
minikube tunnel # to create a tunnel to the minikube service and this has to be run in a separate terminal
kubectl port-forward service/catdog-classifer-service 8080:80 # forward the service to the localhost:8080 and 192.168.49.2 catdog-classifer.localhost 127.0.0.1 catdog-classifer.localhost has to be added to the /etc/hosts file
# test the code in the browser using localhost:8080 of windows or curl -v catdog-classifer.localhost from host machine terminal