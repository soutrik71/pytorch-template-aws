**Install docker and docker-compose on Ubuntu 22.04**
__PreRequisites__:

    * Have an aws account with a user that has the necessary permissions
    * Have the access key either on env variables or in the github actions secrets
    * Have an ec2 runner instance running/created in the aws account
    * Have a s3 bucket created in the aws account
    * Have aws container registry created in the aws account 
__Local VM setup__:
    * Install aws configure and setup the access key and secret key and the right zone
        ```bash
        curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
        unzip awscliv2.zip
        sudo ./aws/install   
        aws configure
        ```
    

__Install docker__:
```bash
sudo apt update
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
sudo systemctl restart docker
sudo reboot
docker --version
docker ps
```
__Install docker-compose__:
```bash
sudo rm /usr/local/bin/docker-compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.30.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
docker-compose --version
```

__Github actions self-hosted runner__:
```bash
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-2.320.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.320.0/actions-runner-linux-x64-2.320.0.tar.gz
echo "93ac1b7ce743ee85b5d386f5c1787385ef07b3d7c728ff66ce0d3813d5f46900  actions-runner-linux-x64-2.320.0.tar.gz" | shasum -a 256 -c
tar xzf ./actions-runner-linux-x64-2.320.0.tar.gz
./config.sh --url https://github.com/soutrik71/pytorch-template-aws --token <Latest>
# cd actions-runner/
./run.sh
# https://github.com/soutrik71/pytorch-template-aws/settings/actions/runners/new?arch=x64&os=linux
```
__Activate aws cli__:
```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
sudo apt install unzip
unzip awscliv2.zip
sudo ./aws/install
aws --version
aws configure

```
__S3 bucket operations__:
```bash
aws s3 cp data s3://deep-bucket-s3/data --recursive
aws s3 ls s3://deep-bucket-s3
aws s3 rm s3://deep-bucket-s3/data --recursive
```
