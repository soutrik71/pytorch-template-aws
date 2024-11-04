## __PreRequisites__:

    * Have an aws account with a user that has the necessary permissions
    * Have the access key either on env variables or in the github actions secrets
    * Have an ec2 runner instance running/created in the aws account
    * Have a s3 bucket created in the aws account
    * Have aws container registry created in the aws account 
  
## __Local VM setup__:
    * Install aws configure and setup the access key and secret key and the right zone
        ```bash
        curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
        unzip awscliv2.zip
        sudo ./aws/install   
        aws configure
        ```
    
