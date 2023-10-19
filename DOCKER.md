# Docker Images

> NOTE: This file should be removed before handover to the customer.

## Repository

Internal Amazon tools flag the PyChunkedGraph Docker image as insecure because it is pulled from public Dockerhub. To
work around this, the team has created a [private ECR Repository](https://tiny.amazon.com/pbc2d4mw/IsenLink) for use
during the prototype engagement.

Note that the repository has been set up
with [tag immutability](https://docs.aws.amazon.com/AmazonECR/latest/userguide/image-tag-mutability.html), so new images
will require an updated tag.

## Using the Image

To authenticate with the registry, use the AWS CLI and profile for the allen-connectome account to login to local docker
client:

```
aws ecr get-login-password --region us-west-2 --profile <YOUR ALLEN AWS PROFILE> | docker login --username AWS --password-stdin 727518754019.dkr.ecr.us-west-2.amazonaws.com/pychunkedgraph
```

Then your normal docker push/pull commands will work like normal:

```
docker pull 727518754019.dkr.ecr.us-west-2.amazonaws.com/pychunkedgraph:graph-tool_dracopy
```

More information on using private ECR repositories can be
found [here](https://docs.aws.amazon.com/AmazonECR/latest/userguide/getting-started-cli.html).
