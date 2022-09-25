# Regression_general_model
This project is developed in view of generalization.
Many of times we need to check the performance of different models to select best performing model.
The project takes formatted csv /excel file and provides model performance of major regression ML models

The pre requisites are as below.
1. The file should be in format csv/excel
2. The dependant data (the result to be predicted) should be the last column in file

There are two APIs to accomplish this.
1. The first api takes file as input.
2. The second api calculates model metrics and provides as result.


BUILD DOCKER IMAGE
```
docker build -t <image_name>:<tagname> .
```
> Note: Image name for docker must be lowercase


To list docker image
```
docker images
```

Run docker image
```
docker run -p 5000:5000 -e PORT=5000 94d4d21daca6 .
```

To check running container in docker
```
docker ps
```

Tos stop docker conatiner
```
docker stop <container_id>
```
sudo docker build -t "regression:latest" .
docker images
docker run -p 5000:5000 -e port=5000 77cda0b7d124
