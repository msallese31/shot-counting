#!/usr/bin/env bash

# TODO: Add flag to login
# docker login --username=shotcounterapp --email=shotcounterapp@gmail.com

docker build -t "shot-counter-backend" .

docker tag shot-counter-backend shotcounterapp/shot-counter-backend
docker push shotcounterapp/shot-counter-backend
