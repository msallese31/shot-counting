#!/usr/bin/env bash


set -euo pipefail

# TODO: Add flag to login
# docker login --username=shotcounterapp --email=shotcounterapp@gmail.com

wheel='false'

while getopts 'w' flag; do
  case "${flag}" in
    w) wheel='true' ;;
    *) error "Unexpected option ${flag}" ;;
  esac
done

if [ "$wheel" = true ]; then
    docker build --no-cache -t "revamp_wheel" -f Dockerfile.build .
fi

docker build -t "shot-counter-backend" .

docker tag shot-counter-backend shotcounterapp/shot-counter-backend
docker push shotcounterapp/shot-counter-backend
