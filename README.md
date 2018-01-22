# shot-counting

This repository holds the code for the Shot Counter Android App's counting server.  The counting server is written in python using the flask server with gunicorn.  

# Current Development Process

1. Write new code (on branch)
2. Review merge code
3. Build/Package into docker image:

```
docker build -t "shot-counter" .
```

# Running the Docker Image

```
docker run -p 12345:5000 shot-counter
```

# Making a request against the server

```
curl localhost:12345/health
```

# Evaluating Performance

```
ab -n 10000 -c 100 localhost:12345/
```

# Running outside of docker

#### Startup 

```
source setupFlask.sh
flask run
```

#### Make Request

```
curl localhost:5000/health
```
