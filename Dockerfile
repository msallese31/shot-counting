FROM revamp_wheel  

ADD requirements.txt /app/requirements.txt  
WORKDIR /app

RUN while read p; \  
    do /env/bin/pip install --no-index -f /wheelhouse $p; \
    done < requirements.txt
ADD . /app  

COPY shot_counting/ /app
COPY data/ /app/data/

EXPOSE 5000
ENTRYPOINT ["/env/bin/gunicorn"]

# Start gunicorn
CMD ["-w","1","-b","0.0.0.0:5000","--threads","1","entrypoint:app", "--log-level=info", "--log-file=/dev/stdout", "--access-logfile","/dev/stdout","--error-logfile","/dev/stdout"]