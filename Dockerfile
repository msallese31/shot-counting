FROM glavin001/alpine-python2-numpy-scipy

MAINTAINER Mike Sallese <msallese31@gmail.com>

COPY shot_counting/ /usr/src/app/
COPY data/ /usr/src/app/data/
COPY requirements.txt /usr/src/app/
WORKDIR /usr/src/app/

RUN pip install -r /usr/src/app/requirements.txt
RUN pip install gunicorn==19.6.0
RUN pip install flask
RUN pip install pandas

EXPOSE 5000
ENTRYPOINT ["/usr/bin/gunicorn"]

# Start gunicorn
CMD ["-w","1","-b","0.0.0.0:5000","--threads","1","entrypoint:app","--access-logfile","/dev/stdout","--error-logfile","/dev/stdout"]