FROM python:3.7.7-buster
WORKDIR /usr/src/app

COPY requirements.txt ./

RUN apt update
RUN apt install -y gfortran
RUN apt install -y libhdf5-dev
RUN pip install --no-cache-dir -r requirements.txt
ENV PATH="/opt/gtk/bin:$PATH"
COPY . .

CMD [ "python", "-u", "./MTE.py" ]
# CMD ["python", "./debug.py" ]
