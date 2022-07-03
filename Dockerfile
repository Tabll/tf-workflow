
FROM python:3.8.13

ENV VERSION 0.1

WORKDIR /home/app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

ADD . .

CMD ["python", "/home/app/main.py"]
