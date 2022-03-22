FROM python:3.9

RUN apt update && apt install -y libopencv-dev && rm -rf /var/lib/apt/lists/*

COPY . /repo

WORKDIR /repo

RUN pip install -r requirements.txt && pip install notebook

RUN jupyter nbconvert --to script  /repo/code/dashboard_LIMEcraft.ipynb  --output /repo/code/script

CMD ["/bin/bash", "-c", "cd /repo/code && python script.py"]
