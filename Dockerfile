FROM python:3.12.4-slim

WORKDIR /usr/src/app

COPY . /usr/src/app

RUN apt-get update && apt-get install -y gcc

RUN pip install --no-cache-dir -U pip && pip install setuptools wheel && cd LLaMA-Factory && pip install -e ".[torch,metrics]" --index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install -r requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple

CMD ["tail", "-f", "/dev/null"]