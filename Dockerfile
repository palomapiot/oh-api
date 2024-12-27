FROM nvidia/cuda:12.5.0-devel-ubuntu22.04

WORKDIR /app

RUN apt update && apt install -y python3-pip git && apt clean
RUN pip3 install --upgrade pip

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
RUN pip install --no-cache-dir "unsloth[cu121-ampere-torch240] @ git+https://github.com/unslothai/unsloth.git"


COPY oh-api/oh_api.py /app/oh_api.py

EXPOSE 8000

CMD ["uvicorn", "oh_api:app", "--host", "0.0.0.0", "--port", "8000"]
