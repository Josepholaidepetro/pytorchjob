FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime

ADD mnist.py /

ENTRYPOINT ["python3", "/mnist.py"]
