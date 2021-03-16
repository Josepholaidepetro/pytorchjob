FROM pytorch/pytorch:latest

COPY pytorchjob.py /

ENTRYPOINT ["python", "/pytorchjob.py"]
