FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime

ADD mnist.py /
WORKDIR /

# Add folder for the logs.
RUN mkdir /katib

RUN chgrp -R 0 / \
  && chmod -R g+rwX / \
  && chgrp -R 0 /katib \
  && chmod -R g+rwX /katib

ENTRYPOINT ["python3", "/mnist.py"]
