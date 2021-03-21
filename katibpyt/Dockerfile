FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime

ADD /katibpyt /opt/katibpyt
WORKDIR /opt/katibpyt

# Add folder for the logs.
RUN mkdir /katib

RUN chgrp -R 0 /opt/katibpytt \
  && chmod -R g+rwX /opt/katibpyt \
  && chgrp -R 0 /katib \
  && chmod -R g+rwX /katib

ENTRYPOINT ["python3", "/opt/katibpyt/mnist.py"]
