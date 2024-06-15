FROM python:3.9-slim

RUN mkdir /pasteur

# Install bdpn
RUN cd /usr/local/ && pip3 install --no-cache-dir bdpn==0.1.5

# The entrypoint runs bdpn_infer with command line arguments
ENTRYPOINT ["bdpn_infer"]