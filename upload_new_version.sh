rm -rf bdpn.egg-info build dist
python setup.py sdist bdist_wheel
twine upload dist/* && \
sudo docker build -t evolbioinfo/bdpn:v0.1.11 -f Dockerfile . && sudo docker login && sudo docker push evolbioinfo/bdpn:v0.1.11