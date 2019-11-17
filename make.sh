#install coco api

git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
cd cocoapi/PythonAPI
make
python setup.py install --user
cd ../../

#install requirements
pip install -r requirements.txt


#install DCNv2
cd src/lib/models/networks/DCNv2
./make.sh
