#install cocoapi

cd ../
git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
cd cocoapi/PythonAPI
make
python setup.py install --user
cd ../../

#install requirements
pip install -r requirements.txt


#complie DCNv2
cd lib/models/networks/DCNv2
./make.sh

