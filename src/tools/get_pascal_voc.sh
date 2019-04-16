mkdir voc
cd voc
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar
tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
tar xvf VOCdevkit_08-Jun-2007.tar
tar xvf VOCtrainval_11-May-2012.tar
tar xvf VOCdevkit_18-May-2011.tar
rm VOCtrainval_06-Nov-2007.tar
rm VOCtest_06-Nov-2007.tar
rm VOCdevkit_08-Jun-2007.tar
rm VOCtrainval_11-May-2012.tar
rm VOCdevkit_18-May-2011.tar
mkdir images
cp VOCdevkit/VOC2007/JPEGImages/* images/
cp VOCdevkit/VOC2012/JPEGImages/* images/
wget https://storage.googleapis.com/coco-dataset/external/PASCAL_VOC.zip
unzip PASCAL_VOC.zip
rm PASCAL_VOC.zip
mv PASCAL_VOC annotations/
cd ..
python merge_pascal_json.py
