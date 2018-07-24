echo 'Getting annotations'
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip
echo 'Installing python libraries'
pip install -r requirements.txt
echo 'Installing coco api'
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
make install
python3 setup.py install
cd ../.././
echo 'Installed all dependencies'


