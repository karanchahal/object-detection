echo 'Installing python libraries'
pip install -r requirements.txt
echo 'Getting annotations'
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
echo 'Installing coco api'
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make all
echo 'Installed all dependencies'

