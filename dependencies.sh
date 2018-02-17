echo 'Getting annotations'
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip
echo 'Getting word embeddings'
wget http://nlpserver2.inf.ufrgs.br/alexandres/vectors/lexvec.enwiki%2bnewscrawl.300d.W.pos.vectors.gz
unzip lexvec.enwiki+newscrawl.300d.W.pos.vectors.gz
echo 'Installing coco api'
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
make install
python setup.py install
cd ../.././
ls
echo 'Installing python libraries'
pip install -r requirements.txt
echo 'Installed all dependencies'

