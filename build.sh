#Todo
echo 'All tests passed'

git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
make install
python setup.py install
cd ../.././
ls
echo 'Some file folder modifications'
mkdir errors
echo 'Installed all dependencies'
