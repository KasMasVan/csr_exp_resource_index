mkdir data/winogrande/
cd data/winogrande/
wget https://storage.googleapis.com/ai2-mosaic/public/winogrande/winogrande_1.1.zip
unzip winogrande_1.1.zip
cp winogrande_1.1/{dev,test,train_xs}* .
