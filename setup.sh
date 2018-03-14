# Create directories for datasets
# mkdir data data/mnist data/catdogs

# Fetch MNIST dataset directly
#wget http://deeplearning.net/data/mnist/mnist.pkl.gz
#gunzip -c *.gz > data/mnist/mnist.pkl

# Process Cats & Dogs dataset 
# Assumes train.zip was downloaded from Kaggle into root directory
# unzip train.zip -d data/catdog
python3 minify_cat_dog.py --savedir data/catdog

# Move images in respective folders
# mkdir data/catdog/train_64x64/train data/catdog/train_64x64/valid
# mkdir data/catdog/train_64x64/cat data/catdog/train_64x64/dog
# find data/catdog/train_64x64/ -name "cat.*" -exec mv {} data/catdog/train_64x64/cat \;
# find data/catdog/train_64x64/ -name "dog.*" -exec mv {} data/catdog/train_64x64/dog \;

# Clean
#rm *.gz *.zip
