# Create directories for datasets
mkdir data data/mnist data/catdogs

# Fetch MNIST dataset directly
wget http://deeplearning.net/data/mnist/mnist.pkl.gz
gunzip -c *.gz > data/mnist/mnist.pkl

# Process Cats & Dogs dataset 
# Assumes train.zip was downloaded from Kaggle into root directory
unzip train.zip -d data/catdog
python3 minify_cat_dog.py --savedir data/catdog

# Clean
rm *.gz *.zip
