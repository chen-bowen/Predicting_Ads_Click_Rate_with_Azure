export KAGGLE_KEY=""
export KAGGLE_USERNAME=""

pip install kaggle

#download dataset from kaggle
kaggle datasets download -d louischen7/2020-digix-advertisement-ctr-prediction

# unzip data
unzip 2020-digix-advertisement-ctr-prediction.zip

# remove extra files
rm 2020-digix-advertisement-ctr-prediction.zip

rm test_data_A.csv
rm test_data_B.csv

# rename train_data to data
mv train_data/ data/