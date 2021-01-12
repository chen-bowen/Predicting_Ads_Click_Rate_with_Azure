export KAGGLE_KEY="c198f5adc1595ac8d1de06ea8070d81c"
export KAGGLE_USERNAME="bowenchen184"

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