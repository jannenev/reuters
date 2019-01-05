# reuters

This project tests different features and parameters for CNN and LSTM and their effects on news article classification.
Glove is used as basic embedding.

Task is multi-label assigning to news articles of Reuters-data. About 300.000 articles in XML-form.


The code produces large files, especilly reuters_all.pkl of 435,8 MB, and its splits to train/dev/test. These are added to .gitignore to NOT be uploaded to github. Instead they can be produced by the code locally.

## Results

### Small amount of training data
3426 samples.


On small data (3426 samples) CNN  based models converge robustly very fast, on couple epochs, while LSTM based models  took a long time, a bit over 50 epochs to start showing any sign convergence. On small data CNN was already reached its point of overfitting, before LSTM started to converge.  The small data was not enough for LSTM. Possibly because with 126 classes, the number of some individual classes remains small. (we did not balance the training set, but used random sample ). The same was true for the LSTM+CNN architecture.

Training time on GTX 1070 GPU / 8GB.

![alt text](https://github.com/jannenev/reuters/blob/master/images/compare_models_small_data.png)




### Full training data
299773 samples. 
On full data CNN is still superior in training time. Here though LSTM can match in accuracy. Test-accuracy was quite equal, also having a bit fluctuation with different runs. CNN+LSTM achieved a quite similar accuracy (0.837) to the best convolutional model,


![alt text](https://github.com/jannenev/reuters/blob/master/images/compare_models_full_data_8x5.png)

There are no big differences in accuracy of models. Also accuracy of single model varies a bit between training with different random initialization.

![alt text](https://github.com/jannenev/reuters/blob/master/images/compare_models_full_data_bar_8x5.png)
