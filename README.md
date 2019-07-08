# Flower_tflite
Recognize Flowers using Transfer Learning

Retraining a classifier trained on Imagenet Dataset using Tensorflow 2.0 to detect the flower species 

1. Setup Input Pipeline
  download the dataset 
  
  https://colab.research.google.com/github/tensorflow/examples/blob/master/community/en/flowers_tf_lite.ipynb#scrollTo=xxL2mjVVGIrV
  
2. Rescale the Images

https://colab.research.google.com/github/tensorflow/examples/blob/master/community/en/flowers_tf_lite.ipynb#scrollTo=aCLb_yV5JfF3&line=4&uniqifier=1

3. Create the base model from the pre-trained convnets
 
 https://colab.research.google.com/github/tensorflow/examples/blob/master/community/en/flowers_tf_lite.ipynb#scrollTo=19IQ2gqneqmS
 
4. Feature Extraction

  base_model.trainable = False
  
 4.1 add classification head
  
  https://colab.research.google.com/github/tensorflow/examples/blob/master/community/en/flowers_tf_lite.ipynb#scrollTo=eApvroIyn1K0&line=3&uniqifier=1
  
 4.2 compile model
 
 https://colab.research.google.com/github/tensorflow/examples/blob/master/community/en/flowers_tf_lite.ipynb#scrollTo=RpR8HdyMhukJ
 
 4.3 train model
 
 https://colab.research.google.com/github/tensorflow/examples/blob/master/community/en/flowers_tf_lite.ipynb#scrollTo=JsaRFlZ9B6WK
 
 4.4 learning curves
 
 ![download](https://user-images.githubusercontent.com/26092430/60843559-a42cc380-a1f4-11e9-91d6-c9ce144ca248.png)

 https://colab.research.google.com/github/tensorflow/examples/blob/master/community/en/flowers_tf_lite.ipynb#scrollTo=53OTCh3jnbwV&line=2&uniqifier=1

5. fine Tuning 

In our feature extraction experiment, you were only training a few layers on top of an MobileNet V2 base model. The weights of the pre-trained network were not updated during training.

One way to increase performance even further is to train (or "fine-tune") the weights of the top layers of the pre-trained model alongside the training of the classifier you added. The training process will force the weights to be tuned from generic features maps to features associated specifically to our dataset.
 
6. Un-freeze the top layers of the model
All you need to do is unfreeze the base_model and set the bottom layers be un-trainable. Then, recompile the model (necessary for these changes to take effect), and resume training.

7. compile model

https://colab.research.google.com/github/tensorflow/examples/blob/master/community/en/flowers_tf_lite.ipynb#scrollTo=NtUnaz0WUDva&line=3&uniqifier=1

![download (1)](https://user-images.githubusercontent.com/26092430/60843708-0a194b00-a1f5-11e9-904a-e87121152346.png)

Convert to TFLite
Saved the model using tf.saved_model.save and then convert the saved model to a tf lite compatible format.

https://colab.research.google.com/github/tensorflow/examples/blob/master/community/en/flowers_tf_lite.ipynb#scrollTo=_LZiKVInWNGy&line=4&uniqifier=1

Using a pre-trained model for feature extraction: When working with a small dataset, it is common to take advantage of features learned by a model trained on a larger dataset in the same domain. This is done by instantiating the pre-trained model and adding a fully-connected classifier on top. The pre-trained model is "frozen" and only the weights of the classifier get updated during training. In this case, the convolutional base extracted all the features associated with each image and you just trained a classifier that determines the image class given that set of extracted features.

Fine-tuning a pre-trained model: To further improve performance, one might want to repurpose the top-level layers of the pre-trained models to the new dataset via fine-tuning. In this case, you tuned your weights such that your model learned high-level features specific to the dataset. This technique is usually recommended when the training dataset is large and very similar to the orginial dataset that the pre-trained model was trained on.

 
 My blog is here...!
 
 Part 1: 
 
 https://medium.com/@ajinkyajawale/recognize-flowers-using-transfer-learning-26c2188c50ba
 
 Part 2:
 
 https://medium.com/@ajinkyajawale/recognize-flowers-using-transfer-learning-80f7e086612b
 
 
 
