### A CNN cat-dog classifier. Since the last update it:
* reaches 85% accuracy on the validation set.
(can be found in the log file above)

"
Epoch 26/26
8000/8000 [==============================] - 1863s - loss: 0.3312 - acc: 0.8598 - val_loss: 0.4031 - val_acc: 0.8500
"

* new convolution layer is added
* new hidden layer is added
* dropout rates are increased
* there is a bit which saves the weights of the model

### In the next release expect:
* Add an option to load a photo and let the classifier actually classify it.
This will be new as for now all the model is doing is testing and validation.
* Load the model's weights. 
