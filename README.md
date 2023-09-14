# imageprediction
This project is an example of building an image classification model using TensorFlow and Keras. It demonstrates how to create a simple convolutional neural network (CNN) to classify images of buses and trains.

Description:
This project uses TensorFlow and Keras to build and train a CNN model for image classification. 
The model is trained to classify images into two categories: buses and trains. It's a simple example that can serve as a starting point for more complex image classification tasks.

Prerequisites

Before you begin, make sure you have the following dependencies installed:

    TensorFlow
    OpenCV (cv2)
    NumPy
    Matplotlib


deep learning:

Deep learning is like teaching a computer to learn and make decisions by itself, kind of like how you teach a pet new tricks, but with a lot of data and math.

Imagine you have a dog, and you want it to learn how to fetch a ball. At first, you show the dog the ball and say, "Fetch!" The dog might not get it right the first time, but you keep trying. 
Each time the dog gets closer to the ball, you give it a treat. Over time, the dog learns to fetch the ball better and better.

Deep learning is a bit like that, but instead of a dog and a ball, you have a computer and lots of data. You give the computer a bunch of data, like pictures of cats and dogs, and you tell it, "This is a cat, and that's a dog." 
The computer then tries to figure out on its own what makes a cat different from a dog. It learns from its mistakes and gets better at telling cats and dogs apart.



Training the Model

To train the image classification model, follow these steps:

Prepare your dataset: Organize your training and validation images into separate directories. 
The datasets are in a folder should be in a single format most preferably in .jpg and .png, those data are labeled by a single name and a single key value should differ from each other.
all the image passed in the model should have same resolution.

Respectively you should have three folders
one for training your model
2nd one for validate the model
3rd folder should have data to test your model

Neural network will takes only numerical values strings and other formats are not allowed in the neural network
Where in this project there is only two types of labled data that is bus and train where bus labeled data are taken as '0' and train labeled data considered as '1' so we used binary
if you have multiple images like car,bike,train you have to choose categorical crossentropy you have to choose this according to your classes.
here in this case only two classes are there that is bus and train so we took binary crossentropy

num_classes = 1
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPool2D(2, 2),

This is the first layer of our cnn model let us see what the key words explain.
f.keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=(200, 200, 3)):

    tf.keras.layers.Conv2D: This layer represents a 2D convolutional layer, which is used to extract features from input images.
    16: This parameter specifies the number of filters (also known as kernels) in the convolutional layer. It determines how many different features the layer will look for.
    (3, 3): These numbers define the size of the filter or kernel. In this case, it's a 3x3 filter.
    activation="relu": This sets the activation function for the layer to the Rectified Linear Unit (ReLU), which introduces non-linearity into the network.we should increase the increase the filters in each of the layers to improve the accurancy
    
    
 tf.keras.layers.Flatten(): This layer flattens the output from the previous layers into a one-dimensional vector. It's necessary before connecting to fully connected layers.   input_shape=(200, 200, 3): This specifies the shape of the input data. In this case, it's an image with a height and width of 200 pixels and 3 color channels (red, green, and blue).



