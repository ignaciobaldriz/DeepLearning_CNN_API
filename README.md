# DeepLearning_CNN_API

Project in process:

Consists of a Deep Learning rest API with Convolutional Neural Networks in Pythorch using the MNIST data set.

1. In the "model.py" file there is trained and generated the model. It is serialized and saved as "mnist_cnn_pht" inside the app folder

2. In the app folder it is included: 
- "main.py": the Flask app code
- "mnist_cnn.pht": the model serialized and saved
- "torch_utils.py": utils to run the model in the app
     
3. There is a data folder that includes only the test data (train is too large for the repository). The code to download both train and test are included in the "model.py" file

4. Extra files:
- "notes.txt": info and links with background and the original project
- "Procfile": for hosting in Heroku
- "reuirements.txt": libraries versions for venv
- "runtime.txt": python version
- "wsgi.py": to run the app.main in Heroku
