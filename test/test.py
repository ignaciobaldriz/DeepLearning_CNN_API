import requests 

'''
1- Execute the "main.py" file
2- In a new Terminal window, go to the "test" file (venv activated)
3- Type: python test.py
4- The prediction would be done, change the file name 'one.png' to make a new prediction using the other images in the folder.
   You can also import an image to the folder and test the prediction.

'''

# https://your-heroku-app-name.herokuapp.com/predict
# http://localhost:5000/predict


resp = requests.post("http://localhost:5000/predict", files={'file': open('cero.png', 'rb')})

print(resp.text)