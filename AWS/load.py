import tensorflow as tf
import urllib.request as request
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# load a saved model
from tensorflow.keras.models import load_model
 


def compare_images(image_database, image_front):
    '''
    Function that receives two images routes, one from the user storaged in the database, and the other is received from the front.
    Then, the two images are compared with the model generated by the CNN.
    It returns a prediction from 0 to 1, the more alike are the two images the closer to 1 the response will be.
    How to use:
    from load import compare_images

    prediction = compare_images(db_route, frontend_route)
    '''
    # load model
    MODEL = load_model('prediction-model.h5',compile=True)
    # Receiving and transforming the database image
    image_database = request.urlretrieve(image_database, 'genuine.jpg')
    image_database = Image.open(image_database[0])
    image_database = image_database.convert('RGB')
    image_database = image_database.resize((150,150))
    image_database = img_to_array(image_database)
    image_database = tf.expand_dims(image_database,0)

    # Receiving and transforming the frontend image
    image_front = request.urlretrieve(image_front, 'test.jpg')
    image_front = Image.open(image_front[0])
    image_front = image_front.convert('RGB')
    image_front = image_front.resize((150,150))
    image_front = img_to_array(image_front)
    image_front = tf.expand_dims(image_front,0)

    # Returning model predicted.
    return MODEL.predict([image_database,image_front])