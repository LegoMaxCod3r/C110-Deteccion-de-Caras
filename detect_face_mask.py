# Importar la biblioteca OpenCV
import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("keras_model.h5")


# Definir un objeto de captura de video
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capturar el video fotograma por fotograma
    check, frame = vid.read()
    img = cv2.resize(frame, (224,224))

    #Convertir la imagen en una matriz "numpy" e incrementar su dimensión
    test_image = np.array(img, dtype=np.float32)
    test_image = np.expand_dims(test_image, axis=0)
    #Normalisar la imagen entre los pixeles
    normalised_image = test_image/255

    #Predecir el resultado basado en la imagen divida en 255
    prediction = model.predict(normalised_image)
    print(prediction)

    

    # Mostrar el fotograma resultante
    cv2.imshow('Resultado', frame)
      
    # Salir de la ventana con la barra espaciadora
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  
# Después del bucle, liberar al objeto de captura
vid.release()

# Destruir todas las ventanas
cv2.destroyAllWindows()


