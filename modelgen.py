from keras.models import model_from_json
from keras.optimizers import Adam

# Load model architecture from JSON file
with open('Stress_Model.json', 'r') as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)

# Load weights into the model
model.load_weights('Stress_Model.h5')  

# Compile the model with a new optimizer
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

emotion_classifier = model
emotion_classifier.save("_mini_XCEPTION_updated.hdf5")
