import io
import numpy as np
from flask import Flask, render_template, request
from tensorflow import keras

ann_model = keras.models.load_model('models/mobile_receipt.h5')

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.route("/")
def index_page():
    return render_template('index.html')

@app.post("/prediction")
def predict_inputs():
    inputs = []
    prediction = 'n/a'

    input_file = request.files.get('input_file', None)
    has_input_file = input_file is not None and input_file.content_type == 'text/csv'

    if has_input_file:
        input_file = request.files.get('input_file')
        input_file.stream.seek(0)
        input_data = io.StringIO(input_file.stream.read().decode("UTF8"))
    else:
        data = request.form
        inputs = list(map(lambda name: [name, data.get(name)], feature_labels.keys()))

    raw_inputs = list(map(lambda val: val[1], inputs))
    inputs = list(map(lambda val: feature_labels_indices[val[0]][val[1]], inputs))

    classifier = request.form.get('classifier')
    final_input = np.expand_dims(np.array(inputs), axis=0)
    raw_prediction = ann_model.predict(inputs)

    return {
        'prediction': prediction
    }

if __name__ == "__main__":
    app.run()
