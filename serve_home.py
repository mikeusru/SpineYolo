from flask import Flask, request, render_template
from PIL import Image
from io import BytesIO

# User defined utility functions
from spine_yolo import SpineYolo
from form import ReusableForm

# Home page
app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def home():
    """Home page of app with form"""

    # Create form
    form = ReusableForm(request.form)

    # On form entry and all conditions met
    if request.method == 'POST' and form.validate():
        # Extract information
        scale = int(request.form['scale'])
        # Generate results
        sp = SpineYolo()
        sp.set_model_path('model_data/yolov3_spines_combined.h5')
        sp.detect(get_image())
        r_image = sp.r_images[0]
        return render_template('spines_found.html', r_image=r_image)

    # Send templates information to index.html
    return render_template('index.html', form=form)

def get_image():
    response = request.get('https://www.maxplanckflorida.org/wp-content/uploads/2018/07/Figure-press-release-01-1-300x297.jpg')
    img = Image.open(BytesIO(response.content))
    return img

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8888)
