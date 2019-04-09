from flask import render_template, request

# Home page
from form import ReusableForm


@app.route("/", methods=['GET', 'POST'])
def home():
    """Home page of app with form"""
    # Create form
    form = ReusableForm(request.form)

    # Send template information to index.html
    return render_template('index.html', form=form)