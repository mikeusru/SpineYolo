from wtforms import (Form, TextField, validators, SubmitField,
DecimalField, IntegerField)

class ReusableForm(Form):

    scale = IntegerField('Enter scale in pixels per um:',
                         default=15, validators=[validators.InputRequired(),
                                                 validators.NumberRange(min=1, max=1000,
                                                 message='Scale must be between 11 and 1000')])
    # Submit button
    submit = SubmitField("Enter")