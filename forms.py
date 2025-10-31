from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import DataRequired

class FileUploadForm(FlaskForm):
    dataset=FileField('Upload CSV or Excel File', validators=[DataRequired()])
    submit=SubmitField("Upload & Analysis")