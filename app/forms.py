from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, SelectField
from wtforms.validators import DataRequired

class LoginForm(FlaskForm):
    # num_forms = 10
    # attributes = [StringField('') for _ in range(num_forms)]
    inputtext = StringField('fff', validators=[DataRequired()])
    # values = [StringField('', validators=[DataRequired()]) for _ in range(num_forms)]
    # password = PasswordField('Password', validators=[DataRequired()])
    # remember_me = BooleanField('Remember Me')
    submit = SubmitField('Submit')
    # select = SelectField('選擇一種指令(以取得pattern)', choices=['aboutMe 關於我', 'request 要求', 'beauty_care 美容', 'choose 選擇', 'else_recommend 推薦其他', 'goodbye 道別', 'greeting 打招呼', 'help_decision 幫助選擇', 'inform 告知', 'noidea 不知道', 'react 使用者回饋', 'reset 清除', 'search_item 尋找物件', 'search_makeup 尋找妝容', 'skinBad 皮膚不好', 'thanks 感謝'])


