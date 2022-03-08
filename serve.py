from flask import Flask, request
from flask_json import FlaskJSON, JsonError, as_json
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline, AutoConfig
import json

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
APP_ROOT = "./"
app.config["APPLICATION_ROOT"] = APP_ROOT
app.config["UPLOAD_FOLDER"] = "files/"

json_app = FlaskJSON(app)

# Downloading model from huggingface
tokenizer = AutoTokenizer.from_pretrained("dvilares/bertinho-gl-base-cased")
model = AutoModelForMaskedLM.from_pretrained("dvilares/bertinho-gl-base-cased")
config = AutoConfig.from_pretrained("dvilares/bertinho-gl-base-cased")

fill_mask = pipeline(
    "fill-mask",
    model=model,
    tokenizer=tokenizer
)


def predict_text(content):
    text_predict = fill_mask(content)

    return prepare_output_format(text_predict)


def prepare_output_format(predict):
    list_options = list()
    for result in predict:
        list_options.append({"content": result['sequence'], "score": result['score']})

    return {"response": {"type": "texts", "texts": list_options}}


@as_json
@app.route("/predict_bertinho", methods=["POST"])
def predict_bertinho():
    data = request.get_json()
    if (data.get('type') != 'text') or ('content' not in data):
        output = invalid_request_error(None)
        return output

    try:
        output = fill_mask(data["content"])  # json with the response
    except Exception as e:
        return invalid_request_error(e)
    return (prepare_output_format(output))


@json_app.invalid_json_error
def invalid_request_error(e):
    """Generates a valid ELG "failure" response if the request cannot be parsed"""
    if (e == None):
        raise JsonError(status_=400, failure={'errors': [
            {'code': 'elg.request.invalid', 'text': 'Invalid request message'}
        ]})
    error = {}
    error["code"] = "elg.service.internalError"
    error["detail"] = {"message":str(e)}

    raise JsonError(status_=404, failure={'errors': [error]})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8866)
