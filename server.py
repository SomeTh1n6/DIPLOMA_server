import uvicorn
from fastapi import FastAPI

import logging
from pydantic import BaseModel

import bert_model
from translator import translate_text_with_translators
import gigachat

logging.basicConfig(filename='server.log', level=logging.DEBUG)
app = FastAPI()

new_classifier, tokenizer = bert_model.start()
class Item(BaseModel):
    text: str

@app.post("/echo/")
async def echo(item: Item):
    # return {"echo": "Классификация: " + item.text}
    translated_text = translate_text_with_translators(item.text, source_language='ru', target_language='en')
    classification = bert_model.get_classification(new_classifier, tokenizer, translated_text)

    print(classification)
    giga_token = gigachat.get_giga_token()
    return {"echo": gigachat.get_response(giga_token, item.text, classification) + "\n\n Но обратите внимание. Данные советы носят ИСКЛЮЧИТЕЛЬНО рекомендательный характер"}
# uvicorn server:app --reload

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)