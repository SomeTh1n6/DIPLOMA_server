from transformers import TFAutoModel, AutoTokenizer
import tensorflow as tf

class BERTForClassification(tf.keras.Model):
    def __init__(self, bert_model, num_classes):
        super().__init__()
        self.bert = bert_model
        self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.bert(inputs)[1]
        return self.fc(x)

    def get_config(self):
        config = {
                'bert_model': self.bert,
                'num_classes': self.fc.units
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

import warnings

def start():
    # 0 - saddness , 1 - angry, 2 - fear , 3 - surprise
    # Отключение предупреждений от TensorFlow
    warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

    model = TFAutoModel.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # Создание экземпляра модели классификатора
    new_classifier = BERTForClassification(model, num_classes=4)
    new_classifier.load_weights('bert_emotion_classifier')
    return new_classifier, tokenizer

def get_classification(new_classifier, tokenizer, input_text: str):
    tokenized_input = tokenizer(input_text, padding=True, truncation=True, return_tensors="tf")

    # Вызов метода call для получения предсказания
    predicted_probabilities = new_classifier.call(tokenized_input)

    # Преобразование предсказанных вероятностей в метки классов
    predicted_labels = predicted_probabilities


    return predicted_labels
