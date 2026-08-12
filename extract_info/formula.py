import base64

import cv2
import numpy as np

FORMULA_MODEL = "PP-FormulaNet_plus-M"


class FormulaReader:
    """Reconhece fórmulas em imagens via PaddleX, devolvendo LaTeX delimitado.

    O modelo é criado na primeira leitura: quem só importa o pacote (a API, por
    exemplo) não paga o carregamento do PaddleX.
    """

    def __init__(self, modelName: str = FORMULA_MODEL):
        self.modelName = modelName
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from paddlex import create_model

            self._model = create_model(self.modelName)
        return self._model

    def read(self, imageBase64: str) -> str:
        imgBytes = base64.b64decode(imageBase64)
        buf = np.frombuffer(imgBytes, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        text = ""
        for res in self.model.predict(input=img, batch_size=1):
            text += res["rec_formula"]

        return "$$" + text + "$$ \n"
