import re
import pytesseract
import cv2
import numpy as np

def extract_id_from_image(image: np.ndarray) -> str:
    """
    Extrai o ID de uma imagem usando OCR (pytesseract) e uma expressão regular.

    :param image: Imagem a ser processada para extração do ID.
    :type image: np.ndarray
    :return: O ID extraído como uma string. Se o ID não for encontrado, retorna "unknown_id".
    :rtype: str
    """
    try:
        # Realizar OCR na imagem
        text = pytesseract.image_to_string(image)

        # Expressão regular para procurar pelo ID (pode incluir letras, números e hífens)
        match = re.search(r"ID:\s*([A-Za-z0-9\-]+)", text)

        if match:
            return match.group(1)

    except pytesseract.TesseractError as e:
        print(f"Erro do Tesseract ao processar a imagem: {e}")
    except Exception as e:
        print(f"Erro inesperado ao extrair ID: {e}")

    # Retorna um ID desconhecido se não for encontrado nenhum ID
    return "unknown_id"
