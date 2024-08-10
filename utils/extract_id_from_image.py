import re
import pytesseract
import cv2

def extract_id_from_image(image) -> str:
    """
    Extrai o ID de uma imagem usando OCR e uma expressão regular.

    :param image: Imagem a ser processada (np.ndarray).
    :type image: np.ndarray
    :return: O ID extraído como uma string. Se o ID não for encontrado, retorna "unknown_id".
    :rtype: str
    """
    try:
        # Realizar OCR na imagem
        text = pytesseract.image_to_string(image)

        # Procurar pelo ID usando uma expressão regular
        match = re.search(r"ID:\s*(\w+)", text)

        if match:
            # Retorna o ID encontrado
            return match.group(1)
    except Exception as e:
        # Imprime o erro caso ocorra uma exceção
        print(f"Erro ao extrair ID: {e}")

    # Retorna um ID desconhecido se não for encontrado nenhum ID
    return "unknown_id"
