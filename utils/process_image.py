import cv2
import numpy as np
import utils

def resize_image(image: np.ndarray, width_img: int, height_img: int) -> np.ndarray:
    """
    Redimensiona a imagem para as dimensões fornecidas.

    :param image: Imagem a ser redimensionada (np.ndarray).
    :param width_img: Largura desejada da imagem (int).
    :param height_img: Altura desejada da imagem (int).
    :return: Imagem redimensionada (np.ndarray).
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("A imagem deve ser um numpy ndarray.")
    return cv2.resize(image, (width_img, height_img))

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Converte a imagem para escala de cinza.

    :param image: Imagem a ser convertida (np.ndarray).
    :return: Imagem em escala de cinza (np.ndarray).
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("A imagem deve ser um numpy ndarray.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def threshold_img(image: np.ndarray) -> np.ndarray:
    """
    Aplica um threshold para binarização da imagem.

    :param image: Imagem em escala de cinza (np.ndarray).
    :return: Imagem binarizada (np.ndarray).
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("A imagem deve ser um numpy ndarray.")
    _, thresh_img = cv2.threshold(image, 170, 255, cv2.THRESH_BINARY_INV)
    return thresh_img

def remove_noise(image: np.ndarray) -> np.ndarray:
    """
    Remove o ruído da imagem usando filtro de difusão não local.

    :param image: Imagem a ser processada (np.ndarray).
    :return: Imagem sem ruído (np.ndarray).
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("A imagem deve ser um numpy ndarray.")
    return cv2.fastNlMeansDenoising(image, None, 30, 7, 21)

def split_boxes(image: np.ndarray, top_removal_height=40, bottom_removal_height=10,
                start_slice=20, end_slice=170, shift_x=50, shift_y=0) -> list:
    """
    Divide a imagem em caixas, removendo partes desnecessárias e dividindo em colunas e linhas.

    :param image: Imagem a ser dividida (np.ndarray).
    :param top_removal_height: Altura a ser removida do topo da imagem (int).
    :param bottom_removal_height: Altura a ser removida da parte inferior da imagem (int).
    :param start_slice: Ponto inicial de fatia horizontal (int).
    :param end_slice: Ponto final de fatia horizontal (int).
    :param shift_x: Deslocamento horizontal para remoção de laterais (int).
    :param shift_y: Deslocamento vertical para remoção de laterais (int).
    :return: Lista de caixas extraídas (list).
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("A imagem deve ser um numpy ndarray.")

    height, width = image.shape[:2]

    # Remover parte superior e inferior
    image = image[top_removal_height:height - bottom_removal_height, :]

    # Dividir imagem em colunas
    columns = np.hsplit(image, 3)
    boxes = []
    for i, column in enumerate(columns):
        # Dividir imagem em linhas
        rows = np.vsplit(column, 31)
        for j, row in enumerate(rows):
            row_height, row_width = row.shape[:2]

            # Garantir que os índices de fatia estejam no intervalo válido
            start_slice = max(0, start_slice)
            end_slice = min(row_width, end_slice)

            # Remover laterais da linha
            row = row[shift_y:row_height - shift_y, start_slice + shift_x:end_slice + shift_x]

            # Dividir caixas na coluna
            cols = np.hsplit(row, 5)
            for t, box in enumerate(cols):
                boxes.append(box)
                # Salvar as caixas apenas se for necessário para depuração
                # cv2.imwrite(f"images/box_{i}_{j}_{t}.png", box)

    return boxes

def process_image(image: np.ndarray, width_img: int, height_img: int) -> np.ndarray:
    """
    Pipeline de processamento da imagem, convertendo para escala de cinza, aplicando threshold adaptativo e remoção de ruído.

    :param image: Imagem a ser processada (np.ndarray).
    :param width_img: Largura desejada da imagem (int).
    :param height_img: Altura desejada da imagem (int).
    :return: Imagem processada (np.ndarray).
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("A imagem deve ser um numpy ndarray.")

    resized_img = resize_image(image, width_img, height_img)
    processed_img = preprocess_image(resized_img)
    adaptive_img = threshold_img(processed_img)
    denoised_img = remove_noise(adaptive_img)

    return denoised_img
