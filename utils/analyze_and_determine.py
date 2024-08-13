from typing import List, Dict

import numpy as np
import cv2

def analyze_responses(boxes: List[np.ndarray], questions: int, choices: int) -> np.ndarray:
    """
    Analisa as respostas marcadas nas caixas de múltipla escolha.

    :param boxes: Lista de caixas de múltipla escolha detectadas na imagem. Cada caixa é uma imagem binária
                  onde as áreas preenchidas representam a marcação da escolha.
    :type boxes: list of np.ndarray
    :param questions: Número de perguntas no gabarito.
    :type questions: int
    :param choices: Número de escolhas possíveis para cada pergunta.
    :type choices: int

    :return: Uma matriz onde cada elemento representa o número de pixels não zero (preenchidos) em uma escolha
             específica de cada pergunta.
    :rtype: np.ndarray
    """
    # Inicializa uma matriz para armazenar os valores de pixels
    pixel_values = np.zeros((questions, choices), dtype=np.float32)
    countC = 0  # Contador de colunas (escolhas)
    countR = 0  # Contador de linhas (perguntas)

    # Itera sobre as caixas de múltipla escolha
    for i, box in enumerate(boxes):
        totalPixels = cv2.countNonZero(box)  # Conta o número de pixels preenchidos na caixa
        pixel_values[countR][countC] = totalPixels  # Armazena o valor na matriz
        countC += 1

        # Se o número de escolhas for atingido, avança para a próxima pergunta
        if countC == choices:
            countR += 1
            countC = 0
    return pixel_values


def determine_answers(arr_pixel_values: np.ndarray, questions: int) -> Dict[int, str]:
    """
    Determina as respostas com base nos valores de pixels analisados.

    :param pixel_values: Matriz de valores de pixels retornada por analyze_responses.
    :type pixel_values: np.ndarray
    :param questions: Número de perguntas no gabarito.
    :type questions: int

    :return: Um dicionário onde as chaves são o número da pergunta e os valores são as respostas determinadas
             ("A", "B", "C", etc.) ou "None" se não houver marcação clara.
    :rtype: dict
    """
    real_number_of_questions = []
    new_arr_pixels_values = []
    final_answers = {}
    type_choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    # Itera sobre as perguntas
    for x in range(questions):
        arr = arr_pixel_values[x]  # Obtém os valores de pixels para cada escolha da pergunta
        index_val = np.where(arr == np.amax(arr))[0]  # Encontra o índice da escolha com o maior número de pixels

        # Se a soma dos pixels na pergunta for muito baixa, ignora (provavelmente o local da box não existe uma pergunta para analizar)
        if arr.sum() < 100:
            continue
        else:
            real_number_of_questions.append(index_val[0])  # Armazena o índice da escolha mais marcada
            new_arr_pixels_values.append(arr)  # Armazena os valores de pixels

    # Determina as respostas baseadas nos valores de pixels
    for i in range(len(real_number_of_questions)):
        if new_arr_pixels_values[i][real_number_of_questions[i]] > 400:
            final_answers[i + 1] = type_choices[real_number_of_questions[i]]  # Se o valor for significativo, armazena a resposta
        else:
            final_answers[i + 1] = "None"  # Caso contrário, marca como "None"

    return final_answers
