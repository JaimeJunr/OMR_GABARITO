import cv2
import numpy as np

def rect_contours(contours: list) -> list:
    """
    Filtra os contornos para retornar apenas aqueles que são aproximadamente retangulares
    e têm uma área maior que 1000.

    :param contours: Lista de contornos encontrados na imagem.
    :type contours: list[np.ndarray]
    :return: Lista de contornos que são retangulares e têm uma área maior que 1000.
    :rtype: list[np.ndarray]
    """
    rect_cont = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.01 * peri, True)
            if len(approx) == 4:  # Se o contorno tem 4 vértices, é um retângulo
                rect_cont.append(contour)
    return rect_cont


def find_contours(adaptive_img: np.ndarray, output: np.ndarray, width_img: int, height_img: int) -> np.ndarray:
    """
    Encontra e desenha contornos retangulares na imagem, e aplica uma transformação de perspectiva
    para corrigir a perspectiva do maior retângulo encontrado.

    :param adaptive_img: Imagem binarizada após processamento.
    :type adaptive_img: np.ndarray
    :param output: Imagem original onde os contornos serão desenhados.
    :type output: np.ndarray
    :param width_img: Largura da imagem de saída.
    :type width_img: int
    :param height_img: Altura da imagem de saída.
    :type height_img: int
    :return: Imagem com contornos desenhados e perspectiva corrigida.
    :rtype: np.ndarray
    """
    contours, _ = cv2.findContours(adaptive_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Encontrar retângulos
    rect_cont = rect_contours(contours)

    if rect_cont:
        biggest_contour = get_corner_points(rect_cont[0])

        if biggest_contour.size != 0:
            cv2.drawContours(output, [biggest_contour], -1, (0, 255, 0), 2)
            biggest_contour = reorder_corner_points(biggest_contour)
            pt1 = np.float32(biggest_contour)
            pt2 = np.float32([[0, 0], [width_img, 0], [0, height_img], [width_img, height_img]])
            matrix = cv2.getPerspectiveTransform(pt1, pt2)
            output = cv2.warpPerspective(output, matrix, (width_img, height_img))

    return output


def get_corner_points(contour: np.ndarray) -> np.ndarray:
    """
    Obtém os pontos de canto de um contorno usando aproximação poligonal.

    :param contour: Contorno de interesse.
    :type contour: np.ndarray
    :return: Pontos de canto aproximados do contorno.
    :rtype: np.ndarray
    """
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.01 * peri, True)

    return approx


def reorder_corner_points(points: np.ndarray) -> np.ndarray:
    """
    Reordena os pontos de canto de acordo com a posição em relação aos quatro cantos
    de um retângulo (top-left, top-right, bottom-left, bottom-right).

    :param points: Pontos de canto a serem reorganizados.
    :type points: np.ndarray
    :return: Pontos de canto reordenados.
    :rtype: np.ndarray
    """
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 1, 2), np.int32)
    add = points.sum(axis=1)
    diff = np.diff(points, axis=1)

    new_points[0] = points[np.argmin(add)]  # [0, 0] - canto superior esquerdo
    new_points[1] = points[np.argmin(diff)]  # [w, 0] - canto superior direito
    new_points[2] = points[np.argmax(diff)]  # [0, h] - canto inferior esquerdo
    new_points[3] = points[np.argmax(add)]  # [w, h] - canto inferior direito

    return new_points
