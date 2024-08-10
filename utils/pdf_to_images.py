import os
import fitz  # PyMuPDF
import numpy as np
import cv2
from typing import List

import utils
from utils.directory_utils import create_directory


def process_pdf_files(pdf_path: str, temp_img_dir: str) -> List[str]:
    """
    Processa arquivos PDF e converte cada página em uma imagem PNG.

    :param pdf_path: Caminho para o diretório contendo arquivos PDF (str).
    :param temp_img_dir: Caminho para o diretório onde as imagens temporárias serão salvas (str).
    :return: Lista de nomes de arquivos de imagens geradas (List[str]).
    """
    files = []
    pdf_files = [f for f in os.listdir(pdf_path) if f.lower().endswith('.pdf')]

    # Cria o diretório temporário para armazenar imagens se não existir
    create_directory(temp_img_dir)

    for pdf_file in pdf_files:
        pdf_file_path = os.path.join(pdf_path, pdf_file)
        images = pdf_to_images(pdf_file_path)

        for idx, image in enumerate(images):
            image_filename = f"{os.path.splitext(pdf_file)[0]}_page_{idx + 1}.png"
            image_path = os.path.join(temp_img_dir, image_filename)
            cv2.imwrite(image_path, image)
            files.append(image_filename)

    return files


def process_image_files(gabaritos_path: str) -> List[str]:
    """
    Processa arquivos de imagem e retorna uma lista de arquivos PNG, JPG e JPEG.

    :param gabaritos_path: Caminho para o diretório contendo arquivos de imagem (str).
    :return: Lista de nomes de arquivos de imagens (List[str]).
    """
    return [f for f in os.listdir(gabaritos_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]


def pdf_to_images(pdf_path: str) -> List[np.ndarray]:
    """
    Converte as páginas de um PDF em imagens e retorna uma lista dessas imagens.

    :param pdf_path: Caminho para o arquivo PDF a ser convertido (str).
    :return: Lista de imagens convertidas em formato numpy.ndarray (List[np.ndarray]).
    """
    # Abre o PDF usando PyMuPDF
    doc = fitz.open(pdf_path)
    images = []

    # Itera por todas as páginas do PDF
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()

        # Converte os dados da imagem em um array numpy
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

        # Converte imagem de RGBA para RGB se necessário
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        # Adiciona a imagem à lista
        images.append(img)

    # Cria o diretório "images" para armazenar as imagens se não existir
    create_directory("images")

    # Salva cada imagem em um arquivo PNG
    for i, img in enumerate(images):
        output_path = f"images/page_{i + 1}.png"
        cv2.imwrite(output_path, img)

    return images
