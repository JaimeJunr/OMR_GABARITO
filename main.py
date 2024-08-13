import json
import os
import cv2
import utils

def main():
    """
    Função principal para processar imagens e PDFs, analisar respostas e salvar os resultados.
    """
    # Configurações
    height_img = 980
    width_img = 750
    questions = 93
    choices = 5
    pdf: bool = False  # Se True, processa PDFs; se False, processa imagens
    answer_sheet_path = "Answers Sheets"
    pdf_path = "pdf"
    temp_img_dir = "temp_images"

    # Criar diretório para armazenar imagens e resultados
    utils.create_directory("Results")
    utils.create_directory(temp_img_dir)

    # Processar arquivos PDF ou imagens, dependendo da configuração
    try:
        if pdf:
            files = utils.process_pdf_files(pdf_path, temp_img_dir)
        else:
            files = utils.process_image_files(answer_sheet_path)
    except Exception as e:
        print(f"Erro ao processar arquivos: {e}")
        return

    for file_name in files:
        file_path = os.path.join(temp_img_dir if pdf else answer_sheet_path , file_name)

        try:
            # Ler imagem
            image = cv2.imread(file_path)
            if image is None:
                print(f"Não foi possível ler a imagem {file_path}.")
                continue

            # Obter ID da página do nome do arquivo
            page_id = utils.extract_id_from_image(image)
            output_dir = os.path.join("Results", page_id)
            utils.create_directory(output_dir)

            # Processar imagem
            adaptive_img = utils.process_image(image, width_img, height_img)
            adaptive_img = utils.resize_image(adaptive_img, width_img, height_img)
            adaptive_img = utils.find_contours(adaptive_img, adaptive_img, width_img, height_img)

            # Extrair e analisar respostas
            boxes = utils.split_boxes(adaptive_img)
            pixel_val = utils.analyze_responses(boxes, questions, choices)

            # Determinar as respostas
            final = utils.determine_answers(pixel_val, questions)

            # Salvar resultados em um JSON
            json_filename = os.path.splitext(file_name)[0] + ".json"
            with open(os.path.join(output_dir, json_filename), "w") as file:
                json.dump(final, file, indent=4)

            # Salvar a imagem processada
            cv2.imwrite(os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.png"), adaptive_img)

        except Exception as e:
            print(f"Erro ao processar o arquivo {file_name}: {e}")

    # Opcional: Limpar diretórios temporários
    # utils.cleanup_directory(temp_img_dir)

if __name__ == "__main__":
    main()
