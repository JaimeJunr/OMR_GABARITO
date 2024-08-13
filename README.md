# Projeto de Processamento de Imagens e PDFs
#### Descrição
Este projeto realiza o processamento de imagens e arquivos PDF para extrair e analisar informações. O objetivo principal é converter PDFs em imagens, processar imagens para identificar contornos e caixas, e salvar os resultados em formatos JSON e PNG.

## Funcionalidades

Conversão de PDFs para Imagens: Extrai páginas de PDFs e salva como imagens PNG.
Processamento de Imagens: Redimensiona, converte para escala de cinza, aplica binarização e remove ruído.
Extração de IDs: Usa OCR para extrair IDs de imagens.
Detecção de Contornos: Identifica e desenha contornos retangulares em imagens.
Divisão de Imagens em Caixas: Divide a imagem em caixas para análise detalhada.


#### Pré-requisitos


Certifique-se de ter os seguintes pré-requisitos instalados:
Python 3.7 ou superior

```bash
   Python 3.7 ou superior
```

#### Dependências do projeto
opencv-python, numpy, pytesseract, PyMuPDF



#### Você pode instalar as dependências usando pip:

```bash
   pip install opencv-python numpy pytesseract PyMuPDF
```

#### Além disso, você precisará do Tesseract OCR instalado no seu sistema!

# Uso

#### Processamento de Imagens
Para processar imagens diretamente, certifique-se de que o diretório gabaritos contém as imagens desejadas. Execute o script main.py com o seguinte comando:
```bash
   python main.py
```

