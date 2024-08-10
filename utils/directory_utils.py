import os


def create_directory(path: str) -> None:
    """
    Cria um diretório no caminho especificado se ele não existir.

    Se o diretório já existir, a função não faz nada. Caso ocorra um erro ao tentar criar
    o diretório, uma exceção será lançada.

    :param path: Caminho do diretório a ser criado.
    :type path: str
    :return: None
    :rtype: None
    :raises OSError: Se ocorrer um erro ao criar o diretório.
    """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError as e:
        print(f"Erro ao criar o diretório {path}: {e}")
        raise
