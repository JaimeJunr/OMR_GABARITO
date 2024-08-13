import os

def create_directory(path: str) -> None:
    """
    Cria um diretório no caminho especificado se ele não existir.

    Se o diretório já existir, a função não faz nada. Se ocorrer um erro ao tentar criar
    o diretório, uma exceção será lançada com uma mensagem informativa.

    :param path: Caminho do diretório a ser criado.
    :type path: str
    :raises OSError: Se ocorrer um erro ao criar o diretório.
    """
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        raise OSError(f"Erro ao criar o diretório '{path}': {e.strerror}") from e
