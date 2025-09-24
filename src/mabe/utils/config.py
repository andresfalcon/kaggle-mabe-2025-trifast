import tomllib
from pathlib import Path


class Cfg(dict):
    """
    Configuración cargada desde un TOML.
    Se comporta como un diccionario, pero mantiene acceso por atributos.
    """

    def __init__(self, data: dict):
        super().__init__(data)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def to_dict(self) -> dict:
        """Convierte el objeto en un dict normal."""
        return dict(self)


def load_config(path: str) -> Cfg:
    """
    Carga un archivo TOML y devuelve un objeto Cfg.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró config en {path}")

    with open(path, "rb") as f:
        data = tomllib.load(f)

    return Cfg(data)
