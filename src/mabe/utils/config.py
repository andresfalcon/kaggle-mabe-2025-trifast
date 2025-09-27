import tomllib
from pathlib import Path

class Cfg(dict):
    """
    Configuración cargada desde un TOML.
    Se comporta como un diccionario, pero mantiene acceso por atributos,
    incluso de manera recursiva.
    """

    def __init__(self, data: dict):
        super().__init__()
        for k, v in data.items():
            if isinstance(v, dict):
                v = Cfg(v)   # conversión recursiva
            elif isinstance(v, list):
                v = [Cfg(x) if isinstance(x, dict) else x for x in v]
            self[k] = v

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def to_dict(self) -> dict:
        """Convierte el objeto en un dict normal."""
        return {k: (v.to_dict() if isinstance(v, Cfg) else v) for k, v in self.items()}


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
