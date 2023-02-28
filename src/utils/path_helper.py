import sys
from pathlib import Path


def set_project_root_path() -> None:
    """
    Add project root folder to system environment variables (os agnostic)
    """

    file_path = Path(__file__).resolve()
    src_path = file_path.parents[1]
    project_root = file_path.parents[2]

    try:
        sys.path.index(str(project_root))
    except ValueError:
        sys.path.append(str(project_root))

    try:
        sys.path.index(str(src_path))
    except ValueError:
        sys.path.append(str(src_path))

    try:
        sys.path.index(project_root.as_posix())
    except ValueError:
        sys.path.append(project_root.as_posix())


if __name__ == '__main__':
    set_project_root_path()
