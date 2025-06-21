"""
File handling utilities
"""

import os
import shutil
import tempfile
from typing import List, Optional


def ensure_directory(directory: str) -> None:
    """
    Ensure a directory exists.

    Args:
        directory: Directory path
    """
    os.makedirs(directory, exist_ok=True)


def get_file_extension(file_path: str) -> str:
    """
    Get the extension of a file.

    Args:
        file_path: Path to the file

    Returns:
        File extension (lowercase)
    """
    _, ext = os.path.splitext(file_path)
    return ext.lower()


def is_image_file(file_path: str) -> bool:
    """
    Check if a file is an image based on extension.

    Args:
        file_path: Path to the file

    Returns:
        True if the file is an image
    """
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
    return get_file_extension(file_path) in image_extensions


def is_video_file(file_path: str) -> bool:
    """
    Check if a file is a video based on extension.

    Args:
        file_path: Path to the file

    Returns:
        True if the file is a video
    """
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    return get_file_extension(file_path) in video_extensions


def create_temp_copy(file_path: str, delete: bool = True) -> str:
    """
    Create a temporary copy of a file.

    Args:
        file_path: Path to the original file
        delete: Whether to delete the temp file when Python exits

    Returns:
        Path to the temporary copy
    """
    # Create temp file with same extension
    ext = get_file_extension(file_path)
    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    temp_path = temp_file.name
    temp_file.close()

    # Copy file
    shutil.copy2(file_path, temp_path)

    # Register for deletion if requested
    if delete:
        import atexit

        atexit.register(
            lambda: os.remove(temp_path) if os.path.exists(temp_path) else None
        )

    return temp_path


def get_files_in_directory(
    directory: str, extensions: Optional[List[str]] = None
) -> List[str]:
    """
    Get files in a directory with optional extension filtering.

    Args:
        directory: Directory path
        extensions: List of extensions to filter by

    Returns:
        List of file paths
    """
    if not os.path.isdir(directory):
        return []

    files = []

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if os.path.isfile(file_path):
            if extensions is None:
                files.append(file_path)
            else:
                file_ext = get_file_extension(file_path)
                if file_ext in extensions:
                    files.append(file_path)

    return files
