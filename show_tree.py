import os

def list_directory_structure(path, max_depth=3, current_depth=1):
    if current_depth > max_depth:
        return []
    result = []
    try:
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_dir():
                    result.append((current_depth, entry.name + '/'))
                    result.extend(list_directory_structure(os.path.join(path, entry.name), max_depth, current_depth + 1))
                else:
                    result.append((current_depth, entry.name))
    except PermissionError:
        pass
    return result

structure = list_directory_structure(os.getcwd())

for depth, name in structure:
    print('  ' * (depth - 1) + name)

