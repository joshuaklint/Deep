def find_index(container,search):
    for i, target in enumerate(container):
        if target == search:
            return i, target
