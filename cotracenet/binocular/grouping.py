"""Keep original-source and resized-cover identities in the same fold."""


def source_groups(sources, covers, indices):
    parents = {}

    def root(value):
        parents.setdefault(value, value)
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    for i in indices:
        parents[root(("source", sources[i]))] = root(("cover", covers[i]))
    groups = {}
    for i in indices:
        groups.setdefault(root(("cover", covers[i])), []).append(i)
    return list(groups.values())
