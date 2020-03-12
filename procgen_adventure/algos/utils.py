def get_values_from_list_dict(epinfobuf, key):
    list_values = [epinfo[key] for epinfo in epinfobuf if key in epinfo]
    return list_values
