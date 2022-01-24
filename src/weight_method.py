def bin_Total(stat):
    if stat<580:
        return 0
    elif stat<680:
        return 1
    else:
        return 2

def bin_HP(stat):
    if stat<80:
        return 0
    elif stat<101:
        return 1
    else:
        return 2

def bin_ATK(stat):
    if stat<91:
        return 0
    elif stat<101:
        return 1
    else:
        return 2

def bin_DEF(stat):
    if stat<91:
        return 0
    elif stat<101:
        return 1
    else:
        return 2

def bin_SP_ATK(stat):
    if stat<91:
        return 0
    elif stat<106:
        return 1
    else:
        return 2

def bin_SP_DEF(stat):
    if stat<91:
        return 0
    elif stat<106:
        return 1
    else:
        return 2

def bin_SPE(stat):
    if stat<91:
        return 0
    elif stat<101:
        return 1
    else:
        return 2

def get_class(pokemon, lim):
    dict_fct = {
        'Total':(bin_Total, 0.39073476/2),
        'HP':(bin_HP, 0.08393827/2),
        'Attack':(bin_ATK, 0.04904614/2),
        'Defense':(bin_DEF, 0.06680186/2),
        'Sp. Atk':(bin_SP_ATK, 0.18185804/2),
        'Sp. Def':(bin_SP_DEF, 0.08710185/2),
        'Speed':(bin_SPE, 0.14051906/2)
    }
    score = sum([dict_fct[feature][0](stat)*dict_fct[feature][1] for feature,stat in pokemon.items()])
    if score > lim:
        return True
    else:
        return False