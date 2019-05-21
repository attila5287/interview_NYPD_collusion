import numpy as np
import matplotlib.path as mpath

# from generate_color4_plot import (
#     generateRandomDark_c0lor()
#     generateRandomLight_c0lor()
#     generateRandomMatplotlib_c0lor()
#     )

def generateRandomDark_c0lor():
    """
    GENERATES A RANDOM COLOR FROM PRE-SELECTED COLORS 
    THAT GO ALONG WITH THE DEFAULT THEME 538
    """
    c0lorList = [
        'firebrick',
        'sandybrown',
        'olivedrab',
        'seagreen',
        'darkcyan',
        'mediumvioletred',
        'coral',
        'darkgoldenrod',
        'olive',
        'cadetblue',
        'crimson',
        'indianred',
        'peru',
        'goldenrod',
        'lightslategray',
        'mediumorchid',
        'tomato',
        'orchid',
        'darkmagenta',
        'dimgrey',
    ]
    rand1nt = np.random.randint(0, len(c0lorList))
    c0lor = c0lorList[rand1nt]
    return c0lor

def generateRandomLight_c0lor():
    """
    GENERATES A RANDOM COLOR FROM PRE-SELECTED COLORS 
    THAT GO ALONG WITH THE DEFAULT THEME 538
    """
    c0lorList = [
        'silver',
        'bisque',
        'moccasin',
        'floralwhite',
        'lightgoldenrodyellow',
        'paleturquoise',
        'aliceblue',
        'plum',
        'mistyrose',
        'peachpuff',
        'lemonchiffon',
        'ghostwhite',
        'blanchedalmond',
        'beige',
        'gainsboro',
        'linen',
        'antiquewhite',
        'thistle',
        'mintcream',
        'lavenderblush'
    ]

    rand1nt = np.random.randint(0, len(c0lorList))
    c0lor = c0lorList[rand1nt]
    return c0lor

def generateRandomMatplotlib_c0lor():
    """
    GENERATES A RANDOM COLOR FROM PRE-SELECTED COLORS 
    THAT GO ALONG WITH THE DEFAULT THEME 538
    """
    light_c0lorList = [
        'silver',
        'bisque',
        'moccasin',
        'floralwhite',
        'lightgoldenrodyellow',
        'paleturquoise',
        'aliceblue',
        'plum',
        'mistyrose',
        'peachpuff',
        'lemonchiffon',
        'ghostwhite',
        'blanchedalmond',
        'beige',
        'gainsboro',
        'linen',
        'antiquewhite',
        'thistle',
        'mintcream',
        'lavenderblush'
    ]
    dark_c0lorList = [
        'firebrick',
        'sandybrown',
        'olivedrab',
        'seagreen',
        'darkcyan',
        'mediumvioletred',
        'coral',
        'darkgoldenrod',
        'olive',
        'cadetblue',
        'crimson',
        'indianred',
        'peru',
        'goldenrod',
        'lightslategray',
        'mediumorchid',
        'tomato',
        'orchid',
        'darkmagenta',
        'dimgrey',
    ]    
    c0lorList = []
    for lightcolor, darkcolor in zip(light_c0lorList,dark_c0lorList):
        c0lorList.append(lightcolor)
        c0lorList.append(darkcolor)
    rand1nt = np.random.randint(0, len(c0lorList))
    c0lor = c0lorList[rand1nt]
    return c0lor


def cut_st4r(n):
    """
    this is a code of a negative-spaced-star-shaped SVG path that will
    mark rounds help player  see important date points easily
    """
    star = mpath.Path.unit_regular_star(n)
    circle = mpath.Path.unit_circle()
    # concatenate the circle with an internal cutout of the star
    verts = np.concatenate([circle.vertices, star.vertices[::-1, ...]])
    codes = np.concatenate([circle.codes, star.codes])
    cut_star = mpath.Path(verts, codes)
    return cut_star


