import numpy as np
import matplotlib.path as mpath

# from generate_color4_plot import (
#     generateRandomDark_c0lor,
#     generateRandomLight_c0lor,
#     generateRandomMatplotlib_c0lor,
#     markerGenerator4plot,
#     cut_st4r
    # 
#     )

def makeItRandomStyle():
    """
    GENERATES RANDOM STYLE, RETURNS NONE
    """
    style_list = list(plt.style.available)
    rand_style_int = np.random.randint(0, len(style_list))
    random_styl3 = style_list[rand_style_int]
    plt.style.use(random_styl3)
    print(random_styl3)
    return random_styl3

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

def markerGenerator4plot():
    """
    CREATES RANDOM MARKERS FOR MATPLOTLIB, DATA VISUALIZATION
    """
    pass

    mark3r = [
        ".",
        "o",
        "v",
        "^",
        "<",
        ">",
        "8",
        "s",
        "p",
        "P",
        "*",
        "h",
        "H",
        "X",
        "D",
        "d"
    ]

    mark3r_desc = [
        "point",
        "circle",
        "triangle_down",
        "triangle_up",
        "triangle_left",
        "triangle_right",
        "octagon",
        "square",
        "pentagon",
        "plus (filled)",
        "star",
        "hexagon1",
        "hexagon2",
        "x (filled)",
        "diamond",
        "thin_diamond"
    ]
    rand0m_index = np.random.randint(0, len(mark3r))
    random_marker = mark3r[rand0m_index]
#     print(len(mark3r))
#     print(len(mark3r_desc))
    return random_marker


