import matplotlib.pyplot as plt

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

nice_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 11,
    "font.size": 12,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}

markers = [
    'o',  # Circle
    's',  # Square
    '^',  # Triangle Up
    'v',  # Triangle Down
    '>',  # Triangle Right
    '<',  # Triangle Left
    'p',  # Pentagon
    '*',  # Star
    'h',  # Hexagon
    'H',  # Hexagon (new variant)
    '+',  # Plus
    'x',  # Cross
    'D',  # Diamond
    'd',  # Thin Diamond
    '|',  # Vertical Line
    '_'   # Horizontal Line
]


def set_size(width,
             square=False,
             fraction=1,
             fraction_height=1,
             subplots=(1, 1)):
    """ Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width -- float or string
            Document width in points, or string of predined document type.
    fraction -- float, optional
            Fraction of the width which you wish the figure to occupy.

    fraction_height -- float, optional
            Fraction of the height which you wish the figure to occupy.

    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'half':
        width_pt = 205
    elif width == 'full':
        width_pt = 430
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction

    # Convert from pt to inches
    inches_per_pt = 2 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2

    if square:
        fig_width_in = fig_width_pt * inches_per_pt
        fig_height_in = fig_width_pt * inches_per_pt
    else:
        # Figure width in inches
        fig_width_in = fig_width_pt * inches_per_pt
        # Figure height in inches
        fig_height_in = fig_width_in * golden_ratio * (
            subplots[0] / subplots[1]) * fraction_height

    return (fig_width_in, fig_height_in)


def generate_unique_colors(n_colors: int):
    """
    Generate a list of unique colors in hexadecimal format using a colormap.

    Parameters:
    n_colors (int): The number of unique colors desired.

    Returns:
    List[str]: A list of unique color hex strings.
    """
    # Generate colors in HSL space, then convert to RGB
    colors = []
    for i in range(n_colors):
        # Calculate hue, saturation (1.0), and lightness (0.5)
        hue = i / n_colors  # Spread hues evenly
        saturation = 1.0
        lightness = 0.5

        # Convert HSL to RGB
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(rgb)

    # Convert to hex format
    hex_colors = [
        '#{:02x}{:02x}{:02x}'.format(int(rgb[0] * 255), int(rgb[1] * 255),
                                     int(rgb[2] * 255)) for rgb in colors
    ]

    return hex_colors


if __name__ == '__main__':
    import colorsys

    import matplotlib.pyplot as plt
    set_size('half')
    test = generate_unique_colors(29)
    breakpoint()
