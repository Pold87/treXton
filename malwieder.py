import shlex
import subprocess

"""

This helper module creates heatmaps

"""



cl_raw = "convert -size 100x100 xc: -sparse-color  Shepards \
              '30,10 red  10,80 blue  70,60 lime  80,20 yellow' \
          -fill white -stroke black \
          -draw 'circle 30,10 30,12  circle 10,80 10,82' \
          -draw 'circle 70,60 70,62  circle 80,20 80,22' \
          sparse_shepards.png"

print shlex.split(cl_raw)



def create_center(x, y, color):

    return str(x) + ',' + str(y) + ' ' + color


def create_centers(centers, matches):

    colors = ['red',
              'orange',
              'green',
              'purple',
              'DeepSkyBlue',
              'DeepSkyBlue4']

    strings = []

    for i, m in enumerate(matches):

        x, y = centers[m]

        color_str = create_center(x, y, colors[i])
        strings.append(color_str)


    return ' '.join(strings)
        

def heatmap(width, height, centers, matches, method='Shepards'):
    
    center_strings_with_colors = create_centers(centers, matches)

    cl =  ['convert', '-size', str(width) + 'x' + str(height),
           'xc:', '-sparse-color', 
           method,
           center_strings_with_colors,
           'sparse_shepards.png']

    subprocess.call(cl)

if __name__ == '__main__':

    heatmap(100, 100)


