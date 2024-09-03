import matplotlib.pyplot as plt
import numpy as np

if __name__=="__main__":
    input_file = 'map_z.txt'
    f_in = open(input_file, 'r')
    input_lines = f_in.readlines()

    x_as = []
    x_ps = []

    for line in input_lines:
        x_a = float(line.split(' => ')[0])
        x_p = line.split(' => ')[1]
        x_p = x_p[1:-2].replace('  ', ' ').strip()
        x_p = x_p.replace('  ', ' ').strip()
        x_p = np.asarray(x_p.split(' ')).astype(np.float)

        x_as.append(x_a)
        x_ps.append(x_p[1])
    
    plt.figure(figsize=(25, 4))
    plt.title('z convertion')
    plt.plot(x_ps)
    plt.xticks(range(len(x_as)), x_as)
    plt.savefig('test_z.png')
    plt.show()

