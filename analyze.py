import os, json, pprint
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Global settings/variables
logs_folder = "Logs"
logs_extra_folder = "Logs_EXTRA"
logs_hi_folder = "Logs_HI"

# Make a list of file names of everything in logs_folder
def get_log_filenames():
    all_filenames = list()
    paths = list()
    for root, dirs, files in os.walk(logs_folder):
        for filename in files:
            paths.append(logs_folder + "/" + filename)
            all_filenames.append(filename)
    for root, dirs, files in os.walk(logs_extra_folder):
        for filename in files:
            paths.append(logs_extra_folder + "/" + filename)
            all_filenames.append(filename)
    for root, dirs, files in os.walk(logs_hi_folder):
        for filename in files:
            paths.append(logs_hi_folder + "/" + filename)
            all_filenames.append(filename)
    return all_filenames, paths

# Let user select a filename given all the file names
def select_file(all_filenames, all_filepaths):
    print(f"The following log files are available:")
    for idx, filename in enumerate(all_filenames):
        print(f"\t[{idx}]\t{filename}")
    selection = input(f"Select one [0-{len(all_filenames)}]: ")
    print("")
    return all_filenames[int(selection)], all_filepaths[int(selection)]

# Load a file given its name
def load_file(filename):
    with open(logs_folder + filename) as file:
        return (filename, json.load(file))

# get average, sd, size, environment and algorithm from Log files
def extract_data_from_file(file):
    contained_fires = []
    f = open(file, "r")
    f1 = f.readlines()
    for x in f1:
        idx_min = x.find(":")
        idx_max = x.find("}")
        value = int(x[idx_min+1:idx_max])
        contained_fires.append(value)
    average = np.mean(contained_fires)
    sd = np.std(contained_fires)
    se = stats.sem(contained_fires)
    size = file[file.find("/")+1:file.find("s-forest")]
    environment = file[file.find("0s-")+2:file.find("-CNN")]
    algorithm = file[file.find("-CNN")+1:]
    print(size + "\t" + environment + "\t" + algorithm)
    print("Mean:\t\t\t\t\t" + str(average))
    print("Standard deviation:\t\t\t" + str(sd))
    print("Standard error:\t\t\t\t" + str(se))
    # Null hypothesis that data is drawn from normal distribution
    W, p = stats.shapiro(contained_fires)
    # if p<0.05:
    #     print('Sample does not look Gaussian (reject H0)')
    # else:
    #     print('Sample looks Gaussian (fail to reject H0)')
    return average, se, size, environment, algorithm

# Plot number of fires contained in bar plot
def plot_number_fires_contained(selected_files):
    # Start the plot
    folder_name = str()
    cnn_average_array = []
    cnn_extra_average_array = []
    hi_cnn_average_array = []
    cnn_sd_array = []
    cnn_extra_sd_array = []
    hi_cnn_sd_array = []

    # Get data for plot
    for selected_file in selected_files:
        average, sd, size, environment, algorithm = extract_data_from_file(selected_file)
        if algorithm == "CNN_EXTRA":
            cnn_extra_average_array.append(average)
            cnn_extra_sd_array.append(sd)
        elif algorithm == "CNN_HI":
            hi_cnn_average_array.append(average)
            hi_cnn_sd_array.append(sd)
        else:
            cnn_average_array.append(average)
            cnn_sd_array.append(sd)

        folder_name = folder_name + selected_file[1]

    # Create base of the bar plot
    plt.clf()
    plt.title("Contained fires for " + str(size) + "*" + str(size) + " map\n (tested on 30 models for 100 episodes)",{'fontsize': 15})
    plt.ylabel("#Fires Contained(in 100 episodes)", {'fontsize': 15})
    # Create bar plot
    barWidth = 0.3

    # The x position of bars
    r1 = np.arange(len(cnn_average_array))
    r2 = [x + barWidth for x in r1]
    if str(size) == '10':
        r3 = [1.6, 2.6, 3.6]
    else:
        r3 = [x + barWidth for x in r2]

    # Create bars
    plt.bar(r1, cnn_average_array, width=barWidth, color='khaki', edgecolor='khaki', yerr=cnn_sd_array, capsize=7, label='DCNN(50 examples)')
    plt.bar(r2, cnn_extra_average_array, width=barWidth, color='lightsalmon', edgecolor='lightsalmon', yerr=cnn_extra_sd_array, capsize=7, label='DCNN(100 examples)')
    plt.bar(r3, hi_cnn_average_array, width=barWidth, color='indianred', edgecolor='indianred', yerr=hi_cnn_sd_array, capsize=7, label='HI-DCNN(100 examples)')

    # general layout
    plt.xticks([r + barWidth for r in range(len(cnn_average_array))], ['F', 'F+H', 'F+H+R', 'F+R'], size=15)
    plt.legend()
    plt.grid()
    plt.ylim(0, 105)

    # Setting the background color
    ax = plt.axes()
    ax.set_facecolor("lightgray")
    # Show graphic
    plt.show()

# computes whether there is a significant difference between two samples
def compute_significance(selected_files):
    contained_fires1 = []
    f = open(selected_files[0], "r")
    f1 = f.readlines()
    for x in f1:
        idx_min1 = x.find(":")
        idx_max1 = x.find("}")
        value = int(x[idx_min1 + 1:idx_max1])
        contained_fires1.append(value)
    contained_fires2 = []
    f = open(selected_files[1], "r")
    f2 = f.readlines()
    for x in f2:
        idx_min2 = x.find(":")
        idx_max2 = x.find("}")
        value = int(x[idx_min2 + 1:idx_max2])
        contained_fires2.append(value)
    W, p1 = stats.shapiro(contained_fires1)
    W, p2 = stats.shapiro(contained_fires2)
    if p1<0.05 or p2<0.05:
        print('One or two of the samples does not look Gaussian, so the Wilcoxon rank-sum test is performed!')
        t, p = stats.ranksums(contained_fires1, contained_fires2)
        if p<0.05:
            significance = True
        else:
            significance = False
        print("P-value :" + str(p) + " t-value :" + str(t) + " so signifance :" + str(significance))
    else:
        print('Samples do look Gaussian, so the two tailed unpaired t-test is performed!')
        t, p = stats.ttest_ind(contained_fires1, contained_fires2, equal_var=False)
        if p<0.05:
            significance = True
        else:
            significance = False
        print("P-value :" + str(p) + " t-value :" + str(t) + " so signifance :" + str(significance))



### MAIN
def main():
    selecting_files = True
    selected_files = []
    all_filenames, all_file_paths = get_log_filenames()

    plot_or_compute_significance = input(f"Plot of compute significance [p/cs]: ")
    if plot_or_compute_significance == "p":
        while selecting_files:
            name, selected_file = select_file(all_filenames, all_file_paths)
            selected_files.append(selected_file)
            all_filenames.remove(name)
            all_file_paths.remove(selected_file)

            if len(selected_files) >= 10:
                file_select_answer = input(f"Select more files? [y/n]: ")
                if file_select_answer == "n":
                    selecting_files = False
        print("")
        plot_number_fires_contained(selected_files)
    elif plot_or_compute_significance == "cs":
        while selecting_files:
            name, selected_file = select_file(all_filenames, all_file_paths)
            selected_files.append(selected_file)
            all_filenames.remove(name)
            all_file_paths.remove(selected_file)

            if len(selected_files) >= 2:
                    selecting_files = False
        print("")
        compute_significance(selected_files)





if __name__ == "__main__":
    main()
