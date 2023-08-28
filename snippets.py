import pandas as pd

wavelengths = None


def load_wavelenth_file(filename):
    global wavelengths
    wavelengths = pd.read_json(filename)['wavelength'].astype(int)


def get_wavelength_index(wavelength):
    return int(wavelengths[wavelength])


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        exit
    load_wavelenth_file(sys.argv[1])
    print(get_wavelength_index(int(sys.argv[2])))
