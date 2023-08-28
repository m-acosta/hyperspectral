import pandas as pd

# RED_RANGE = [655, 700]
# NIR_RANGE = [750, 1076]
# GREEN1 = 550
# YELLOW1 = 680
# RED1 = 680
# RED3 = 670
# NIR1 = 800
# NIR3 = 900
# NIR4 = 970
# NIR5 = 700

BLUE_RANGE = [400, 500]
GREEN_RANGE = [540, 570]
RED_RANGE = [600, 700]
NIR_RANGE = [760, 960]
SWIR1 = [1550, 1751]
BLUE1 = 445
NIR5 = 700
NIR6 = 705
NIR7 = 750
RED3 = 670
YELLOW1 = 550


def preprocess_spectra(df, n):
    df = average_dataframe(df, n).transpose()
    offset = int(df.iloc[0][0])
    nm = (int(x) for x in df.iloc[0])
    df = df.drop("Wavelengths", 0)
    df.columns = nm

    return df, offset


def average_dataframe(df, n):
    # Separate label column from data
    label = df.iloc[:, 0]
    df = df.iloc[:, 1:]

    if (df.shape[1]) % n != 0:
        raise Exception(
            "Number of columns: {}. Not divisible by {}.".format(df.shape[1], n))

    # Average every N columns
    average_df = pd.DataFrame()
    average_df["Wavelengths"] = label
    for i in range((df.shape[1]) // n):
        average_df[i] = df.iloc[:, i * n:(i + 1) * n].mean(axis=1)

    return average_df


def average_ranges(df, offset):
    average_dataframe = pd.DataFrame()

    blue_ranges = (b - offset for b in BLUE_RANGE)
    green_ranges = (g - offset for g in GREEN_RANGE)
    red_ranges = (r - offset for r in RED_RANGE)
    nir_ranges = (n - offset for n in NIR_RANGE)

    average_dataframe["BLUE"] = df.iloc[
        :, next(blue_ranges):next(blue_ranges)].mean(axis=1)
    average_dataframe["GREEN"] = df.iloc[
        :, next(green_ranges):next(green_ranges)].mean(axis=1)
    average_dataframe["RED"] = df.iloc[
        :, next(red_ranges):next(red_ranges)].mean(axis=1)
    average_dataframe["NIR"] = df.iloc[
        :, next(nir_ranges):next(nir_ranges)].mean(axis=1)

    return average_dataframe


# Broadband
def NDVI(RED1, NIR1):
    return (NIR1 - RED1) / (NIR1 + RED1)


def ENDVI(GREEN1, BLUE1, NIR1):
    return ((NIR1 + GREEN1) - (2 * BLUE1)) / ((NIR1 + GREEN1) + (2 * BLUE1))


def GNDVI(GREEN1, NIR1):
    return (NIR1 - GREEN1) / (NIR1 + GREEN1)


def EVI(RED1, BLUE1, NIR1):
    return (2.5 * (NIR1 - RED1)) / (NIR1 + 6 * RED1 - 7.5 * BLUE1 + 1)


def RedEdge(RED3, NIR5):
    return NIR5 / RED3


# Narrowband
def mND705(NIR6, NIR7, BLUE1):
    return (NIR7 - NIR6) / (NIR7 + NIR6 - 2 * BLUE1)


def MCARI(NIR5, RED3, YELLOW1):
    return (NIR5 - RED3) - (0.23 * (NIR5 - YELLOW1) * (NIR5 / RED3))


def WBI(W970, W900):
    return W970 / W900


def NDWI(W857, W1241):
    return (W857 - W1241) / (W857 + W1241)


def calculate_broadband(df, offset):
    avg_df = average_ranges(df, offset)
    broadband = pd.DataFrame()

    broadband['NDVI'] = NDVI(avg_df['RED'], avg_df['NIR'])
    broadband['ENDVI'] = ENDVI(avg_df['GREEN'], avg_df['BLUE'], avg_df['RED'])
    broadband['GNDVI'] = GNDVI(avg_df['GREEN'], avg_df['NIR'])
    broadband['EVI'] = EVI(avg_df['RED'], avg_df['BLUE'], avg_df['NIR'])
    broadband['RedEdge'] = RedEdge(avg_df['RED'], avg_df['NIR'])

    return broadband


def calculate_narrowband(df, offset):
    narrowband = pd.DataFrame()

    narrowband['mND705'] = mND705(df.iloc[:, NIR6 - offset],
                                  df.iloc[:, NIR7 - offset],
                                  df.iloc[:, BLUE1 - offset])
    narrowband['MCARI'] = MCARI(df.iloc[:, NIR5 - offset],
                                df.iloc[:, RED3 - offset],
                                df.iloc[:, YELLOW1 - offset])
    narrowband['WBI'] = WBI(df.iloc[:, 970 - offset],
                            df.iloc[:, 900 - offset])

    return narrowband


def calculate_indices(spectra, offset):
    return pd.concat([calculate_broadband(spectra, offset),
                      calculate_narrowband(spectra, offset)],
                     axis=1,
                     sort=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('i', type=str, help='path to file')
    parser.add_argument('-n', type=int, default=15,
                        help='number of spectra per reading')
    args = parser.parse_args()

    save_name = ''.join(args.i.split('.')[:-1] + ['_indices.csv'])

    print("Reading csv")
    in_csv = pd.read_csv(args.i, sep=None, engine='python')
    spectra, offset = preprocess_spectra(in_csv, args.n)

    df = calculate_indices(spectra, offset)

    df.to_csv(save_name, index=False)
