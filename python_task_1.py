{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyO/0xbGfe4ShornLk28/WbE",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/MyclineShareena/BizCardX--Extracting-Business-Card-Data-with-OCR/blob/main/python_task_1.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Importing Library"
      ],
      "metadata": {
        "id": "ChZMuJt4Ujej"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd"
      ],
      "metadata": {
        "id": "WZ8gqyCiUhaw"
      },
      "execution_count": 24,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Load the CSV file into Pandas Dataframe"
      ],
      "metadata": {
        "id": "uVIuwwi4Uofm"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Read the dataset\n",
        "df = pd.read_csv(\"/content/dataset-1.csv\")"
      ],
      "metadata": {
        "id": "qdQog5cwUm-f"
      },
      "execution_count": 25,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Question 1"
      ],
      "metadata": {
        "id": "ppvvAog5PXHv"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def generate_car_matrix(df):\n",
        "\n",
        "    # Use pivot to create the matrix\n",
        "    car_matrix = df.pivot(index='id_1', columns='id_2', values='car')\n",
        "\n",
        "    # Fill NaN values with 0\n",
        "    car_matrix = car_matrix.fillna(0)\n",
        "\n",
        "    # Set diagonal values to 0\n",
        "    for i in range(len(car_matrix)):\n",
        "        car_matrix.iloc[i, i] = 0\n",
        "\n",
        "    return car_matrix\n",
        "\n",
        "car_matrix = generate_car_matrix(df)\n",
        "print(car_matrix)\n"
      ],
      "metadata": {
        "id": "DWStxkfn7XI7",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "82d669c8-e921-4ece-c290-819801346230"
      },
      "execution_count": 26,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "id_2    801    802    803    804    805    806    807    808    809    821  \\\n",
            "id_1                                                                         \n",
            "801    0.00   2.80   6.00   7.70  11.70  13.40  16.90  19.60  21.00  23.52   \n",
            "802    2.80   0.00   3.40   5.20   9.20  10.90  14.30  17.10  18.50  20.92   \n",
            "803    6.00   3.40   0.00   2.00   6.00   7.70  11.10  13.90  15.30  17.72   \n",
            "804    7.70   5.20   2.00   0.00   4.40   6.10   9.50  12.30  13.70  16.12   \n",
            "805   11.70   9.20   6.00   4.40   0.00   2.00   5.40   8.20   9.60  12.02   \n",
            "806   13.40  10.90   7.70   6.10   2.00   0.00   3.80   6.60   8.00  10.42   \n",
            "807   16.90  14.30  11.10   9.50   5.40   3.80   0.00   2.90   4.30   6.82   \n",
            "808   19.60  17.10  13.90  12.30   8.20   6.60   2.90   0.00   1.70   4.12   \n",
            "809   21.00  18.50  15.30  13.70   9.60   8.00   4.30   1.70   0.00   2.92   \n",
            "821   23.52  20.92  17.72  16.12  12.02  10.42   6.82   4.12   2.92   0.00   \n",
            "822   24.67  22.07  18.87  17.27  13.17  11.57   7.97   5.27   4.07   1.80   \n",
            "823   26.53  23.93  20.73  19.13  15.03  13.43   9.83   7.13   5.93   3.67   \n",
            "824   27.92  25.32  22.12  20.52  16.42   7.80  11.22   8.52   7.32   5.06   \n",
            "825   29.08  26.48  23.28  21.68  17.58  15.98  12.38   9.68   8.48   6.22   \n",
            "826   30.87  28.27  25.07  23.47  19.37  17.77  14.17  11.47  10.27   8.01   \n",
            "827   32.53  29.93  26.73  25.13  21.03  19.43  15.83  13.13  11.93   9.43   \n",
            "829   36.32  33.72  30.52  28.92  24.82  23.22  19.62  16.92  15.72  13.26   \n",
            "830   38.27  35.67  32.47  30.87  26.77  25.17  21.57  18.87  17.67  15.17   \n",
            "831   39.24  36.64  33.44  31.84  27.74  26.14  22.54  19.84  18.64  16.15   \n",
            "\n",
            "id_2    822    823    824    825    826    827    829    830    831  \n",
            "id_1                                                                 \n",
            "801   24.67  26.53  27.92  29.08  30.87  32.53  36.32  38.27  39.24  \n",
            "802   22.07  23.93  25.32  26.48  28.27  29.93  33.72  35.67  36.64  \n",
            "803   18.87  20.73  22.12  23.28  25.07  26.73  30.52  32.47  33.44  \n",
            "804   17.27  19.13  20.52  21.68  23.47  25.13  28.92  30.87  31.84  \n",
            "805   13.17  15.03  16.42  17.58  19.37  21.03  24.82  26.77  27.74  \n",
            "806   11.57  13.43  14.82  15.98  17.77  19.43  23.22  25.17  26.14  \n",
            "807    7.97   9.83  11.22  12.38  14.17  15.83  19.62  21.57  22.54  \n",
            "808    5.27   7.13   8.52   9.68  11.47  13.13  16.92  18.87  19.84  \n",
            "809    4.07   5.93   7.32   8.48  10.27  11.93  15.72  17.67  18.64  \n",
            "821    1.80   3.67   5.06   6.22   8.01   9.43  13.26  15.17  16.15  \n",
            "822    0.00   2.21   3.60   4.76   6.55   8.00  11.81  13.74  14.68  \n",
            "823    2.21   0.00   1.79   2.94   4.74   6.15  10.00  11.89  12.87  \n",
            "824    3.60   1.79   0.00   1.71   3.50   4.92   8.77  10.66  11.64  \n",
            "825    4.76   2.94   1.71   0.00   2.20   3.65   7.46   9.35  10.33  \n",
            "826    6.55   4.74   3.50   2.20   0.00   2.05   5.81   7.71   8.69  \n",
            "827    8.00   6.15   4.92   3.65   2.05   0.00   4.14   6.06   7.04  \n",
            "829   11.81  10.00  21.40   7.46   5.81   4.14   0.00   2.38   3.36  \n",
            "830   13.74  11.89  10.66   0.00   7.71   6.06   2.38   0.00   1.39  \n",
            "831   14.68  12.87  11.64  10.33   8.69   7.04   3.36   1.39   0.00  \n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Question 2"
      ],
      "metadata": {
        "id": "rIfos5E6ShiA"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def get_type_count(df):\n",
        "\n",
        "    # Add a new categorical column 'car_type'\n",
        "    conditions = [\n",
        "        (df['car'] <= 15),\n",
        "        (df['car'] > 15) & (df['car'] <= 25),\n",
        "        (df['car'] > 25)\n",
        "    ]\n",
        "    choices = ['low', 'medium', 'high']\n",
        "    df['car_type'] = pd.cut(df['car'], bins=[-float('inf'), 15, 25, float('inf')], labels=choices)\n",
        "\n",
        "    # Calculate the count of occurrences for each car_type category\n",
        "    type_count = df['car_type'].value_counts().to_dict()\n",
        "\n",
        "    # Sort the dictionary alphabetically based on keys\n",
        "    type_count = dict(sorted(type_count.items()))\n",
        "\n",
        "    return type_count\n",
        "\n",
        "type_count = get_type_count(df)\n",
        "print(df)\n",
        "print(type_count)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "n-U4XdyHPiOE",
        "outputId": "33563216-f45e-4e5c-8c6f-a437eb3532ec"
      },
      "execution_count": 28,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "     id_1  id_2  route   moto    car     rv   bus  truck car_type\n",
            "0     829   827      1   2.05   4.14   4.14  10.1   15.2      low\n",
            "1     829   821      4   6.63  13.26  13.26  32.4   48.5      low\n",
            "2     829   804      7  14.41  28.92  28.92  64.7   97.0     high\n",
            "3     829   822      6   5.90  11.81  11.81  28.8   43.2      low\n",
            "4     829   826      9   2.87   5.81   5.81  14.2   21.2      low\n",
            "..    ...   ...    ...    ...    ...    ...   ...    ...      ...\n",
            "336   803   802      3   1.70   3.40   3.40   6.9   10.3      low\n",
            "337   803   805      4   3.00   6.00   6.00  12.0   17.9      low\n",
            "338   803   825      3  11.59  23.28  23.28  50.1   75.2   medium\n",
            "339   803   806      9   3.80   7.70   7.70  15.3   23.0      low\n",
            "340   803   830      1  16.18  32.47  32.47  72.6  108.8     high\n",
            "\n",
            "[341 rows x 9 columns]\n",
            "{'high': 56, 'low': 196, 'medium': 89}\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Question 3"
      ],
      "metadata": {
        "id": "505wP9xFT33w"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def get_bus_indexes(df):\n",
        "\n",
        "    # Calculate the mean value of the 'bus' column\n",
        "    mean_bus_value = df['bus'].mean()\n",
        "    print(2*mean_bus_value)\n",
        "\n",
        "    # Identify indices where 'bus' values are greater than twice the mean\n",
        "    bus_indexes = df[df['bus'] > 2 * mean_bus_value].index.tolist()\n",
        "\n",
        "    # Sort the list in ascending order\n",
        "    bus_indexes.sort()\n",
        "\n",
        "    return bus_indexes\n",
        "\n",
        "bus_indexes = get_bus_indexes(df)\n",
        "print(bus_indexes)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Y2ajoLOgT3cl",
        "outputId": "f2314708-b352-4918-be8c-d33f5b732de5"
      },
      "execution_count": 34,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "63.86568914956011\n",
            "[2, 7, 12, 17, 25, 30, 54, 64, 70, 97, 144, 145, 149, 154, 160, 201, 206, 210, 215, 234, 235, 245, 250, 309, 314, 319, 322, 323, 334, 340]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Question 4"
      ],
      "metadata": {
        "id": "scLCHiF6XBtG"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def filter_routes(df):\n",
        "\n",
        "    # Filter routes based on the average of 'truck' column values\n",
        "    filtered_routes = df.groupby('route')['truck'].mean()\n",
        "    filtered_routes = filtered_routes[filtered_routes > 7].index.tolist()\n",
        "    # Sort the list in ascending order\n",
        "    filtered_routes.sort()\n",
        "\n",
        "    return filtered_routes\n",
        "\n",
        "filtered_routes = filter_routes(df)\n",
        "print(filtered_routes)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4ug-oZtqXFK_",
        "outputId": "0006fbb1-5ef2-457c-e74a-823f237c0dc6"
      },
      "execution_count": 39,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Question 5"
      ],
      "metadata": {
        "id": "O-sCuQBlZNQn"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def multiply_matrix(car_matrix):\n",
        "\n",
        "    # Copy the input DataFrame to avoid modifying the original\n",
        "    modified_matrix = car_matrix.copy()\n",
        "\n",
        "    # Modify values based on specified logic\n",
        "    modified_matrix = modified_matrix.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)\n",
        "\n",
        "    # Round values to 1 decimal place\n",
        "    modified_matrix = modified_matrix.round(1)\n",
        "\n",
        "    return modified_matrix\n",
        "\n",
        "modified_matrix = multiply_matrix(car_matrix)\n",
        "print(modified_matrix)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8UrSqtvKZbLn",
        "outputId": "175f7d4d-d03f-43a2-e859-1b460022f786"
      },
      "execution_count": 41,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "id_2   801   802   803   804   805   806   807   808   809   821   822   823  \\\n",
            "id_1                                                                           \n",
            "801    0.0   3.5   7.5   9.6  14.6  16.8  21.1  24.5  15.8  17.6  18.5  19.9   \n",
            "802    3.5   0.0   4.2   6.5  11.5  13.6  17.9  21.4  23.1  15.7  16.6  17.9   \n",
            "803    7.5   4.2   0.0   2.5   7.5   9.6  13.9  17.4  19.1  22.2  23.6  15.5   \n",
            "804    9.6   6.5   2.5   0.0   5.5   7.6  11.9  15.4  17.1  20.2  21.6  23.9   \n",
            "805   14.6  11.5   7.5   5.5   0.0   2.5   6.8  10.2  12.0  15.0  16.5  18.8   \n",
            "806   16.8  13.6   9.6   7.6   2.5   0.0   4.8   8.2  10.0  13.0  14.5  16.8   \n",
            "807   21.1  17.9  13.9  11.9   6.8   4.8   0.0   3.6   5.4   8.5  10.0  12.3   \n",
            "808   24.5  21.4  17.4  15.4  10.2   8.2   3.6   0.0   2.1   5.2   6.6   8.9   \n",
            "809   15.8  23.1  19.1  17.1  12.0  10.0   5.4   2.1   0.0   3.6   5.1   7.4   \n",
            "821   17.6  15.7  22.2  20.2  15.0  13.0   8.5   5.2   3.6   0.0   2.2   4.6   \n",
            "822   18.5  16.6  23.6  21.6  16.5  14.5  10.0   6.6   5.1   2.2   0.0   2.8   \n",
            "823   19.9  17.9  15.5  23.9  18.8  16.8  12.3   8.9   7.4   4.6   2.8   0.0   \n",
            "824   20.9  19.0  16.6  15.4  20.5   9.8  14.0  10.6   9.2   6.3   4.5   2.2   \n",
            "825   21.8  19.9  17.5  16.3  22.0  20.0  15.5  12.1  10.6   7.8   5.9   3.7   \n",
            "826   23.2  21.2  18.8  17.6  24.2  22.2  17.7  14.3  12.8  10.0   8.2   5.9   \n",
            "827   24.4  22.4  20.0  18.8  15.8  24.3  19.8  16.4  14.9  11.8  10.0   7.7   \n",
            "829   27.2  25.3  22.9  21.7  18.6  17.4  24.5  21.2  19.7  16.6  14.8  12.5   \n",
            "830   28.7  26.8  24.4  23.2  20.1  18.9  16.2  23.6  22.1  19.0  17.2  14.9   \n",
            "831   29.4  27.5  25.1  23.9  20.8  19.6  16.9  24.8  23.3  20.2  18.4  16.1   \n",
            "\n",
            "id_2   824   825   826   827   829   830   831  \n",
            "id_1                                            \n",
            "801   20.9  21.8  23.2  24.4  27.2  28.7  29.4  \n",
            "802   19.0  19.9  21.2  22.4  25.3  26.8  27.5  \n",
            "803   16.6  17.5  18.8  20.0  22.9  24.4  25.1  \n",
            "804   15.4  16.3  17.6  18.8  21.7  23.2  23.9  \n",
            "805   20.5  22.0  24.2  15.8  18.6  20.1  20.8  \n",
            "806   18.5  20.0  22.2  24.3  17.4  18.9  19.6  \n",
            "807   14.0  15.5  17.7  19.8  24.5  16.2  16.9  \n",
            "808   10.6  12.1  14.3  16.4  21.2  23.6  24.8  \n",
            "809    9.2  10.6  12.8  14.9  19.7  22.1  23.3  \n",
            "821    6.3   7.8  10.0  11.8  16.6  19.0  20.2  \n",
            "822    4.5   5.9   8.2  10.0  14.8  17.2  18.4  \n",
            "823    2.2   3.7   5.9   7.7  12.5  14.9  16.1  \n",
            "824    0.0   2.1   4.4   6.2  11.0  13.3  14.6  \n",
            "825    2.1   0.0   2.8   4.6   9.3  11.7  12.9  \n",
            "826    4.4   2.8   0.0   2.6   7.3   9.6  10.9  \n",
            "827    6.2   4.6   2.6   0.0   5.2   7.6   8.8  \n",
            "829   16.0   9.3   7.3   5.2   0.0   3.0   4.2  \n",
            "830   13.3   0.0   9.6   7.6   3.0   0.0   1.7  \n",
            "831   14.6  12.9  10.9   8.8   4.2   1.7   0.0  \n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Question 6\n"
      ],
      "metadata": {
        "id": "Da3rkV-Da1YN"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "time_df = pd.read_csv('/content/dataset-2.csv')\n",
        "\n",
        "def time_check(time_df):\n",
        "\n",
        "    # Combine 'startDay' and 'startTime' to create a datetime column 'start_timestamp'\n",
        "    time_df['start_timestamp'] = pd.to_datetime(time_df['startDay'] + ' ' + time_df['startTime'], errors='coerce', format='%Y-%m-%d %H:%M:%S')\n",
        "\n",
        "    # Combine 'endDay' and 'endTime' to create a datetime column 'end_timestamp'\n",
        "    time_df['end_timestamp'] = pd.to_datetime(time_df['endDay'] + ' ' + time_df['endTime'], errors='coerce', format='%Y-%m-%d %H:%M:%S')\n",
        "\n",
        "    # Drop rows with NaT values (resulting from parsing errors)\n",
        "    time_df = time_df.dropna(subset=['start_timestamp', 'end_timestamp'])\n",
        "\n",
        "    # Calculate the duration for each ('id', 'id_2') pair\n",
        "    time_df['duration'] = time_df['end_timestamp'] - time_df['start_timestamp']\n",
        "\n",
        "    # Check if the duration covers a full 24-hour period and spans all 7 days of the week\n",
        "    time_check_result = (\n",
        "        (time_df['duration'] >= pd.Timedelta(days=1)) &\n",
        "        (time_df['duration'] <= pd.Timedelta(days=7)) &\n",
        "        (time_df['start_timestamp'].dt.time == pd.to_datetime('00:00:00').time()) &\n",
        "        (time_df['end_timestamp'].dt.time == pd.to_datetime('23:59:59').time())\n",
        "    )\n",
        "\n",
        "    return time_check_result\n",
        "\n",
        "time_check_result = time_check(time_df)\n",
        "print(time_check_result)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "M8TF1bOEa6vk",
        "outputId": "1963b356-c170-4c6f-dea3-975375edbf35"
      },
      "execution_count": 60,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Series([], dtype: bool)\n"
          ]
        }
      ]
    }
  ]
}