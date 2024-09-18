# LapTimeExtractor

**LapTimeExtractor** is a Python tool designed to process one or multiple videos from a specified folder and automatically extract GPS data to calculate lap time metrics. This tool is ideal for analyzing racing videos where GPS data is embedded, and lap times need to be extracted and plotted for performance evaluation.

## Features
- Extract GPS data from MP4 videos.
- Calculate lap times and sector times based on GPS points.
- Filter out outlier GPS points based on speed.
- Generate a detailed CSV file with lap and sector times.
- Plot lap tracks, including sector lines, on a customizable map.

## Requirements

To run **LapTimeExtractor**, ensure you have the following installed:

- Python 3.x
- Required Python packages (install with `pip`):
  - `pandas`
  - `matplotlib`
  - `numpy`
  - `geopy`
- `exiftool` (required for extracting GPS data from MP4 videos)

You can install the required Python packages with:

```bash
pip install pandas matplotlib numpy geopy
```

To install `exiftool`:
- On macOS: `brew install exiftool`
- On Linux: Use your package manager, e.g., `sudo apt-get install exiftool`
- On Windows: [Download from the official website](https://exiftool.org/).

## How to Use

1. **Clone the repository**:

```bash
git clone https://github.com/yourusername/LapTimeExtractor.git
cd LapTimeExtractor
```

2. **Place your MP4 videos in a directory**. The tool expects the full absolute path to this directory. You need to set this path in the `MP4_DIRECTORY` constant within the `lap_time_extractor.py` script.

```python
MP4_DIRECTORY = '/absolute/path/to/your/mp4/videos'
```

3. **Customize Plot Settings**:  
   You can modify the following constants at the top of the script to adjust plot configurations:
   
   - **`PLOT_TITLE`**: The title of the plot.  
     Example:
     ```python
     PLOT_TITLE = 'Your Custom Title for Lap Times'
     ```

   - **`PLOT_OUTPUT_FILE`**: The name of the output file where the plot will be saved.  
     Example:
     ```python
     PLOT_OUTPUT_FILE = 'custom_lap_plot.png'
     ```

4. **Add or Update Sector Definitions**:  
   If your race track has different sectors, you can update the sector points by modifying the `SECTOR_POINTS` list in the script. You need to provide latitude and longitude for each sector. The script automatically calculates times based on when you cross these sector points.

   Example:
   ```python
   SECTOR_POINTS = [
       {'name': 'Sector 1', 'lat': 41.789794, 'lon': -3.580734},
       {'name': 'Sector 2', 'lat': 41.790549, 'lon': -3.582725},
       {'name': 'Sector 3', 'lat': 41.791172, 'lon': -3.582078},
       {'name': 'Finish', 'lat': 41.791313, 'lon': -3.583558}  # Finish line
   ]
   ```

5. **Run the Script**:  
   Once everything is set up, you can run the script to process the videos and extract lap times:

```bash
python lap_time_extractor.py
```

6. **Review the Results**:  
   The script will generate two output files:
   - A **CSV file** containing the lap and sector times (`laps_and_sectors.csv`).
   - A **PNG file** with the lap tracks plotted (`lap_tracks.png`).

## Example Output

Here is an example of the generated CSV file:

```csv
lap_number,lap_time,Sector 1,Sector 2,Sector 3,Finish,formatted_lap_time,Sector 1_time_formatted,Sector 2_time_formatted,Sector 3_time_formatted,Finish_time_formatted
1,97.23,30.45,28.31,22.12,16.35,1:37,00:30,00:28,00:22,00:16
2,96.89,30.10,28.20,22.40,16.19,1:36,00:30,00:28,00:22,00:16
...
```

And a plot showing the lap tracks with sector markers:

![lap_tracks](https://github.com/user-attachments/assets/4db7c897-7158-46f2-b46e-b2b022902cd8)


## Customization Options

- **CSV with Sector Definitions**:  
  You can load your sector definitions from a CSV file by modifying the script to read sector points from a file.

- **Speed Threshold**:  
  Adjust the `MAX_SPEED_MPS` constant to change the speed threshold used for filtering out GPS points that imply unrealistic speed (measured in meters per second).

```python
MAX_SPEED_MPS = 12  # m/s approximately 400 km/h
```

## License
This project is licensed under the APACHE 2.0 License
