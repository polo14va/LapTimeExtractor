import pandas as pd
import re
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
from geopy.distance import geodesic
import subprocess
import shutil  # For removing directories

# ============================
# Constants
# ============================

# Directory containing MP4 files
MP4_DIRECTORY = '/Users/pedro/Desktop/KOTAR-15-9-2024/DCIM/100GOPRO'

# Temporary directory for GPX files (inside MP4_DIRECTORY)
GPX_OUTPUT_DIRECTORY = os.path.join(MP4_DIRECTORY, 'gpx_data')

# Starting coordinates (Finish line)
START_LATITUDE = 41.791313
START_LONGITUDE = -3.583558

# Sector definitions (excluding 'Finish' from the sector order)
SECTOR_POINTS = [
    {'name': 'Sector 1', 'lat': 41.789794, 'lon': -3.580734},
    {'name': 'Sector 2', 'lat': 41.790549, 'lon': -3.582725},
    {'name': 'Sector 3', 'lat': 41.791172, 'lon': -3.582078},
    {'name': 'Finish', 'lat': 41.791313, 'lon': -3.583558}  # Handled separately
]

# Plot configurations
PLOT_TITLE = 'KOTARR BMW 1250GS Adventure'
PLOT_OUTPUT_FILE = 'lap_tracks.png'

# CSV output file
CSV_OUTPUT_FILE = 'laps_and_sectors.csv'

# Maximum allowed speed in meters per second for filtering outliers
MAX_SPEED_MPS = 120  #m/s Approximately 400 km/h

# Threshold distance in meters to consider crossing a sector
SECTOR_THRESHOLD_METERS = 5

# Line configurations for plotting sectors
LINE_LENGTH_METERS = 10
LINE_HEIGHT_METERS = 5

# ============================
# Functions
# ============================

def generate_gpx_files_from_mp4(mp4_directory, gpx_output_directory):
    """
    Generates GPX files from MP4 videos using exiftool.
    """
    os.makedirs(gpx_output_directory, exist_ok=True)  # Create the directory if it doesn't exist

    for file in os.listdir(mp4_directory):
        if file.upper().endswith('.MP4'):
            file_path = os.path.join(mp4_directory, file)
            output_gpx = os.path.join(gpx_output_directory, f"{os.path.splitext(file)[0]}.gpx")
            try:
                # Execute exiftool command to extract GPS data and save in GPX format
                with open(output_gpx, 'w') as gpx_file:
                    subprocess.run([
                        'exiftool', '-ee', '-gps:all', '-gpx:all', '-p',
                        '$GPSLatitude,$GPSLongitude,$GPSAltitude,$GPSDateTime',
                        file_path
                    ], stdout=gpx_file)
                print(f"GPX file generated: {output_gpx}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

def dms_to_dd(dms):
    """
    Converts coordinates from degrees, minutes, and seconds to decimal degrees.
    """
    try:
        parts = re.split(r'[^\d\.]+', dms)
        degrees = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        dd = degrees + (minutes / 60.0) + (seconds / 3600.0)
        if 'S' in dms or 'W' in dms:
            dd *= -1
        return dd
    except Exception as e:
        print(f"Error in dms_to_dd: {e}")
        return None

def parse_line(line):
    """
    Parses a line of GPS data and returns a list containing timestamp, latitude, longitude, and elevation.
    """
    try:
        parts = line.split(',')
        lat = dms_to_dd(parts[0])
        lon = dms_to_dd(parts[1])
        ele = float(parts[2].split(' ')[0])
        time = datetime.strptime(parts[3], '%Y:%m:%d %H:%M:%S.%f')
        return [time, lat, lon, ele]
    except Exception as e:
        print(f"Error in parse_line: {e}")
        return None

def process_file(file_path):
    """
    Processes a GPX file and returns a DataFrame with GPS data.
    """
    all_data = []
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.strip():
                    parsed = parse_line(line.strip())
                    if parsed:
                        all_data.append(parsed)
        
        df = pd.DataFrame(all_data, columns=['timestamp', 'latitude', 'longitude', 'elevation'])
        return df
    except Exception as e:
        print(f"Error in process_file: {e}")
        return pd.DataFrame(columns=['timestamp', 'latitude', 'longitude', 'elevation'])

def combine_files(directory):
    """
    Combines all GPX files into a single DataFrame.
    """
    all_data = []
    gpx_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith('.gpx')]
    gpx_files.sort(key=os.path.getmtime)

    for file_path in gpx_files:
        df = process_file(file_path)
        if not df.empty:
            all_data.append(df)
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame(columns=['timestamp', 'latitude', 'longitude', 'elevation'])

def latlon_to_xy(lat, lon, lat0, lon0):
    """
    Converts latitude and longitude coordinates to planar x, y coordinates relative to a reference point.
    """
    k = 111320  # meters per degree of latitude
    x = (lon - lon0) * np.cos(np.radians(lat0)) * k
    y = (lat - lat0) * k
    return x, y

def point_to_segment_distance(x0, y0, x1, y1, x2, y2):
    """
    Calculates the minimum distance from a point (x0, y0) to a segment defined by (x1, y1) and (x2, y2).
    """
    dx = x2 - x1
    dy = y2 - y1
    if dx == dy == 0:
        return np.hypot(x0 - x1, y0 - y1)
    t = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx*dx + dy*dy)
    t = max(0, min(1, t))
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    return np.hypot(x0 - closest_x, y0 - closest_y)

def line_segment_crosses_circle(x1, y1, x2, y2, cx, cy, r):
    """
    Checks if a line segment crosses a circle with center (cx, cy) and radius r.
    """
    # Translate the coordinate system so the circle is at the origin
    x1 -= cx
    y1 -= cy
    x2 -= cx
    y2 -= cy
    distance = point_to_segment_distance(0, 0, x1, y1, x2, y2)
    return distance <= r

def filter_outliers(df, max_speed=MAX_SPEED_MPS):
    """
    Filters out GPS points that imply a speed greater than max_speed (m/s).
    """
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['prev_lat'] = df['latitude'].shift(1)
    df['prev_lon'] = df['longitude'].shift(1)
    df['prev_time'] = df['timestamp'].shift(1)
    
    def calculate_speed(row):
        if pd.isnull(row['prev_lat']) or pd.isnull(row['prev_lon']) or pd.isnull(row['prev_time']):
            return np.nan
        distance = geodesic((row['prev_lat'], row['prev_lon']), (row['latitude'], row['longitude'])).meters
        time_diff = (row['timestamp'] - row['prev_time']).total_seconds()
        if time_diff == 0:
            return np.nan
        speed = distance / time_diff  # m/s
        return speed
    
    df['speed'] = df.apply(calculate_speed, axis=1)
    initial_count = len(df)
    df_filtered = df[df['speed'] <= max_speed].copy()
    filtered_count = len(df_filtered)
    print(f"Filtered out {initial_count - filtered_count} GPS points for speed > {max_speed} m/s")
    
    # Remove auxiliary columns
    df_filtered = df_filtered.drop(columns=['prev_lat', 'prev_lon', 'prev_time', 'speed'])
    
    return df_filtered.reset_index(drop=True)

def detect_laps_and_sectors(df, start_lat, start_lon, sector_points, threshold=SECTOR_THRESHOLD_METERS):
    """
    Detects laps and calculates the times for each sector.
    
    Parameters:
    - df: DataFrame with filtered GPS data.
    - start_lat, start_lon: Coordinates of the finish line.
    - sector_points: List of dictionaries with 'name', 'lat', 'lon' for each sector.
    - threshold: Distance in meters to consider crossing a sector point.
    
    Returns:
    - df: Original DataFrame with 'lap_number' assigned.
    - laps_df: DataFrame with lap and sector times.
    """
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['lap_number'] = np.nan

    # Convert all coordinates to x/y relative to the start point
    df['x'], df['y'] = latlon_to_xy(df['latitude'], df['longitude'], start_lat, start_lon)

    # Prepare sector coordinates in x/y
    sector_coords = {}
    for sector in sector_points:
        sx, sy = latlon_to_xy(sector['lat'], sector['lon'], start_lat, start_lon)
        sector_coords[sector['name']] = (sx, sy)

    # Finish line coordinates
    finish_x, finish_y = 0, 0  # Using start_lat, start_lon as reference

    lap_number = 0
    in_lap = False
    lap_data = []
    sector_order = [sector['name'] for sector in sector_points if sector['name'] != 'Finish']
    finish_sector = next((sector for sector in sector_points if sector['name'] == 'Finish'), None)
    if finish_sector is None:
        print("Error: 'Finish' sector not defined.")
        return df, pd.DataFrame()

    current_sector_idx = 0
    sector_start_times = {}

    row_prev = None
    lap_start_idx = None

    for idx, row in df.iterrows():
        if row_prev is not None:
            x1, y1 = row_prev['x'], row_prev['y']
            x2, y2 = row['x'], row['y']

            # Check if the finish line is crossed to start a lap
            crosses_finish_line = line_segment_crosses_circle(x1, y1, x2, y2, finish_x, finish_y, threshold)

            if crosses_finish_line:
                if not in_lap:
                    # Start a new lap
                    lap_number += 1
                    in_lap = True
                    lap_start_time = row['timestamp']
                    lap_start_idx = idx
                    sector_start_times = {sector: None for sector in sector_order}
                    sector_times_current_lap = {}
                    print(f"Starting Lap {lap_number} at {lap_start_time}")
                else:
                    # End the current lap upon crossing the finish line again
                    lap_end_time = row['timestamp']
                    lap_end_idx = idx
                    lap_time = (lap_end_time - lap_start_time).total_seconds()
                    
                    # Calculate time for Sector 4 (Finish)
                    if 'Sector 3' in sector_times_current_lap and sector_times_current_lap['Sector 3'] is not None:
                        sector4_time = (lap_end_time - sector_start_times['Sector 3']).total_seconds()
                        sector_times_current_lap['Finish'] = sector4_time
                    else:
                        sector_times_current_lap['Finish'] = None
                    
                    # Verify all sectors were detected
                    if all(sector_start_times[sector] is not None for sector in sector_order) and sector_times_current_lap.get('Finish') is not None:
                        lap_data.append({
                            'lap_number': lap_number,
                            'lap_time': lap_time,
                            **sector_times_current_lap
                        })
                        # Assign lap number to GPS points
                        df.loc[lap_start_idx:lap_end_idx, 'lap_number'] = lap_number
                        print(f"Ending Lap {lap_number} at {lap_end_time} with time {lap_time} seconds")
                    else:
                        print(f"Lap {lap_number} discarded due to incomplete sectors.")
                    
                    in_lap = False
                    current_sector_idx = 0
                    sector_start_times = {}
                    sector_times_current_lap = {}
            elif in_lap:
                # Check if sectors are crossed in order
                if current_sector_idx < len(sector_order):
                    expected_sector = sector_order[current_sector_idx]
                    sx, sy = sector_coords[expected_sector]
                    crosses_sector = line_segment_crosses_circle(x1, y1, x2, y2, sx, sy, threshold)
                    if crosses_sector:
                        sector_start_times[expected_sector] = row['timestamp']
                        if current_sector_idx == 0:
                            # First sector
                            sector_time = (sector_start_times[expected_sector] - lap_start_time).total_seconds()
                        else:
                            # Time since the previous sector
                            prev_sector_name = sector_order[current_sector_idx - 1]
                            if sector_start_times[prev_sector_name] is not None:
                                sector_time = (sector_start_times[expected_sector] - sector_start_times[prev_sector_name]).total_seconds()
                            else:
                                # Previous sector time is None, which should not happen
                                print(f"Warning: Previous sector '{prev_sector_name}' time is None in Lap {lap_number}")
                                sector_time = None
                        sector_times_current_lap[expected_sector] = sector_time
                        print(f"Lap {lap_number}: Sector '{expected_sector}' detected at {row['timestamp']} with time {sector_time} seconds")
                        current_sector_idx += 1

        row_prev = row

    # If still in a lap at the end of data
    if in_lap:
        lap_end_time = df.iloc[-1]['timestamp']
        lap_end_idx = df.index[-1]
        lap_time = (lap_end_time - lap_start_time).total_seconds()
        
        # Calculate time for Sector 4 (Finish)
        if 'Sector 3' in sector_times_current_lap and sector_times_current_lap['Sector 3'] is not None:
            sector4_time = (lap_end_time - sector_start_times['Sector 3']).total_seconds()
            sector_times_current_lap['Finish'] = sector4_time
        else:
            sector_times_current_lap['Finish'] = None
        
        # Verify all sectors were detected
        if all(sector_start_times[sector] is not None for sector in sector_order) and sector_times_current_lap.get('Finish') is not None:
            lap_data.append({
                'lap_number': lap_number,
                'lap_time': lap_time,
                **sector_times_current_lap
            })
            df.loc[lap_start_idx:lap_end_idx, 'lap_number'] = lap_number
            print(f"Ending Lap {lap_number} at the end of data with time {lap_time} seconds")
        else:
            print(f"Lap {lap_number} discarded due to incomplete sectors at the end of data.")

    laps_df = pd.DataFrame(lap_data)
    if not laps_df.empty:
        # Format lap times
        laps_df['formatted_lap_time'] = laps_df['lap_time'].apply(lambda x: str(timedelta(seconds=x))[:-3])
        for sector in ['Sector 1', 'Sector 2', 'Sector 3', 'Finish']:
            if sector in laps_df.columns:
                laps_df[f'{sector}_time_formatted'] = laps_df[sector].apply(
                    lambda x: str(timedelta(seconds=x))[:-3] if pd.notnull(x) and x is not None else None
                )
            else:
                laps_df[f'{sector}_time_formatted'] = None

    return df, laps_df

def lat_lon_to_offset(lat, lon, offset_meters, height_meters):
    """
    Converts latitude and longitude to degree offsets for drawing sector lines.
    """
    lat_offset = offset_meters / 111320
    lon_offset = offset_meters / (111320 * abs(np.cos(np.radians(lat))))
    height_offset = height_meters / (111320 * abs(np.cos(np.radians(lat))))
    return lat_offset, lon_offset, height_offset

def rotate_point(x, y, angle_degrees):
    """
    Rotates a point (x, y) around the origin by a given angle in degrees.
    """
    angle_radians = np.radians(angle_degrees)
    x_rotated = x * np.cos(angle_radians) - y * np.sin(angle_radians)
    y_rotated = x * np.sin(angle_radians) + y * np.cos(angle_radians)
    return x_rotated, y_rotated

def plot_laps(combined_df, laps_df, start_lat, start_lon, sector_points, threshold=SECTOR_THRESHOLD_METERS,
             line_length=LINE_LENGTH_METERS, line_height=LINE_HEIGHT_METERS, output_file=PLOT_OUTPUT_FILE,
             ideal_time=None, ideal_times=None):
    """
    Generates a PNG plot with lap tracks and lines indicating each sector's location.
    Adds the ideal sector time and its breakdown (excluding Finish, adding Finish time to Sector 1).
    """
    plt.figure(figsize=(20, 15))

    # Get the best lap
    if not laps_df.empty:
        best_lap_idx = laps_df['lap_time'].idxmin()
        best_lap_number = laps_df.iloc[best_lap_idx]['lap_number']
    else:
        best_lap_number = None

    # Get the colormap
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(len(laps_df))]

    # Variables to determine plot limits
    all_latitudes = []
    all_longitudes = []
    
    # Plot laps
    for index, row in laps_df.iterrows():
        lap_number = row['lap_number']
        # Filter data for the current lap
        lap_df = combined_df[combined_df['lap_number'] == lap_number]
        latitudes = lap_df['latitude'].tolist()
        longitudes = lap_df['longitude'].tolist()
        formatted_lap_time = row['formatted_lap_time']
        
        all_latitudes.extend(latitudes)
        all_longitudes.extend(longitudes)
        
        # Determine if this is the best lap
        if lap_number == best_lap_number:
            label = f"$\\mathbf{{Lap {lap_number:02d}: {formatted_lap_time}}}$"
            linewidth = 2
        else:
            label = f"Lap {lap_number:02d}: {formatted_lap_time}"
            linewidth = 1
        
        # Plot each lap track
        plt.plot(longitudes, latitudes, label=label, linestyle='-', linewidth=linewidth, color=colors[index])

    # Calculate offsets for sector lines
    sector_lines = {}
    for sector in sector_points:
        if sector['name'] == 'Finish':
            # Draw Finish lines in red
            lat_offset, lon_offset, height_offset = lat_lon_to_offset(sector['lat'], sector['lon'], line_length / 2, line_height / 2)
            sector_lines[sector['name']] = {
                'lat_offsets': [sector['lat'] - height_offset, sector['lat'] + height_offset],
                'lon_offsets': [sector['lon'] - lon_offset, sector['lon'] + lon_offset]
            }
            # Draw vertical and horizontal lines for Finish
            plt.plot(
                [sector['lon'], sector['lon']],
                [sector_lines[sector['name']]['lat_offsets'][0], sector_lines[sector['name']]['lat_offsets'][1]],
                color='red',
                linestyle='--',
                linewidth=2,
                label='Finish'
            )
            plt.plot(
                [sector_lines[sector['name']]['lon_offsets'][0], sector_lines[sector['name']]['lon_offsets'][1]],
                [sector['lat'], sector['lat']],
                color='red',
                linestyle='--',
                linewidth=2
            )
        else:
            # Draw sector lines in blue and rotate them 45 degrees
            lat_offset, lon_offset, height_offset = lat_lon_to_offset(sector['lat'], sector['lon'], line_length / 2, line_height / 2)
            
            # Apply 45-degree rotation
            lon_offset_rot, lat_offset_rot = rotate_point(lon_offset, lat_offset, 45)

            sector_lines[sector['name']] = {
                'lat_offsets': [sector['lat'] - lat_offset_rot, sector['lat'] + lat_offset_rot],
                'lon_offsets': [sector['lon'] - lon_offset_rot, sector['lon'] + lon_offset_rot]
            }

            # Draw rotated vertical and horizontal lines for each sector
            plt.plot(
                [sector_lines[sector['name']]['lon_offsets'][0], sector_lines[sector['name']]['lon_offsets'][1]],
                [sector['lat'], sector['lat']],
                color='blue',
                linestyle='--',
                linewidth=2,
                label=sector['name']
            )
            plt.plot(
                [sector['lon'], sector['lon']],
                [sector_lines[sector['name']]['lat_offsets'][0], sector_lines[sector['name']]['lat_offsets'][1]],
                color='blue',
                linestyle='--',
                linewidth=2
            )

    # Calculate plot limits with a margin
    if all_latitudes and all_longitudes:
        lat_min = min(all_latitudes)
        lat_max = max(all_latitudes)
        lon_min = min(all_longitudes)
        lon_max = max(all_longitudes)
        
        lat_margin = (lat_max - lat_min) * 0.050  # 5% margin
        lon_margin = (lon_max - lon_min) * 0.050  # 5% margin
        
        plt.xlim(lon_min - lon_margin, lon_max + lon_margin)
        plt.ylim(lat_min - lat_margin, lat_max + lat_margin)


    # Add ideal sector time to the plot
    if ideal_time is not None:
        # Add Finish time to Sector 1
        if ideal_times and ideal_times.get('Finish'):
            ideal_times['Sector 1'] += ideal_times.pop('Finish')
        
        ideal_time_formatted = str(timedelta(seconds=ideal_time))[:-3]
        ideal_times_str = "\n".join([f"{sector}: {str(timedelta(seconds=ideal_times[sector]))[:-3]}" for sector in ['Sector 1', 'Sector 2', 'Sector 3']])
        plt.text(lon_max, lat_min, f"Ideal Sector Times\n{ideal_time_formatted}\n\n{ideal_times_str}", 
                 fontsize=12, verticalalignment='bottom', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.7))

    # Hide axis labels and scales
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.gca().set_xlabel('')
    plt.gca().set_ylabel('')
    
    # Add the title
    plt.title(PLOT_TITLE, fontsize=24)
    
    # Add the legend
    plt.legend(loc='upper right', title='Lap Times', fontsize=14, title_fontsize=16)
    
    # Adjust layout to prevent clipping
    plt.tight_layout()
    
    # Save the plot to a file
    plt.savefig(output_file, dpi=300)
    plt.close()

def calculate_ideal_sector_time(laps_df, sector_order):
    """
    Calculates the ideal time by summing the best times of each sector.
    """
    ideal_time = 0
    ideal_times = {}
    for sector in sector_order:
        sector_times = laps_df[sector].dropna()
        if not sector_times.empty:
            best_sector_time = sector_times.min()
            ideal_times[sector] = best_sector_time
            ideal_time += best_sector_time
        else:
            ideal_times[sector] = None
    return ideal_time, ideal_times

# ============================
# Main Execution
# ============================

if __name__ == "__main__":
    # Generate GPX files from MP4
    generate_gpx_files_from_mp4(MP4_DIRECTORY, GPX_OUTPUT_DIRECTORY)

    # Combine GPX files into a single DataFrame
    combined_df = combine_files(GPX_OUTPUT_DIRECTORY)

    if combined_df.empty:
        print("No GPS data found in the .gpx files.")
    else:
        print(f"Combined data: {len(combined_df)} GPS points.")

        # Filter outlier GPS points based on speed
        combined_df = filter_outliers(combined_df, max_speed=MAX_SPEED_MPS)  # 30 m/s ≈ 108 km/h

        if combined_df.empty:
            print("All GPS points were filtered out as outliers.")
        else:
            print(f"Data after filtering: {len(combined_df)} GPS points.")

            # Detect laps and sectors
            combined_df, laps_df = detect_laps_and_sectors(
                combined_df, START_LATITUDE, START_LONGITUDE, SECTOR_POINTS, threshold=SECTOR_THRESHOLD_METERS
            )

            if laps_df.empty:
                print("No complete laps detected (all sectors detected).")
            else:
                # Calculate ideal sector times
                sector_order = ['Sector 1', 'Sector 2', 'Sector 3', 'Finish']
                ideal_time, ideal_times = calculate_ideal_sector_time(laps_df, sector_order)

                # Display ideal time in console
                ideal_time_formatted = str(timedelta(seconds=ideal_time))[:-3]
                print(f"\n**Ideal Sector Time:** {ideal_time_formatted}")
                print("Breakdown by sector:")
                for sector, time_sec in ideal_times.items():
                    if time_sec is not None:
                        time_formatted = str(timedelta(seconds=time_sec))[:-3]
                        print(f"  {sector}: {time_formatted}")
                    else:
                        print(f"  {sector}: Not available")

                # Determine the best lap
                best_lap_idx = laps_df['lap_time'].idxmin()
                best_lap = laps_df.iloc[best_lap_idx]
                best_lap_number = best_lap['lap_number']
                best_lap_time = best_lap['formatted_lap_time']

                print("\nDetected Laps:")
                for index, row in laps_df.iterrows():
                    lap_number = row['lap_number']
                    lap_time = row['formatted_lap_time']
                    sector_times = []
                    for sector in sector_order:
                        sector_time = row.get(sector)
                        if pd.notnull(sector_time) and sector_time is not None:
                            sector_time_formatted = row.get(f'{sector}_time_formatted')
                            sector_times.append(f"{sector}: {sector_time_formatted}")
                        else:
                            sector_times.append(f"{sector}: Not detected")
                    sector_times_str = ', '.join(sector_times)
                    print(f"LAP {lap_number:02d}: T-{lap_time}, {sector_times_str}")
                
                print(f"\n**Best Lap:** LAP {best_lap_number:02d} with time T-{best_lap_time}")

                # Save lap and sector data to CSV
                laps_df.to_csv(CSV_OUTPUT_FILE, index=False)
                print(f"Lap and sector times have been saved to '{CSV_OUTPUT_FILE}'.")

                # Plot the laps with tracks and sector lines
                plot_laps(
                    combined_df, laps_df, START_LATITUDE, START_LONGITUDE,
                    SECTOR_POINTS, threshold=SECTOR_THRESHOLD_METERS,
                    line_length=LINE_LENGTH_METERS, line_height=LINE_HEIGHT_METERS,
                    output_file=PLOT_OUTPUT_FILE, ideal_time=ideal_time, ideal_times=ideal_times
                )
                print(f"The lap plot has been saved as '{PLOT_OUTPUT_FILE}'.")

    # Remove the temporary GPX directory after completion
    try:
        shutil.rmtree(GPX_OUTPUT_DIRECTORY)
        print(f"Directory '{GPX_OUTPUT_DIRECTORY}' has been successfully removed.")
    except Exception as e:
        print(f"Error removing directory '{GPX_OUTPUT_DIRECTORY}': {e}")
