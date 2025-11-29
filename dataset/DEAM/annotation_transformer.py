import csv

input_csv = ['DEAM_Annotations/static_annotations_averaged_songs_1_2000.csv', 'DEAM_Annotations/static_annotations_averaged_songs_2000_2058.csv']
output_csv = 'DEAM_Annotations/static_annotations_averaged_songs.csv'

field_names = ['song_id', 'valence_mean', 'valence_std', 'arousal_mean', 'arousal_std']

with open(output_csv, "w", newline="") as f_out:   # Comment 2 below
  writer = csv.DictWriter(f_out, fieldnames=field_names, extrasaction='ignore')
  writer.writeheader()

  for filename in input_csv:
    with open(filename, "r", newline="") as f_in:
      reader = csv.DictReader(f_in)
      for line in reader:
        writer.writerow(line)