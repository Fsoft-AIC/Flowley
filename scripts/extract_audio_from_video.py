import argparse
import pandas as pd
import multiprocessing as mp
import os
import glob
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Extract audio from video files')
    parser.add_argument('--tsv_file', type=str, help='TSV file containing video filepaths')
    parser.add_argument('--input_dir',
                        type=str,
                        help='Input directory containing video files. Use if tsv_file is not provided.')
    parser.add_argument('--output_dir', type=str, help='Output directory to save audio files')
    parser.add_argument('--num_workers',
                        type=int,
                        default=mp.cpu_count() // 2,
                        help='Number of workers to use')
    parser.add_argument('--ffmpeg_path', type=str, default="ffmpeg", help='Path to ffmpeg.')
    return parser.parse_args()


def extract_audio(input_file, ffmpeg_path, output_dir):
    output_file = os.path.join(output_dir, os.path.basename(input_file).replace(".mp4", ".wav"))
    if os.path.exists(output_file):
        return
    os.system(f'{ffmpeg_path} -i {input_file} -vn -acodec pcm_s16le -ar 16000 -ac 1 {output_file}')


def main():
    args = parse_args()
    if args.tsv_file:
        df = pd.read_csv(args.tsv_file, sep='\t')
        input_files = df['path'].tolist()
    else:
        input_files = glob.glob(os.path.join(args.input_dir, '**/*.mp4'), recursive=True)
    os.makedirs(args.output_dir, exist_ok=True)

    with mp.Pool(args.num_workers) as pool:
        list(tqdm(pool.starmap(extract_audio, [(input_file, args.ffmpeg_path, args.output_dir)
                                               for input_file in input_files]),
                  total=len(input_files), desc="Extracting audio"))


if __name__ == '__main__':
    main()
