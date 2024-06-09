import math

import numpy
import pandas
import librosa

from matplotlib import pyplot as plt

import argparse

import logging
from logger import get_logger

logger = get_logger(__name__, logging.DEBUG)

np = numpy  # code from FMP uses this


def normalize_feature_sequence(X, norm='2', threshold=0.0001, v=None):
    """Normalizes the columns of a feature sequence

    Notebook: C3/C3S1_FeatureNormalization.ipynb

    Args:
        X (np.ndarray): Feature sequence
        norm (str): The norm to be applied. '1', '2', 'max' or 'z' (Default value = '2')
        threshold (float): An threshold below which the vector ``v`` used instead of normalization
            (Default value = 0.0001)
        v (float): Used instead of normalization below ``threshold``. If None, uses unit vector for given norm
            (Default value = None)

    Returns:
        X_norm (np.ndarray): Normalized feature sequence
    """
    assert norm in ['1', '2', 'max', 'z']

    K, N = X.shape
    X_norm = np.zeros((K, N))

    if norm == '2':
        if v is None:
            v = np.ones(K, dtype=np.float64) / np.sqrt(K)
        for n in range(N):
            s = np.sqrt(np.sum(X[:, n] ** 2))
            if s > threshold:
                X_norm[:, n] = X[:, n] / s
            else:
                X_norm[:, n] = v
    else:
        raise ValueError("Norm type not supported")

    return X_norm


def compute_features(audio, sr, hop_length=512, n_mfcc=13, n_fft=None):
    if n_fft is None:
        n_fft = next_power_of_2(hop_length)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, hop_length=hop_length, n_mfcc=n_mfcc)
    # Normalize using Euclidean norm - as the diagonal matching code expects it
    mfcc = normalize_feature_sequence(mfcc)

    return mfcc


def cost_matrix_dot(X, Y):
    """Computes cost matrix via dot product

    Notebook: C7/C7S2_DiagonalMatching.ipynb

    Args:
        X (np.ndarray): First sequence (K x N matrix)
        Y (np.ndarray): Second sequence (K x M matrix)

    Returns:
        C (np.ndarray): Cost matrix
    """
    return 1 - np.dot(X.T, Y)


def matching_function_diag(C, cyclic=False):
    """Computes diagonal matching function

    Notebook: C7/C7S2_DiagonalMatching.ipynb

    Args:
        C (np.ndarray): Cost matrix
        cyclic (bool): If "True" then matching is done cyclically (Default value = False)

    Returns:
        Delta (np.ndarray): Matching function
    """
    N, M = C.shape
    print(f'N:{N}, M:{M}')
    assert N <= M, "N <= M is required"
    Delta = C[0, :]
    for n in range(1, N):
        Delta = Delta + np.roll(C[n, :], -n)
    Delta = Delta / N
    if cyclic is False:
        Delta[M - N + 1:M] = np.inf
    return Delta


def mininma_from_matching_function(Delta, rho=2, tau=0.2, num=None):
    """Derives local minima positions of matching function in an iterative fashion

    Notebook: C7/C7S2_DiagonalMatching.ipynb

    Args:
        Delta (np.ndarray): Matching function
        rho (int): Parameter to exclude neighborhood of a matching position for subsequent matches (Default value = 2)
        tau (float): Threshold for maximum Delta value allowed for matches (Default value = 0.2)
        num (int): Maximum number of matches (Default value = None)

    Returns:
        pos (np.ndarray): Array of local minima
    """
    Delta_tmp = numpy.array(Delta).copy()
    M = len(Delta)
    pos = []
    num_pos = 0
    rho = int(rho)
    if num is None:
        num = M
    while num_pos < num and np.sum(Delta_tmp < tau) > 0:
        m = np.argmin(Delta_tmp)
        # print(Delta_tmp.shape)
        # print('argmin', m, Delta_tmp[int(m)])
        pos.append(m)
        num_pos += 1
        # exclude this region from candidate minimums
        s = max(0, m - rho)
        e = min(m + rho, M)
        # print(s, e)
        Delta_tmp[s:e] = np.inf
    pos = np.array(pos).astype(int)
    return pos


def next_power_of_2(x):
    return 2 ** (math.ceil(math.log(x, 2)))


def plot_results(scores, threshold=None, events=None):
    fig, ax = plt.subplots(1, figsize=(30, 5))
    ax.plot(scores.reset_index()['time'], scores['distance'])

    if threshold is not None:
        ax.axhline(threshold, ls='--', alpha=0.5, color='black')

    if events is not None:
        for idx, e in events.iterrows():
            ax.axvspan(e['start'], e['end'], color='green', alpha=0.5)

    import matplotlib.ticker
    x_formatter = matplotlib.ticker.FuncFormatter(ticker_format_minutes_seconds)
    ax.xaxis.set_major_formatter(x_formatter)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=10 * 60))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=60))
    ax.grid(axis='x')
    ax.grid(axis='x', which='minor')

    return fig


def ticker_format_minutes_seconds(x, pos):
    hours = int(x // 3600)
    minutes = int((x % 3600) // 60)
    seconds = int(x % 60)

    return "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)


def find_audio(long, short, sr=44100, time_resolution=0.500, max_matches=10, score_threshold=0.1, ad_length=11):
    # distance between frames in feature representation [seconds]

    hop_length = int(time_resolution * samplerate)

    hop_length = 512
    print(f'hot_length:{hop_length}')

    # compute features for the audio
    query = compute_features(short, sr=sr, hop_length=hop_length)
    clip = compute_features(long, sr=sr, hop_length=hop_length)

    # Compute cost matrix and matching function
    C = cost_matrix_dot(query, clip)
    Delta = matching_function_diag(C)

    scores = pandas.DataFrame({
        'time': librosa.times_like(Delta, hop_length=hop_length, sr=samplerate),
        'distance': Delta,
    }).set_index('time')

    # convert to discrete
    match_idx = mininma_from_matching_function(scores['distance'].values,
                                               num=max_matches, rho=query.shape[1], tau=score_threshold)

    matches = scores.reset_index().loc[match_idx]
    matches = matches.rename(columns={'time': 'start'})
    # matches['end'] = matches['start'] + (query.shape[1] * time_resolution)
    matches['end'] = matches['start'] + ad_length
    logger.debug(f"matched_start:{matches['start']}, matched_end:{matches['end']}")

    matches = matches.reset_index()

    return scores, matches


import subprocess


def get_bitrate(file_path):
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=bit_rate', '-of',
           'default=noprint_wrappers=1:nokey=1', file_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    bitrate_bits_per_second = result.stdout.strip()  # 获取比特率字符串，单位为 bits/s
    if bitrate_bits_per_second.isdigit():
        bitrate_kbps = int(bitrate_bits_per_second) / 1000  # 将比特率转换为 kb/s
        return str(int(bitrate_kbps)) + 'k'  # 返回 ffmpeg 命令需要的格式
    else:
        return '128k'  # 如果无法获取比特率，则默认返回128kb/s


import os
import soundfile as sf


def remove_matched_segments(long_audio_path, matches, output_path, sr, force_update=False):
    # 获取原始文件的比特率
    bitrate = get_bitrate(long_audio_path)
    logger.debug(f"bitrate:{bitrate}")

    long_file_wav = long_audio_path.replace('.mp3', '.wav')
    # 加载长音频文件为立体声，不改变采样率
    long_audio, sr = librosa.load(long_file_wav, sr=None, mono=False)

    # 逆序处理匹配项
    for _, row in matches[::-1].iterrows():
        start_sample = int(row['start'] * sr)
        end_sample = int(row['end'] * sr)

        # 删除匹配到的片段，注意维持立体声格式
        long_audio = np.concatenate((long_audio[:, :start_sample], long_audio[:, end_sample:]), axis=1)

    # 指定输出临时WAV文件的路径
    output_temp_file_wav = long_file_wav.replace('.wav', '_temp.wav')

    # 使用soundfile保存修改后的立体声音频到新文件
    # 明确指定文件格式为WAV，子类型为PCM_16
    sf.write(output_temp_file_wav, long_audio.T, sr, format='WAV', subtype='PCM_16')

    # 使用ffmpeg转换WAV为MP3格式，确保输出为双声道立体声
    # ffmpeg_cmd = f"ffmpeg -i {output_temp_file_wav} -ac 2 -codec:a libmp3lame -b:a {bitrate}"
    ffmpeg_cmd = f'ffmpeg -i "{output_temp_file_wav}" -ac 2 -codec:a libmp3lame -b:a {bitrate}'

    if force_update:
        ffmpeg_cmd += " -y"
    ffmpeg_cmd += f' "{output_path}"'

    logger.debug(f"ffmped_cmd:{ffmpeg_cmd}")

    # 执行ffmpeg命令
    os.system(ffmpeg_cmd)

    # 删除临时WAV文件
    os.remove(output_temp_file_wav)


def remove_ads(ad_file: str, input_file: str, output_file: str):

    short_file = ad_file
    long_file = input_file

    long_file_wav = long_file.replace('.mp3', '.wav')

    #    os.system(f"ffmpeg -i {long_file}  {long_file_wav} -y")
    os.system(f'ffmpeg -i "{long_file}" "{long_file_wav}" -y')

    duration_short = librosa.get_duration(filename=short_file)
    duration_long = librosa.get_duration(filename=long_file)

    short, sr = librosa.load(short_file, sr=samplerate)
    long, sr = librosa.load(long_file_wav, sr=samplerate)

    logger.debug(f"Duration of short audio: {duration_short} seconds")
    logger.debug(f"Duration of long audio: {duration_long} seconds")

    scores, matches = find_audio(long, short, samplerate, max_matches=max_matches, score_threshold=threshold,
                                 ad_length=duration_short)

    # print results
    for idx, m in matches.iterrows():
        td = pandas.Timedelta(m['start'], unit='s').round('1s').to_pytimedelta()
        logger.error(f'{input_file} match {idx}: {td}')

    # 计算输出文件的路径
    # output_file = long_file.replace('.mp3', '_clean.mp3')

    # visualize results
    if (len(matches) > 0):
        # 移除匹配到的内容并保存结果
        remove_matched_segments(long_file, matches, output_file, samplerate, True)
        os.remove(long_file_wav)
        logger.debug(f'Removed matched segments and saved to {output_file}')

        output_fig_file = output_file.replace('.mp3', '.png')
        logger.debug(f'printing {output_fig_file}')
        fig = plot_results(scores, events=matches, threshold=threshold)
        fig.savefig(output_fig_file)
    else:
        logger.debug(f'No matched segments found, just copy the input file {long_file} to the output file {output_file}')
        os.system(f'cp "{long_file}" "{output_file}" ')

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


from mutagen.easyid3 import EasyID3
from mutagen.id3 import ID3, APIC, error

from mutagen.id3 import ID3, TIT2, ID3NoHeaderError

def update_metadata(file_path):
    try:
        audio = ID3(file_path)
    except ID3NoHeaderError:
        audio = ID3()

    title = os.path.splitext(os.path.basename(file_path))[0]

    # 清除现有的所有标签（如果需要保留封面或其他特定标签，请在此进行调整）
    audio.clear()

    # 仅添加标题标签
    audio.add(TIT2(encoding=3, text=title))

    # 保存更改，如果文件之前没有ID3标签，这也会添加一个标签
    audio.save(file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remove ads from audio files.')
    parser.add_argument('--ad_file', type=str, required=True, help='Path to the ad file.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input mp3 files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output mp3 files.')

    args = parser.parse_args()
    ad_file = args.ad_file
    input_dir = args.input_dir
    output_dir = args.output_dir

    # configuration
    samplerate = 44100
    threshold = 0.02
    max_matches = 5

    # 遍历input_dir中的所有文件和文件夹
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            # 构建完整的输入文件路径
            if file.lower().endswith('.mp3'):
                input_file = os.path.join(root, file)

                # 构建输出文件的路径，保持与input_dir相同的目录结构
                relative_path = os.path.relpath(root, input_dir)
                output_file_dir = os.path.join(output_dir, relative_path)
                output_file = os.path.join(output_file_dir, file)

                logger.debug(f'Processing input_file {input_file}, output_file:{output_file}')

                # 确保输出文件的目录存在
                ensure_dir(output_file)

                # 对每个文件执行remove_ads操作
                remove_ads(ad_file, input_file, output_file)

                # 更新标题元数据
                update_metadata(output_file)