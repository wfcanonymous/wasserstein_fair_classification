import sys
import gdown
import urllib.request
from fairlib import datasets

if __name__ == '__main__':
    dataset = sys.argv[1]
    if dataset == "Bios":
        urls = [
            "https://drive.google.com/file/d/1YJ17Rn8F_fuQhvt8HWBBip7cnKHFeVw0/view?usp=share_link",
            "https://drive.google.com/file/d/1GYvAMVN95SPeg3MGdV0I8_ZT-KMgA52e/view?usp=share_link",
            "https://drive.google.com/file/d/1oDae8M3E_2jrzj6Mv8P6k_mYvXiiSMRz/view?usp=share_link",
        ]

        local_dirs = [
            "data/bios/bios_dev_df.pkl",
            "data/bios/bios_test_df.pkl",
            "data/bios/bios_train_df.pkl",
        ]

        for _url, _dir in zip(urls, local_dirs):
            gdown.download(url=_url, output=_dir, quiet=False, fuzzy=True)
    elif dataset == "Moji":
        datasets.prepare_dataset("moji", "data/moji")
    elif dataset == "Marked_personas":
        urllib.request.urlretrieve('https://github.com/myracheng/markedpersonas/blob/main/data/dv2/dv2_story_generations.csv',
                                   filename="./data/dv2_story_generations.csv")
    else:
        print('Error : wrong dataset name as input, must be Bios or Moji.')
        sys.exit(1)
    print('Download done !')
