import os


def download_dataset():
    link_list = ['https://storage.googleapis.com/dataset-uploader/bbc/bbc-text.csv']

    if not os.path.isdir('../input'):
        os.makedirs('../input')
        for item in link_list:
            os.system(f"wget --no-check-certificate {item} -O ../input/{item.split('/')[-1]}")
    else:
        for item in link_list:
            if not os.path.exists(f"../input/{item.split('/')[-1]}"):
                os.system(f"wget --no-check-certificate {item} -O ../input/{item.split('/')[-1]}")
            else:
                print('File already exits')


if __name__ == '__main__':
    download_dataset()
