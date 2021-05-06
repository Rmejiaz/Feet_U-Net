import os

def download_dataset():
    """
    Function for downloading the feet dataset inside a google colab or kaggle notebook. It can also work in jupyter in linux.
    It just downloads the complete dataset in the current directory
    """
    script1 = """
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='1SVBx33Yrab7OrABDqp6hoW0uTJTv8OOO -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id="1SVBx33Yrab7OrABDqp6hoW0uTJTv8OOO -O Data.zip && rm -rf /tmp/cookies.txt
    """
    script2 = """unzip Data.zip"""

    os.system(script1)
    os.system(script2)

