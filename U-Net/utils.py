def download_dataset():
    """
    Function for downloading the feet dataset inside a google colab or kaggle notebook. It can also work in jupyter in linux.
    """
    FILEID = "1SVBx33Yrab7OrABDqp6hoW0uTJTv8OOO"
    !wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$FILEID -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id="$FILEID -O Data.zip && rm -rf /tmp/cookies.txt

    !unzip Data.zip

