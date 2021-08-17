import os
from absl import flags, app, logging
from absl.flags import FLAGS

flags.DEFINE_string('path',None,'relative path to download the dataset')


def download_dataset():
    """
    Function for downloading the feet dataset inside a google colab or kaggle notebook. It can also work in jupyter in linux.
    It just downloads the complete dataset in the current directory
    """
    ID = "1BfuvmCI_LjSpfT05UMnWXPBXBrEwo73E"
    script1 = f"""
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='{ID} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id="{ID} -O Data.zip && rm -rf /tmp/cookies.txt
    """
    script2 = "unzip Data.zip"
    script3 = "rm Data.zip"
    os.system(script1) # Download zip
    os.system(script2) # unzip
    os.system(script3) # delete zip

def main(_argv):
    path = FLAGS.path
    
    if path:
        os.chdir(path)

    download_dataset()  

if __name__ == '__main__':
    app.run(main) 
