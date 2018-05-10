# import os
#
# import tarfile
# import zipfile
#
# import numpy as np
#
# from Import.importData import glove_zip_file, glove_vectors_file, data_set_zip, train_set_file, test_set_file
#
# try:
#     from urllib.request import urlretrieve, urlopen
# except ImportError:
#     from urllib import urlretrieve
#     # from urllib2 import
# # large file - 862 MB
# if (not os.path.isfile(glove_zip_file) and
#         not os.path.isfile(glove_vectors_file)):
#     urlretrieve("http://nlp.stanford.edu/data/glove.6B.zip",
#                 glove_zip_file)
#
# def unzip_single_file(zip_file_name, output_file_name):
#     if not os.path.isfile(output_file_name):
#         with open(output_file_name, 'wb') as out_file:
#             with zipfile.ZipFile(zip_file_name) as zipped:
#                 for info in zipped.infolist():
#                     if output_file_name in info.filename:
#                         with zipped.open(info) as requested_file:
#                             out_file.write(requested_file.read())
#                             return
#
# def targz_unzip_single_file(zip_file_name, output_file_name, interior_relative_path):
#     if not os.path.isfile(output_file_name):
#         with tarfile.open(zip_file_name) as un_zipped:
#             un_zipped.extract(interior_relative_path+output_file_name)
# unzip_single_file(glove_zip_file, glove_vectors_file)
#
#
# # Deserialize GloVe vectors
# glove_wordmap = {}
# with open(glove_vectors_file, "r", encoding="utf8") as glove:
#     for line in glove:
#         name, vector = tuple(line.split(" ", 1))
#         glove_wordmap[name] = np.fromstring(vector, sep=" ")
