import pandas as pd
import os
from pydub import AudioSegment

#TODO: Make this applicable by terminal

filenames = os.listdir(data_dir)

audio_filenames = []
for filename in filenames:
    if ".csv" not in (filename.lower()):
        audio_filenames.append(filename)


annotation_dict = {}
for audio_filename in audio_filenames:
    #csv should have same name as audio file except extension
    csv_filename = audio_filename.split(".")[0] + ".csv"
    #Using C engine because it's supposed to be faster, requires delimeter to be commas
    data_frame = pd.read_csv(data_dir + "/" + csv_filename, engine="c")
    #Storing all annotations in dictionary
    annotation_dict[audio_filename]=[]
    for _, row in data_frame.iterrows():
        annotation_dict[audio_filename].append(
                (float(row["onset"]),
                float(row["offset"]),
                str(row["class"]))
            )


for audio_filename in annotation_dict:
    print(audio_filename)
    annotations = annotation_dict[audio_filename]
    new_record_start = 0
    new_record_length = 10*1000
    hop_length = 5 * 1000
    record = AudioSegment.from_wav(data_dir + audio_filename)
    i = 0
    last = False
    while not last:
        #=> last new record from old record
        if (new_record_start + new_record_length) > len(record):
            new_record_start = len(record) - new_record_length
            last = True
        new_record = record[new_record_start:new_record_start + new_record_length]
        new_annotations = []
        for annotation in annotation_dict[audio_filename]:
            if (annotation[0]*1000) > (new_record_start + new_record_length):
                continue
            elif (annotation[1]*1000) < new_record_start:
                continue
            else:
                onset = (max(annotation[0]*1000, new_record_start) - new_record_start)/1000
                offset = (min(annotation[1]*1000, new_record_start+new_record_length) - new_record_start)/1000
                new_annotations.append((onset,offset,annotation[2]))
        if len(new_annotations) > 0:
            new_filename = audio_filename.split(".")[0] + "_" + str(i)
            print(new_filename)
            i+=1
            new_record.export(out_f=out_dir + new_filename + ".wav", format="wav")
            new_df = pd.DataFrame(new_annotations, columns=["onset", "offset", "class"])
            new_df.to_csv(out_dir + new_filename + ".csv")
        new_record_start += hop_length
        

