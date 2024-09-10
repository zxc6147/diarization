# 할거 메모
# diarization 코드 리팩토링
# toy n개로 쪼개서 model.fit iteration으로 학습 하기
# 이후 aihub data 쪼개서 사용
# predict와 L1, L2, cos 유사도 실행 시간 비교


import numpy as np
import uisrnn
import time
import math
import glob
import json
import librosa
import os

#a == 0이면 model load
#a == 1이면 my train data
#a == 2이면 toy 처음부터 학습
a = 1

start_time = time.time()

print("start")

if(a == 1):
    train_data = np.load('./my_train_data.npz', allow_pickle=True)
else:
    train_data = np.load('./toy_training_data.npz', allow_pickle=True)

test_data = np.load('./toy_testing_data.npz', allow_pickle=True)
train_sequence = train_data['train_sequence']
train_cluster_id = train_data['train_cluster_id']

test_sequences = test_data['test_sequences'].tolist()
test_cluster_ids = test_data['test_cluster_ids'].tolist()



model_args, training_args, inference_args = uisrnn.parse_arguments()

#이터레이션 숫자
training_args.train_iteration = 100
training_args.enforce_cluster_id_uniqueness = False
print(model_args)
print(training_args)
print(inference_args)

train_sequence = np.array(train_sequence)
train_sequence = np.squeeze(train_sequence)

train_cluster_id = np.array(train_cluster_id)
train_cluster_id = np.squeeze(train_cluster_id)

print (train_sequence.shape)
print (train_cluster_id.shape)

print(train_sequence)
print (train_cluster_id)

print("npz done : ", time.time() - start_time)

model = 0
#learn, save
if (a == 1 or a == 2):
    model = uisrnn.UISRNN(model_args)
    #내 npz로 fit이 지금 아예 안되는 듯
    model.fit(train_sequence, train_cluster_id, training_args) 
    temp_arr = np.array(model)
    np.save('model', temp_arr)
    print('model saved!!')
#load
else : 
    temp_arr = np.load('./model.npy', allow_pickle=True)
    model = temp_arr.tolist()

#예외처리
if model == 0:
   print("error, no file detected")


""" 
#predict
predicted_cluster_ids = []
test_record = []

i = 0
for(test_sequence, test_cluster_id) in zip(test_sequences, test_cluster_ids):
  predicted_cluster_id = model.predict(test_sequence, inference_args)
  predicted_cluster_ids.append(predicted_cluster_id)
  accuracy = uisrnn.compute_sequence_match_accuracy(test_cluster_id, predicted_cluster_id)
  test_record.append((accuracy, len(test_cluster_id)))
  print('Ground truth labels: ')
  print(test_cluster_id)
  print('Predicted labels: ')
  print(predicted_cluster_id)
  print('-' * 100)
  i+=1

output_result = uisrnn.output_result(model_args, training_args, test_record)
print(output_result)  
print(i) 

 """





# 75개 다 불러옴
# print(len(file_list))

frame_size = 256
my_test_sequences = []
my_test_cluster_ids = []

# 파일 저장 위치 global
file_list = glob.glob('E:/ts5/*.wav')

#파일 진행 퍼센트 확인 >> tqdm 가능
percent = 0

#파일 불러오기
for file in file_list:
    my_test_sequence = []

    #wav 파일 librosa로 불러오기
    data_float, sample_rate = librosa.load(file, sr = None)
    data_float = data_float.astype('float64')

    #print("type은?")
    #print(type(data_float[0]))

    for i in range(0, len(data_float), frame_size):
        frame = data_float[i:i+frame_size]
        #padding
        if len(frame) < frame_size:
            padding_length = frame_size - len(frame)
            frame = np.pad(frame, (0, padding_length), mode='constant')

        frame = np.expand_dims(frame, axis=1)
        my_test_sequence.append(frame)

    my_test_sequence = np.array(my_test_sequence)
    my_test_sequence = np.squeeze(my_test_sequence)

    #wav 전체 시간 = 전체 프레임 / sample rate
    #t = len(my_test_sequences) * frame_size / sample_rate
    #print(t)
    my_test_sequences.append(my_test_sequence)

    #같은 이름 다른 확장자
    json_file = os.path.splitext(file)[0]+'.json'
    #print(json_file)

    json_data = 0
    with open(json_file, 'r', encoding='UTF-8') as f:
        json_data = json.load(f)
    
    if json_data==0:
        print("error, no json file")

    result = []
    for file in json_file:
        # utterance 항목에서 필요한 값들 추출
        for utterance in json_data['utterance']:
            start = utterance['start']
            end = utterance['end']
            speaker_id = utterance['speaker_id'][-1:]  # speaker_id의 뒤 두 자리 추출
        
            # 추출한 데이터로 새로운 딕셔너리 생성
            result.append({
                'start': start,
                'end': end,
                'speaker_id': speaker_id
            })

    my_test_cluster_id = []
    for i in range(len(my_test_sequence)):
        my_test_cluster_id.append('0')

    for dic in result:
        #시작 시간 float로 표현
        n_start = float(dic['start'])
        n_end = float(dic['end'])

        try:
            #n_speak = '0_' + dic['speaker_id']
            n_speak = dic['speaker_id']
        except ValueError:
            n_speak = '0'
        

        # (nf = start)초일때 배열 my_test_cluster_id[N] 
        # nt = end
        # array_n_start = floor(len(my_test_cluster_id) * n_start / t)
        # array_n_end = floor(len(my_test_cluster_id) * n_end / t)
        # my_test_cluster_id[Nf]~[Nt] speaker_id(n_speak)로 초기화
        t = len(my_test_sequence) * frame_size / sample_rate
        array_n_start = math.floor(len(my_test_cluster_id) * n_start / t)
        array_n_end = math.floor(len(my_test_cluster_id) * n_end / t)

        while array_n_start <= array_n_end:
            my_test_cluster_id[array_n_start] = n_speak
            array_n_start += 1

    print(np.asarray(my_test_cluster_id).shape)
    my_test_cluster_ids.append(my_test_cluster_id)


    percent+=1
    print("percentage : ", percent / len(file_list) * 100)

    if percent == 1:
        break 

print("wav파일 모두 읽어서 배열에 저장한 시간 : ", time.time() - start_time)

my_test_sequences = np.array(my_test_sequences)
my_test_sequences = np.squeeze(my_test_sequences)

my_test_cluster_ids = np.array(my_test_cluster_ids)
my_test_cluster_ids = np.squeeze(my_test_cluster_ids) 

#개수 맞추기?
my_test_sequences = my_test_sequences[0:2000]
my_test_cluster_ids = my_test_cluster_ids[0:2000]

np.savez('my_train_data.npz', train_sequence=my_test_sequences, train_cluster_id=my_test_cluster_ids)


print("저장한 np array들을 npz로 저장한 시간 : ", time.time() - start_time)

 

predicted_cluster_ids = []
test_record = []


predicted_cluster_id = model.predict(my_test_sequences, inference_args)
predicted_cluster_ids.append(predicted_cluster_id)
accuracy = uisrnn.compute_sequence_match_accuracy(my_test_cluster_ids, predicted_cluster_id)
test_record.append((accuracy, len(my_test_cluster_ids)))
print('Ground truth labels: ')
print(my_test_cluster_ids)
print('Predicted labels: ')
print(predicted_cluster_id)
print('-' * 100)

output_result = uisrnn.output_result(model_args, training_args, test_record)
print(output_result)



print("실행 시간 : ", time.time() - start_time)