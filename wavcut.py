import wave

# 원하는 시간 단위 (초)
start = 0 # 시작 시간, seconds
end = 50 # 끝나는 시간, seconds

# file to extract the snippet from
with wave.open('/media/zxc6147/새 볼륨/ts5/DGBBB21000001.wav', "rb") as infile:
    # get file data
    nchannels = infile.getnchannels()
    sampwidth = infile.getsampwidth()
    framerate = infile.getframerate()
    # set position in wave to start of segment
    infile.setpos(int(start * framerate))
    # extract data
    data = infile.readframes(int((end - start) * framerate))

# write the extracted data to a new file
with wave.open('test.wav', 'w') as outfile:
    outfile.setnchannels(nchannels)
    outfile.setsampwidth(sampwidth)
    outfile.setframerate(framerate)
    outfile.setnframes(int(len(data) / sampwidth))
    outfile.writeframes(data)