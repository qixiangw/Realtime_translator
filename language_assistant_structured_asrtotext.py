# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
# Licensed under the Amazon Software License  http://aws.amazon.com/asl/

import boto3
import pyaudio
import os
import asyncio
import json
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import Result, Transcript, TranscriptEvent
from pytictoc import TicToc
import concurrent
import time

class textcolors:
    if not os.name == 'nt':
        blue = '\033[94m'
        green = '\033[92m'
        warning = '\033[93m'
        fail = '\033[91m'
        end = '\033[0m'
    else:
        blue = ''
        green = ''
        warning = ''
        fail = ''
        end = ''

class AudioConfig:
    def __init__(self):
        self.recorded_frames = []
        self.device_info = {}
        self.useloopback = False
        self.defaultframes = 1024
        self.recordtime = 100
        self.p = pyaudio.PyAudio()

def select_audio_device(config):
    """选择音频设备并确定使用模式"""
    # 获取默认设备
    try:
        default_device_index = config.p.get_default_input_device_info()
    except IOError:
        default_device_index = -1

    # 显示可用设备
    print(textcolors.blue + "Available devices:\n" + textcolors.end)
    for i in range(0, config.p.get_device_count()):
        info = config.p.get_device_info_by_index(i)
        print(textcolors.green + str(info["index"]) + textcolors.end + ": \t %s \n \t %s \n" % (
            info["name"], 
            config.p.get_host_api_info_by_index(info["hostApi"])["name"]
        ))
        if default_device_index == -1:
            default_device_index = info["index"]

    # 处理无可用设备的情况
    if default_device_index == -1:
        print(textcolors.fail + "No device available. Quitting." + textcolors.end)
        exit()

    # 获取用户选择或使用默认设备
    device_id = int(input("Choose device [" + textcolors.blue + str(default_device_index) + textcolors.end + "]: ") or default_device_index)
    print("")

    # 获取设备信息
    try:
        config.device_info = config.p.get_device_info_by_index(device_id)
    except IOError:
        config.device_info = config.p.get_device_info_by_index(default_device_index)
        print(textcolors.warning + "Selection not available, using default." + textcolors.end)

    # 选择音频模式
    is_input = config.device_info["maxInputChannels"] > 0
    is_wasapi = (config.p.get_host_api_info_by_index(config.device_info["hostApi"])["name"]).find("WASAPI") != -1

    if is_input:
        print(textcolors.blue + "Selection is input using standard mode.\n" + textcolors.end)
    else:
        if is_wasapi:
            config.useloopback = True
            print(textcolors.green + "Selection is output. Using loopback mode.\n" + textcolors.end)
        else:
            print(textcolors.fail + "Selection is input and does not support loopback mode. Quitting.\n" + textcolors.end)
            exit()

    return config

def select_translation_direction():
    """选择翻译方向并返回相应的参数"""
    direction = 1
    direction = int(input("Choose source and target language to translate. 1 for en to zh, 2 for zh to en [" + 
                         textcolors.blue + str(direction) + textcolors.end + "]: ") or str(direction))
    
    params = {}
    if direction == 1:
        params['source_language'] = "en"
        params['target_language'] = "zh"
        params['lang_code_for_polly'] = "cmn-CN"
        params['voice_id'] = "Zhiyu"
        params['lang_code_for_transcribe'] = "en-US"
    elif direction == 2:
        params['source_language'] = "zh"
        params['target_language'] = "en"
        params['lang_code_for_polly'] = "en-US"
        params['voice_id'] = "Joanna"
        params['lang_code_for_transcribe'] = "zh-CN"
    else:
        raise Exception("Languages not implemented!")
    
    return params

class MicrophoneStream:
    def __init__(self, config):
        self.stream = config.p.open(
            format=pyaudio.paInt16,
            channels=config.device_info["maxInputChannels"],
            rate=int(config.device_info["defaultSampleRate"]),
            input=True,
            frames_per_buffer=config.defaultframes,
            input_device_index=config.device_info["index"]
        )
        rate=int(config.device_info["defaultSampleRate"])
        frames_per_buffer=config.defaultframes
        print("Stream initialized:")
        print(frames_per_buffer)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            data = self.stream.read(defaultframes, exception_on_overflow=False)
            return data
        except IOError as e:
            print(f"IOError occurred: {e}")
            return None

    def close(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

def get_mic_stream(config):
    return MicrophoneStream(config)

def claude_translate(text, source_lang, target_lang):
    """使用Bedrock的Claude模型进行翻译"""
    global total_translation_time, translation_count
    bedrock = boto3.client('bedrock-runtime', region_name='us-west-2')
    model="anthropic.claude-3-5-sonnet-20241022-v2:0"
    translate_timer.tic()  # 开始计时
    
    prompt = f"Translate the following {source_lang} text to {target_lang}. Only return the translated text without any explanations or additional context:\n\n{text}"
    
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text":prompt
                    }
                ]   
            }
        ],
        "temperature": 0.0,
        "top_p": 1
    })
    
    response = bedrock.invoke_model(
        modelId=model,
        body=body
    )
    
    response_body = json.loads(response.get('body').read())
    
    translation_time = translate_timer.tocvalue()  # 获取翻译耗时
    total_translation_time += translation_time  # 累加总时间
    translation_count += 1  # 增加计数
    
    return response_body['content'][0]['text']

def Translate_service(text, source_lang, target_lang):
    translate = boto3.client(service_name='translate', region_name='us-west-2', use_ssl=True)
    trans_result = translate.translate_text(
                            Text = text,
                            SourceLanguageCode = source_lang,
                            TargetLanguageCode = target_lang
                        )
    text = trans_result.get("TranslatedText")
    return text

class MyEventHandler(TranscriptResultStreamHandler):
    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        global count
        global running_average
        global total_latency

        t.tic()
        results = transcript_event.transcript.results
        if len(results) > 0:
            if len(results[0].alternatives) > 0:
                transcript = results[0].alternatives[0].transcript
                print("transcript:", transcript)

                if hasattr(results[0], "is_partial") and results[0].is_partial == False:
                    t.tic()
                    if results[0].channel_id == "ch_0":
                        # 使用amazon saas service translate
                        text = Translate_service(
                            transcript,
                            params['source_language'],
                            params['target_language']
                        )
                        print("translated text:" + text)

                        #For doing accuracy measurements. Remove when not required.
                        with open("transcribe.txt", "a", encoding='utf-8') as f:
                            f.write(transcript + "\n")

                        with open("translate.txt", "a", encoding='utf-8') as f:
                            f.write(text + "\n")

                    t.toc("full result sent to translate:")

        count += 1
        total_latency += t.tocvalue()
        running_average = total_latency/count
        if (count % 1000 == 0) == True:
            print("Average Time so far: ", running_average)

async def transcribe(config):
    client = TranscribeStreamingClient(region="us-west-2")
    stream = await client.start_stream_transcription(
        language_code=params['lang_code_for_transcribe'],
        media_sample_rate_hz=int(config.device_info["defaultSampleRate"]),
        media_encoding="pcm",
    )
    recorded_frames = []
    async def write_chunks(stream):
        print("getting mic stream")
        mic_stream = get_mic_stream(config)
        try:
            async for chunk in mic_stream:
                if chunk is not None:
                    recorded_frames.append(chunk)
                    await stream.input_stream.send_audio_event(audio_chunk=chunk)
        finally:
            mic_stream.close()
            await stream.input_stream.end_stream()

    handler = MyEventHandler(stream.output_stream)
    await asyncio.gather(write_chunks(stream), handler.handle_events())

async def main():
    # 初始化配置
    config = AudioConfig()
    
    # 选择音频设备和模式
    config = select_audio_device(config)
    
    # 选择翻译方向
    global params
    params = select_translation_direction()
    
    # 初始化计时器和状态变量
    global t, translate_timer
    t = TicToc()
    translate_timer = TicToc()
    
    global count, total_latency, running_average, total_translation_time
    global translation_count, defaultframes
    count = 0
    total_latency = 0
    running_average = []
    total_translation_time = 0
    translation_count = 0
    defaultframes = config.defaultframes
    
    # 设置线程执行器
    global executor, loop
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
    loop = asyncio.get_event_loop()
    
    # 开始转录
    await transcribe(config)

if __name__ == "__main__":
    asyncio.run(main())
