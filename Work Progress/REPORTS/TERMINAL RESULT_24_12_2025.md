python config/run_pipeline.py
(.venv) PS C:\Users\chall\Downloads\PROJECTS\Voxent or EchoForge\VOXENT> python config/run_pipeline.py 
C:\Users\chall\Downloads\PROJECTS\Voxent or EchoForge\VOXENT\.venv\Lib\site-packages\pyannote\audio\core\io.py:47: Us
erWarning:                                                                                                           torchcodec is not installed correctly so built-in audio decoding will fail. Solutions are:
* use audio preloaded in-memory as a {'waveform': (channel, time) torch.Tensor, 'sample_rate': int} dictionary;
* fix torchcodec installation. Error message was:

Could not load libtorchcodec. Likely causes:
          1. FFmpeg is not properly installed in your environment. We support
             versions 4, 5, 6 and 7.
          2. The PyTorch version (2.8.0+cpu) is not compatible with
             this version of TorchCodec. Refer to the version compatibility
             table:
             https://github.com/pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec.
          3. Another runtime dependency; see exceptions below.
        The following exceptions were raised as we tried to load libtorchcodec:

[start of libtorchcodec loading traceback]
FFmpeg version 7: Could not find module 'C:\Users\chall\Downloads\PROJECTS\Voxent or EchoForge\VOXENT\.venv\Lib\site-
packages\torchcodec\libtorchcodec_core7.dll' (or one of its dependencies). Try using the full path with constructor syntax.                                                                                                               FFmpeg version 6: Could not find module 'C:\Users\chall\Downloads\PROJECTS\Voxent or EchoForge\VOXENT\.venv\Lib\site-
packages\torchcodec\libtorchcodec_core6.dll' (or one of its dependencies). Try using the full path with constructor syntax.                                                                                                               FFmpeg version 5: Could not find module 'C:\Users\chall\Downloads\PROJECTS\Voxent or EchoForge\VOXENT\.venv\Lib\site-
packages\torchcodec\libtorchcodec_core5.dll' (or one of its dependencies). Try using the full path with constructor syntax.                                                                                                               FFmpeg version 4: Could not find module 'C:\Users\chall\Downloads\PROJECTS\Voxent or EchoForge\VOXENT\.venv\Lib\site-
packages\torchcodec\libtorchcodec_core4.dll' (or one of its dependencies). Try using the full path with constructor syntax.                                                                                                               [end of libtorchcodec loading traceback].
  warnings.warn(
2025-12-24 13:20:17,817 - INFO - Configuration validation passed
2025-12-24 13:20:17,820 - INFO - Starting batch processing of 2 files
Processing files:   0%|                                                                       | 0/2 [00:00<?, ?it/s]2
025-12-24 13:20:17,824 - INFO - Processing file: SaveClip.App_AQMGdlsfbIp7x2NXYZ8wKYDI4cJebCC1am_SlMX33HZV2bFNDV-teg2M0n-PoCDozUg8TwjqdLzYfeN2ei9gB4NFznFPuoa1Mx7V8GU_mp3.wav                                                             2025-12-24 13:20:17,825 - INFO - Configuration validation passed
2025-12-24 13:20:17,825 - WARNING - ML model not found at models/ml_gender_classifier.pkl, falling back to pitch-base
d classification                                                                                                     2025-12-24 13:20:26,700 - INFO - HTTP Request: HEAD https://huggingface.co/pyannote/speaker-diarization-community-1/r
esolve/main/config.yaml "HTTP/1.1 403 Forbidden"                                                                     
Could not download Pipeline from pyannote/speaker-diarization-community-1.
It might be because the repository is private or gated:

* visit https://hf.co/pyannote/speaker-diarization-community-1 to accept user conditions
* visit https://hf.co/settings/tokens to create an authentication token
* load the Pipeline with the `token` argument:
    >>> Pipeline.from_pretrained('pyannote/speaker-diarization-community-1', token='hf_....')

2025-12-24 13:20:26,864 - ERROR - Error processing file data/input_calls\SaveClip.App_AQMGdlsfbIp7x2NXYZ8wKYDI4cJebCC
1am_SlMX33HZV2bFNDV-teg2M0n-PoCDozUg8TwjqdLzYfeN2ei9gB4NFznFPuoa1Mx7V8GU_mp3.wav: Failed to load diarization pipeline: 403 Client Error. (Request ID: Root=1-694b9b45-30567b545f4e2491263740a0;b8506b08-4dd8-4fee-9d21-8d5796ab1645)      
Cannot access gated repo for url https://huggingface.co/pyannote/speaker-diarization-community-1/resolve/main/config.
yaml.                                                                                                                Access to model pyannote/speaker-diarization-community-1 is restricted and you are not in the authorized list. Visit 
https://huggingface.co/pyannote/speaker-diarization-community-1 to ask for access.                                   Processing files:  50%|███████████████████████████████▌                               | 1/2 [00:09<00:09,  9.04s/it]2
025-12-24 13:20:26,887 - INFO - Processing file: SaveClip.App_AQNGxC8bWFDBf_Iz3MoSshZCuOHDVF_4M3Cta9NVxuDWRfhUDpKo51cdv7zk1INEiU8jwjV_mj23zg7n7WWatiDzew9jh57Cnr7K7FE_mp3.wav                                                             2025-12-24 13:20:26,887 - INFO - Configuration validation passed
2025-12-24 13:20:29,179 - INFO - HTTP Request: HEAD https://huggingface.co/pyannote/speaker-diarization-community-1/r
esolve/main/config.yaml "HTTP/1.1 403 Forbidden"                                                                     
Could not download Pipeline from pyannote/speaker-diarization-community-1.
It might be because the repository is private or gated:

* visit https://hf.co/pyannote/speaker-diarization-community-1 to accept user conditions
* visit https://hf.co/settings/tokens to create an authentication token
* load the Pipeline with the `token` argument:
    >>> Pipeline.from_pretrained('pyannote/speaker-diarization-community-1', token='hf_....')

2025-12-24 13:20:29,189 - ERROR - Error processing file data/input_calls\SaveClip.App_AQNGxC8bWFDBf_Iz3MoSshZCuOHDVF_
4M3Cta9NVxuDWRfhUDpKo51cdv7zk1INEiU8jwjV_mj23zg7n7WWatiDzew9jh57Cnr7K7FE_mp3.wav: Failed to load diarization pipeline: 403 Client Error. (Request ID: Root=1-694b9b47-2aee2cf50e8be4e757a10610;91ae0d2a-06fd-49e4-81c1-b1400227d736)      
Cannot access gated repo for url https://huggingface.co/pyannote/speaker-diarization-community-1/resolve/main/config.
yaml.                                                                                                                Access to model pyannote/speaker-diarization-community-1 is restricted and you are not in the authorized list. Visit 
https://huggingface.co/pyannote/speaker-diarization-community-1 to ask for access.                                   Processing files: 100%|███████████████████████████████████████████████████████████████| 2/2 [00:11<00:00,  5.68s/it] 
2025-12-24 13:20:29,193 - INFO - Running data augmentation...
Starting dataset balancing with augmentation...
Current distribution: {'male': 0, 'female': 0, 'uncertain': 0}
Dataset balancing completed. Created 0 augmented samples.
Final distribution: {'male': 0, 'female': 0, 'uncertain': 0}
2025-12-24 13:20:29,197 - INFO - Data augmentation completed
2025-12-24 13:20:29,197 - INFO - Batch processing completed: 0 successful, 2 failed


python train_ml_classifier.py --min-confidence 70
2025-12-24 13:22:44,827 - INFO - Loaded configuration from config/config.yaml
2025-12-24 13:22:44,829 - ERROR - Training failed: Metadata file not found: data/voice_dataset/metadata.csv


python web_app.py
(.venv) PS C:\Users\chall\Downloads\PROJECTS\Voxent or EchoForge\VOXENT> python web_app.py
Traceback (most recent call last):
  File "C:\Users\chall\Downloads\PROJECTS\Voxent or EchoForge\VOXENT\web_app.py", line 3, in <module>
    from flask import Flask, request, jsonify, send_from_directory, render_template, redirect, url_for
ModuleNotFoundError: No module named 'flask'
