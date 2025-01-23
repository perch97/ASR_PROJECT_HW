# Проект по распознаванию речи (ASR)

<p align="center">
  <a href="#о-проекте">О проекте</a> •
  <a href="#Установка и запуск модели"">Установка и запуск модели</a> •
  <a href="#Логи обучения">Логи обучения</a> •
  <a href="#метрики-wer-и-cer-для-финальной-модели">Метрики WER и CER для финальной модели</a> •
</p>


## О проекте

Этот репозиторий содержит пример решения задачи автоматического распознавания речи (ASR). 
Задание можно найти [здесь](https://github.com/NickKar30/SpeechAI/tree/main/hw2). 

## Установка и запуск модели

Через терминал
п.1
  ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

п.2 
   ```bash
   pip install -r requirements.txt
   ```
п.3 
   ```bash
   pre-commit install
   ```
п.4
```bash
  export WANDB_API_KEY=<use-your-api-key>
  echo $WANDB_API_KEY
```
п.5 запуск модели
```bash
  python3 train.py -cn=CONFIG_NAME #check configs in src/configs
```
п.6 запуск инференса модели
```bash
   python3 inference.py HYDRA_CONFIG_ARGUMENTS
```
либо в Google Colab

п.1
  ```bash
  !git clone https://github.com/perch97/asr_hw.git
  %cd asr_hw
  ```

п.2  
  ```bash
  import wandb
  wandb.login(key='use-your-api-key') # Pass the API key using the 'key' keyword argument
  ```
п.3
  ```bash
  import os
  os.environ["WANDB_API_KEY"] = use-your-api-key
  ```
п.4 запуск модели
```bash
    !python3 train.py -cn=train_argmax
```

п.5 запуск инференса модели
```bash
    !python3 inference.py -cn=inference
```
## Логи обучения


Логи обучения с использованием варианта train_argmax.yaml доступны [по ссылке.](https://wandb.ai/vadim-smir97-simon-fraser-university/ASR_HW/runs/9cwgliuq/logs)
обучение с конфигурацией`-cn=train_argmax` - более 2х часов на A100 GPU.

[Ссылка на `W&B`](https://wandb.ai/vadim-smir97-simon-fraser-university/ASR_HW?nw=nwuservadimsmir97)



## Метрики WER и CER для финальной модели
Результат запуска модели можно посмотреть Train.ipynb
```bash
    test_CER_(Argmax): 0.07413463661593472
    test_WER_(Argmax): 0.23709396768903815
    test_CER_(LM-BeamSearch): 0.06139948323314231
    test_WER_(LM-BeamSearch): 0.164760857707281
```


