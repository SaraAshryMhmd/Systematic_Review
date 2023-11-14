import json
import urllib.parse

import pandas as pd
import requests
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()

# Replace with your API Key
api_key = os.getenv('API_KEY')

def main():
        # Page size
    count = 25

    # Page number
    start = 0

    # Your query
    #myquery = """((("Activity recognition*") OR ("Activity classification") OR ("Activities recognition*") OR ("Physical activit*") OR ("Activities classification") OR ("HAR") OR ("HARs")) AND (("Accelerometer*") OR ("Inertial Measurement Unit*") OR ("IMU") OR ("IMUs") OR ("Time Series Data")) AND (("Deep learning") OR ("Transfer learning") OR ("Neutral network*") OR ("RNN") OR ("RNNs") OR ("long short term memory") OR ("LSTM") OR ("Bi-LSTM") OR ("BiLSTM") OR ("attention based") OR ("transformer") OR ("CNN") OR ("CNNs") OR ("Generative Adversarial Network*") OR ("GAN") OR ("GANs") OR ("One-Shot Learning") OR ("Multi-Task Learning") OR ("Gated Recurrent Unit") OR ("GRU") OR ("GRUs") OR ("WaveNet") OR ("Convolutional Network*") OR ("Encoder-Decoder") OR ("Autoencoder*") OR ("VAE") OR ("VAEs") OR ("Domain Adaptation") OR ("Meta-Learning")) AND (("Human") OR ("Patient*") OR ("People") OR ("Individual*") OR ("Person*") OR ("Population*") OR ("participant*") OR ("player*") OR ("children*") OR ("mother*") OR ("pregnant*") OR ("adult*") OR ("user*") OR ("pedestrian*") OR ("citizen*")))"""

    myquery = """((("Activity recognition*") OR ("Activities recognition*") OR ("Physical activit*") OR ("Activity classification") OR ("Activities classification*") OR ("Activity detection*") OR ("Activities detection*") OR ("Activity prediction*") OR ("Activities prediction*") OR ("Physical exercises") OR ("daily activit*") OR ("Activity of daily living") OR ("ADL") OR ("human activit*") OR ("Sport") OR ("gait abnormalities") OR ("gait deviations") OR ("gait detection") OR ("gait symmetry") OR ("logistics activities") OR ("manual work") OR ("step counting") OR ("stride estimation") OR ("hand gesture") OR ("movement") OR ("Fall detection") OR ("Warehousing") OR ("order picking") OR ("Frisbee") OR ("Dancing") OR ("Hiking") OR ("Climb Stairs") OR ("Skiing") OR ("Yoga") OR ("Rowing") OR ("Swimming") OR ("Aerobics") OR ("Football") OR ("Soccer") OR ("Basketball") OR ("Volleyball") OR ("Tennis") OR ("Badminton") OR ("Gymnastics") OR ("Outdoor activities") OR ("Indoor activities") OR ("Skateboarding") OR ("Rollerblading") OR ("Surfing") OR ("Kayaking") OR ("Canoeing") OR ("Parkour") OR ("Taekwondo") OR ("Karate") OR ("Squash") OR ("Racquetball") OR ("Kiteboarding") OR ("Jumping") OR ("Walking") OR ("Canyoning") OR ("Paddleboarding") OR ("Biking ") OR ("Handball") OR ("Weightlifting") OR ("Skydiving") OR ("Feeding ") OR ("eating") OR ("Dressing") OR ("Personal hygiene") OR ("Brushing teeth") OR ("Ambulating") OR ("Cooking") OR ("Cleaning ") OR ("vacuuming") OR ("mopping") OR ("Getting dressed") OR ("Washing hands") OR ("Reading") OR ("Writing") OR ("Making the bed") OR ("Combing hair") OR ("Washing face") OR ("Household Chores") OR ("Zumba") OR ("Shaving") OR ("Sweeping") OR ("Ironing clothes") OR ("praying") OR ("sprinting") OR ("upstairs") OR ("downstairs") OR ("clapping") OR ("lying down") OR ("sleeping") OR ("dribbling") OR ("playing") OR ("staying") OR ("open door") OR ("push up") OR ("pull up") OR ("burpee") OR ("box jump") OR ("press") OR ("thruster") OR ("sit up") OR ("deadlift") OR ("games") OR ("Treadmill running") OR ("stepping") OR ("lower limb") OR ("upper limb")) AND (("Accelerometer*") OR ("Inertial Measurement Unit*") OR ("IMU") OR ("IMUs") OR ("Gyroscope") OR ("Pressure Sensors") OR ("magnetometer") OR ("Barometer")) AND (("Transfer learning*") OR ("Adversarial Domain Adaptation*") OR ("Domain-Adversarial Neural Network*") OR ("DANN") OR ("Adapting") OR ("Multi-Task Learning") OR ("MTL") OR ("Pretrained") OR ("Pretraining") OR ("pre-trained") OR ("pre-train") OR ("Feature Transfer") OR ("Meta-Learning") OR ("MAML") OR ("Knowledge Distillation") OR ("Transfer Knowledge") OR ("Knowledge Transfer") OR ("Transfer Ensemble") OR ("Progressive Neural Network*") OR ("PNN") OR ("Fine-tuning") OR ("Few-shot Learning") OR ("One-Shot Learning") OR ("Weakly Supervised Classification") OR ("Inception") OR ("Xception") OR ("VGG ") OR ("Visual Geometry Group") OR ("Residual Networks") OR ("ResNet")) AND (("Human") OR ("People ") OR ("pedestrian*") OR ("Individual*") OR ("Person*") OR ("Population*") OR ("participant*") OR ("player*") OR ("children*") OR ("adult*") OR ("user*") OR ("disabilities*") OR ("Athlete*")))"""
    #myquery = """((("Activity recognition*") OR ("Activities recognition*") OR ("Physical activit*") OR ("Activity classification") OR ("Activities classification*") OR ("Activity detection*") OR ("Activities detection*") OR ("Activity prediction*") OR ("Activities prediction*") OR ("Physical exercises") OR ("daily activit*") OR ("Activity of daily living") OR ("ADL") OR ("human activit*") OR ("Sport")) AND (("Accelerometer*") OR ("Inertial Measurement Unit*") OR ("IMU") OR ("IMUs") OR ("Gyroscope") OR ("Pressure Sensors") OR ("magnetometer") OR ("Barometer")) AND (("Transfer learning*") OR ("Adversarial Domain Adaptation*") OR ("Domain-Adversarial Neural Network*") OR ("DANN") OR ("Adapting") OR ("Multi-Task Learning") OR ("MTL") OR ("Pretrained") OR ("Pretraining") OR ("pre-trained") OR ("pre-train") OR ("Feature Transfer") OR ("Meta-Learning") OR ("MAML") OR ("Knowledge Distillation") OR ("Transfer Knowledge") OR ("Knowledge Transfer") OR ("Transfer Ensemble") OR ("Progressive Neural Network*") OR ("PNN") OR ("Fine-tuning") OR ("Few-shot Learning") OR ("One-Shot Learning") OR ("Weakly Supervised Classification") OR ("Inception") OR ("Xception") OR ("VGG ") OR ("Visual Geometry Group") OR ("Residual Networks") OR ("ResNet")) AND (("Human") OR ("People ") OR ("pedestrian*") OR ("Individual*") OR ("Person*") OR ("Population*") OR ("participant*") OR ("player*") OR ("children*") OR ("adult*") OR ("user*") OR ("disabilities*") OR ("Athlete*")))"""
    #myquery = """((("Activity recognition*") OR ("Physical activit*") OR ("Activity classification") OR ("Activity detection*") OR ("Activity prediction*") OR ("Physical exercises") OR ("daily activit*") OR ("Activity of daily living") OR ("ADL") OR ("human activit*") OR ("Sport")) AND (("Accelerometer*") OR ("Inertial Measurement Unit*") OR ("IMU") OR ("IMUs") OR ("Gyroscope") OR ("Pressure Sensors") OR ("magnetometer") OR ("Barometer")) AND (("Transfer learning*") OR ("Adversarial Domain Adaptation*") OR ("DANN") OR ("Adapting") OR ("Multi Task Learning") OR ("MTL") OR ("Pretrained") OR ("Pretraining") OR ("Feature Transfer") OR ("MAML") OR ("Knowledge Distillation") OR ("Transfer Knowledge") OR ("Knowledge Transfer") OR ("Transfer Ensemble") OR ("Progressive Neural Network*") OR ("PNN") OR ("Fine tuning") OR ("Few shot Learning") OR ("One Shot Learning") OR ("Weakly Supervised Classification") OR ("Inception") OR ("Xception") OR ("VGG ") OR ("Visual Geometry Group") OR ("Residual Networks") OR ("ResNet")) AND (("Human") OR ("People ") OR ("pedestrian*") OR ("children*") OR ("adult*")))"""

    #myquery = """((("push up") OR ("pull up") OR ("burpee") OR ("box jump") OR ("press") OR ("thruster") OR ("sit up") OR ("deadlift") OR ("games") OR ("Treadmill running") OR ("stepping") OR ("lower limb") OR ("upper limb")) AND (("Accelerometer*") OR ("Inertial Measurement Unit*") OR ("IMU") OR ("IMUs") OR ("Gyroscope") OR ("Pressure Sensors") OR ("magnetometer") OR ("Barometer")) AND (("Transfer learning*") OR ("Adversarial Domain Adaptation*") OR ("Domain-Adversarial Neural Network*") OR ("DANN") OR ("Adapting") OR ("Multi-Task Learning") OR ("MTL") OR ("Pretrained") OR ("Pretraining") OR ("pre-trained") OR ("pre-train") OR ("Feature Transfer") OR ("Meta-Learning") OR ("MAML") OR ("Knowledge Distillation") OR ("Transfer Knowledge") OR ("Knowledge Transfer") OR ("Transfer Ensemble") OR ("Progressive Neural Network*") OR ("PNN") OR ("Fine-tuning") OR ("Few-shot Learning") OR ("One-Shot Learning") OR ("Weakly Supervised Classification") OR ("Inception") OR ("Xception") OR ("VGG ") OR ("Visual Geometry Group") OR ("Residual Networks") OR ("ResNet")) AND (("Human") OR ("People ") OR ("pedestrian*") OR ("children*") OR ("adult*") OR ("user*") OR ("disabilities*") OR ("Athlete*")))"""

    query = f"TITLE({myquery}) OR ABS({myquery})"

    headers = {
        "X-ELS-APIKey": api_key,
        "Accept": "application/json",
    }

    df = pd.DataFrame()

    # Send a first request to get the total number of results
    response = requests.get(
        f"https://api.elsevier.com/content/search/scopus?start={start}&count={count}&query={query}", headers=headers)

    # Create the progress bar
    total_results = int(
        response.json()['search-results']['opensearch:totalResults'])
    pbar = tqdm(total=total_results)

    while True:
        response = requests.get(
            f"https://api.elsevier.com/content/search/scopus?start={start}&count={count}&query={query}", headers=headers)
        if response.status_code == 200:
            data = response.json()
            if 'search-results' in data and 'entry' in data['search-results']:
                df = pd.concat([df, pd.json_normalize(
                    data['search-results']['entry'])], ignore_index=True)

                # Update the progress bar
                pbar.update(len(data['search-results']['entry']))

                start_index = int(data['search-results']['opensearch:startIndex'])
                items_per_page = int(
                    data['search-results']['opensearch:itemsPerPage'])
                if start_index + items_per_page >= total_results:
                    pbar.close()
                    break  # Exit the loop if there are no more pages

                # Go to the next page
                start += count
            else:
                print("No 'search-results' or 'entry' in data")
                print(data)
                break
        else:
            print(f"Request failed with status code {response.status_code}")
            break

    df.to_excel('scopus.xlsx', sheet_name='scopus', header=True, index=True)

main()