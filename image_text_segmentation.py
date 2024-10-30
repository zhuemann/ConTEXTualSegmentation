import os
from sklearn import model_selection
import torchvision.transforms as transforms
from transformers import AutoTokenizer, RobertaModel, T5Model, T5Tokenizer
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import gc
import albumentations as albu

from models.ConTEXTual_model import ConTEXTual_model
from dataloader_image_text import TextImageDataset
from utility import dice_coeff
import ssl
import nltk
ssl.SSLContext.verify_mode = ssl.VerifyMode.CERT_OPTIONAL


def train_image_text_segmentation(config, batch_size=8, epoch=1, dir_base = "/home/zmh001/r-fcb-isilon/research/Bradshaw/", n_classes = 2):
    nltk.download('punkt')
    # model specific global variables
    IMG_SIZE = config["IMG_SIZE"]
    LR = 5e-5
    dir_base = config["dir_base"]
    seed = config["seed"]
    BATCH_SIZE = config["batch_size"]
    N_EPOCHS = config["epochs"]

    # the folder in which the test dataframe, model, results log will all be saved to
    save_location = config["save_location"]

    #dataframe_location = os.path.join(dir_base, "Zach_Analysis/candid_data/pneumothorax_with_multisegmentation_text_negatives_balanced_df.xlsx")
    #dataframe_location = os.path.join(dir_base, 'Zach_Analysis/candid_data/pneumothorax_with_multisegmentation_positive_text_df.xlsx')
    #dataframe_location = os.path.join(dir_base, 'Zach_Analysis/candid_data/pneumothorax_with_text_df.xlsx') #pneumothorax_df chest_tube_df rib_fracture
    #dataframe_location = os.path.join(dir_base, 'Zach_Analysis/candid_data/pneumothorax_large_df.xlsx')
    # gets the candid labels and saves it off to the location
    #df = get_candid_labels(dir_base=dir_base)
    #df = get_all_text_image_pairs(dir_base=dir_base)
    #print(df)
    #df.to_excel(dataframe_location, index=False)
    dataframe_location = 'image_name_text_label_redacted.csv'

    # reads in the dataframe as it doesn't really change to save time
    #df = pd.read_excel(dataframe_location, engine='openpyxl')
    df = read_csv(dataframe_location)
    df.set_index("image_id", inplace=True)

    # location of synonyms scraped from Radlex
    #wordReplacementPath = os.path.join(dir_base, 'Zach_Analysis/lymphoma_data/words_and_their_synonyms.xlsx')
    wordReplacementPath = os.path.join('words_and_their_synonyms.xlsx')
    dfWord = pd.read_excel(wordReplacementPath, engine='openpyxl')
    dfWord.set_index("word", inplace=True)
    # sets up the list of synonyms obtained from Radlex
    wordDict = dfWord.to_dict()
    for key in list(wordDict["synonyms"].keys()):
        string = wordDict["synonyms"][key][2:-2]
        wordList = string.split("', '")
        wordDict["synonyms"][key] = wordList

    # use t5 as text encoder
    t5_path = os.path.join(dir_base, 'Zach_Analysis/models/t5_large/')
    tokenizer = T5Tokenizer.from_pretrained(t5_path)
    language_model = T5Model.from_pretrained(t5_path)

    # Splits the data into 80% train and 20% valid and test sets
    train_df, test_valid_df = model_selection.train_test_split(
        df, train_size=config["train_samples"], random_state=seed, shuffle=True #stratify=df.label.values
    )
    # Splits the test and valid sets in half so they are both 10% of total data
    test_df, valid_df = model_selection.train_test_split(
        test_valid_df, test_size=config["test_samples"], random_state=seed, shuffle=True #stratify=test_valid_df.label.values
    )
    # Saves the test dataframe in case you want to acess it in the future
    test_dataframe_location = os.path.join(save_location, 'pneumothorax_testset_df_seed' + str(config["seed"]) + '.xlsx')
    test_df.to_excel(test_dataframe_location, index=True)

    # emprically the good augmentations that are report invariant
    use_vision_augmentations = True
    if use_vision_augmentations:
        albu_augs = albu.Compose([
            #albu.HorizontalFlip(p=.5),
            albu.OneOf([
                albu.RandomContrast(),
                albu.RandomGamma(),
                albu.RandomBrightness(),
            ], p=.3),
            albu.OneOf([
                albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                albu.GridDistortion(),
                albu.OpticalDistortion(distort_limit=2, shift_limit=0.5),
            ], p=.3),
            albu.ShiftScaleRotate(),
                                    ])

    transforms_valid = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.PILToTensor(),
        ]
    )

    transforms_resize = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.PILToTensor()])
    output_resize = transforms.Compose([transforms.Resize((1024, 1024))])

    training_set = TextImageDataset(train_df, tokenizer, 512, mode="train", transforms = albu_augs, resize=transforms_resize, dir_base = dir_base, img_size=IMG_SIZE, wordDict = wordDict)
    valid_set =    TextImageDataset(valid_df, tokenizer, 512,               transforms = transforms_valid, resize=transforms_resize, dir_base = dir_base, img_size=IMG_SIZE, wordDict = None)
    test_set =     TextImageDataset(test_df,  tokenizer, 512,               transforms = transforms_valid, resize=transforms_resize, dir_base = dir_base, img_size=IMG_SIZE, wordDict = None)

    train_params = {'batch_size': BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 4
                    }

    test_params = {'batch_size': BATCH_SIZE,
                   'shuffle': True,
                   'num_workers': 4
                   }

    training_loader = DataLoader(training_set, **train_params)
    valid_loader = DataLoader(valid_set, **test_params)
    test_loader = DataLoader(test_set, **test_params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_obj = ConTEXTual_model(lang_model=language_model, n_channels=3, n_classes=1, bilinear=False)

    # freezes the parameters in the language model to stop language representation from collapsing
    for param in language_model.parameters():
        param.requires_grad = False

    test_obj.to(device)

    criterion = nn.BCEWithLogitsLoss()

    # defines which optimizer is being used
    optimizer = torch.optim.AdamW(params=test_obj.parameters(), lr=LR)

    best_acc = 0
    del train_df
    valid_log = []
    for epoch in range(1, N_EPOCHS + 1):

        test_obj.train()
        training_dice = []
        gc.collect()
        torch.cuda.empty_cache()

        for _, data in tqdm(enumerate(training_loader, 0)):

            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            targets = torch.squeeze(targets)
            images = data['images'].to(device, dtype=torch.float)

            outputs = test_obj(images, ids, mask, token_type_ids)
            outputs = output_resize(torch.squeeze(outputs, dim=1))
            optimizer.zero_grad()

            loss = criterion(outputs, targets)

            if _ % 400 == 0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # put output between 0 and 1 and rounds to nearest integer ie 0 or 1 labels
            sigmoid = torch.sigmoid(outputs)
            outputs = torch.round(sigmoid)

            # calculates the dice coefficent for each image and adds it to the list
            for i in range(0, outputs.shape[0]):
                dice = dice_coeff(outputs[i], targets[i])
                dice = dice.item()
                # gives a dice score of 1 if correctly predicts negative
                if torch.max(outputs[i]) == 0 and torch.max(targets[i]) == 0:
                    dice = 1

                training_dice.append(dice)

        avg_training_dice = np.average(training_dice)
        print(f"Epoch {str(epoch)}, Average Training Dice Score = {avg_training_dice}")

        # each epoch, look at validation data
        with torch.no_grad():

            test_obj.eval()
            valid_dice = []
            gc.collect()
            for _, data in tqdm(enumerate(valid_loader, 0)):
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.float)
                targets = torch.squeeze(targets)
                images = data['images'].to(device, dtype=torch.float)

                outputs = test_obj(images, ids, mask, token_type_ids)

                outputs = output_resize(torch.squeeze(outputs, dim=1))
                targets = output_resize(targets)

                # put output between 0 and 1 and rounds to nearest integer ie 0 or 1 labels
                sigmoid = torch.sigmoid(outputs)
                outputs = torch.round(sigmoid)

                # calculates the dice coefficent for each image and adds it to the list
                for i in range(0, outputs.shape[0]):
                    dice = dice_coeff(outputs[i], targets[i])
                    dice = dice.item()
                    if torch.max(outputs[i]) == 0 and torch.max(targets[i]) == 0:
                        dice = 1
                    valid_dice.append(dice)

            avg_valid_dice = np.average(valid_dice)
            print(f"Epoch {str(epoch)}, Average Valid Dice Score = {avg_valid_dice}")
            valid_log.append(avg_valid_dice)

            if avg_valid_dice >= best_acc:
                best_acc = avg_valid_dice
                save_path = os.path.join(config["save_location"], "best_segmentation_model_seed_test" + str(seed))
                torch.save(test_obj.state_dict(), save_path)

    row_ids = []
    saved_path = os.path.join(config["save_location"], "best_segmentation_model_seed_test" + str(seed))
    test_obj.load_state_dict(torch.load(saved_path))
    test_obj.eval()

    with torch.no_grad():
        test_dice = []
        gc.collect()
        for _, data in tqdm(enumerate(test_loader, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            targets = torch.squeeze(targets)
            images = data['images'].to(device, dtype=torch.float)

            outputs = test_obj(images, ids, mask, token_type_ids)
            outputs = output_resize(torch.squeeze(outputs, dim=1))
            targets = output_resize(targets)

            sigmoid = torch.sigmoid(outputs)
            outputs = torch.round(sigmoid)
            row_ids.extend(data['row_ids'])

            for i in range(0, outputs.shape[0]):
                dice = dice_coeff(outputs[i], targets[i])
                dice = dice.item()
                if torch.max(outputs[i]) == 0 and torch.max(targets[i]) == 0:
                    dice = 1
                test_dice.append(dice)

        avg_test_dice = np.average(test_dice)
        print(f"Epoch {str(epoch)}, Average Test Dice Score = {avg_test_dice}")

        return avg_test_dice, valid_log
