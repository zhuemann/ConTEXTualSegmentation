import argparse
import os

import pandas as pd

from image_text_segmentation import train_image_text_segmentation

def create_parser():
    parser = argparse.ArgumentParser(description="The main file to run multimodal setup. Consists of pre-training joint representation, masked language modeling and report generation.")
    parser.add_argument('--local', '-l', type=bool, help="Should the program run locally", default=False)
    arg = parser.parse_args()
    return arg


if __name__ == '__main__':
    args = create_parser()
    #local = args.local
    local = False
    if local:
        directory_base = "Z:/"
    else:
        directory_base = "/UserData/"

    config = {"seed": 98, "batch_size": 2, "dir_base": directory_base, "epochs": 100, "n_classes": 2, "LR": 1e-3,
              "IMG_SIZE": 1024, "train_samples": .8, "test_samples": .5, "data_path": "D:/candid_ptx/",
              "report_gen": False, "mlm_pretraining": False, "contrastive_training": True, "save_location": ""}

    #dataframe_location = os.path.join(directory_base, 'Zach_Analysis/candid_data/pneumothorax_with_multisegmentation_text_negatives_balanced_df.xlsx')

    seeds = [98, 117, 295, 456, 915]
    accuracy_list = []

    for seed in seeds:

        folder_name = "higher_res_for_paper/t5_language_att_no_vision_augmentions_larger_img_v30/seed" + str(seed) + "/"

        save_string = "/UserData/Zach_Analysis/result_logs/candid_result/image_text_segmentation_for_paper/" + folder_name
        save_location = os.path.join(directory_base, save_string)

        config["seed"] = seed
        config["save_location"] = save_location

        acc, valid_log = train_image_text_segmentation(config)
        df = pd.DataFrame(valid_log)
        df["test_accuracy"] = acc

        filepath = os.path.join(config["save_location"], "valid_150ep_seed" + str(seed) + '.xlsx')
        df.to_excel(filepath, index=False)

