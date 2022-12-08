import matplotlib.pyplot as plt
from PIL import Image
import requests
import tempfile
import os
import json


def show_card(df_cards, card_id, size=(286, 395)):
    card_img_path = {}
    if card_id not in card_img_path:
        # download and save image to file
        req = requests.get(df_cards.loc[card_id]["img"])
        filename = tempfile.mktemp()
        with open(filename, "wb") as f:
            f.write(req.content)
        card_img_path[card_id] = filename
    return Image.open(card_img_path[card_id]).resize(size)


def show_deck(*, deck, preds=[], cols=10, sort_key=None, reverse=True):
    if sort_key is None:
        deck.sort(key=sort_key)
    deck_cards = [show_card(c) for c in deck]
    pred_cards = [show_card(c) for c in preds]

    # enough cols for all cards, if there are preds, we skip a space
    rows = int((len(deck) + len(preds) + cols - (not len(preds))) / cols)
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 2.5))
    ax = ax.ravel()
    for i, img in enumerate(deck_cards + pred_cards):
        if not len(deck) % cols == 0 and i >= len(deck):
            i += 1
        ax[i].imshow(img)
    for axes in ax:
        axes.axis("off")
    fig.tight_layout()
    plt.show()


def get_deck_characteristics(df_cards, row):
    cards_id_list = row['cards']

    class_list = []
    universal_cards_num = 0
    cost_list = []
    total_cost = 0
    attack_list = []
    health_list = []

    for card_id in cards_id_list:
        temp_df = df_cards.loc[card_id]

        class_list.append(temp_df['class'])

        if temp_df['class'] == 'universal':
            universal_cards_num += 1

        cost_list.append(temp_df['cost'])
        total_cost += temp_df['cost']

        attack_list.append(temp_df['attack'])

        health_list.append(temp_df['health'])

    return class_list, universal_cards_num, cost_list, total_cost, attack_list, health_list