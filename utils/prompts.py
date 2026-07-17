"""
=========================================================
Prompt Expansion Engine
Project : DRONIONS AI
Author  : Taha Ozturk
=========================================================

This module expands a target object into multiple
semantic prompts for open-vocabulary detectors
such as YOLO-World or Grounding DINO.

Example:

Input:
    charger

Output:
[
    "charger",
    "phone charger",
    "usb charger",
    ...
]
"""

from typing import Dict, List


PROMPT_DATABASE: Dict[str, List[str]] = {

    "charger": [
        "charger",
        "phone charger",
        "usb charger",
        "usb wall charger",
        "charging adapter",
        "power adapter",
        "phone charging brick",
        "power brick",
        "white charger"
    ],

    "phone": [
        "phone",
        "smartphone",
        "mobile phone",
        "cell phone",
        "iphone",
        "android phone"
    ],

    "laptop": [
        "laptop",
        "notebook computer",
        "portable computer"
    ],

    "backpack": [
        "backpack",
        "school bag",
        "travel backpack",
        "black backpack"
    ],

    "bottle": [
        "bottle",
        "water bottle",
        "plastic bottle",
        "drink bottle"
    ],

    "keyboard": [
        "keyboard",
        "computer keyboard"
    ],

    "mouse": [
        "computer mouse",
        "wireless mouse",
        "mouse"
    ],

    "keys": [
        "keys",
        "house keys",
        "keychain",
        "car keys"
    ],

    "wallet": [
        "wallet",
        "leather wallet"
    ]
}


NEGATIVE_DATABASE: Dict[str, List[str]] = {

    "charger": [
        "power strip",
        "extension cord",
        "wall socket",
        "electrical outlet",
        "surge protector"
    ],

    "phone": [
        "tablet",
        "remote control"
    ]
}


def get_prompts(target: str) -> List[str]:
    """
    Returns expanded prompts.

    Parameters
    ----------
    target : str

    Returns
    -------
    List[str]
    """

    target = target.lower().strip()

    if target in PROMPT_DATABASE:
        return PROMPT_DATABASE[target]

    return [target]


def get_negative_prompts(target: str) -> List[str]:
    """
    Returns negative prompts.

    Parameters
    ----------
    target : str

    Returns
    -------
    List[str]
    """

    target = target.lower().strip()

    return NEGATIVE_DATABASE.get(target, [])