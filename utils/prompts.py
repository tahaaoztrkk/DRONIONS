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
    ],

    # Measured on a real frame: the bare prompt "box" matched only the 3 m
    # red wall (conf 0.230) and missed the actual cardboard box entirely.
    # Adding these four put the real box top of the list at 0.359, and 0.566
    # once the negatives below were in play.
    "box": [
        "cardboard box",
        "carton",
        "brown box",
        "package",
        "shipping box"
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
    ],

    # Giving the detector somewhere else to put big flat rectangular surfaces
    # stops them being forced into the target class.
    #
    # Deliberately NOT including "floor"/"ground": measured on a real frame,
    # adding them flipped the ranking the wrong way (wall 0.491 over box 0.448,
    # versus box 0.566 over wall 0.531 without them). Negatives compete with
    # the positives for the same detections, so a negative that matches a large
    # part of the scene distorts the rest of the scores. Keep them to things
    # actually being confused for the target.
    "box": [
        "wall",
        "barrier",
        "panel"
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

    # Near-misses cost as much as a completely unknown word: the lookup is on
    # an exact string, so "key" missed the "keys" entry and a whole flight
    # searched on one prompt instead of four. Nothing said so at the time,
    # which is what made it worth handling rather than just documenting.
    for variant in (target + "s", target.rstrip("s")):
        if variant != target and variant in PROMPT_DATABASE:
            return PROMPT_DATABASE[variant]

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