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

    # The room's objects were all reachable by a single word until a flight
    # showed what that costs: the phone was only ever found as "smartphone",
    # never as "phone". A target with one prompt is a target the detector gets
    # one chance at, and open-vocabulary detection is sensitive to the exact
    # wording in a way no confidence threshold compensates for.
    "mug": [
        "mug",
        "coffee mug",
        "cup",
        "coffee cup",
        "ceramic mug",
        "teacup"
    ],

    "book": [
        "book",
        "hardcover book",
        "paperback book",
        "closed book",
        "textbook",
        "novel"
    ],

    "bowl": [
        "bowl",
        "cereal bowl",
        "ceramic bowl",
        "white bowl",
        "dish",
        "soup bowl"
    ],

    "headphones": [
        "headphones",
        "headset",
        "over-ear headphones",
        "gaming headset",
        "black headphones",
        "earphones"
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

    # What each of these gets confused with from a drone's viewpoint, which is
    # from above and at an angle. A mug seen from overhead is a dark circle and
    # a bowl is a light one; a book lying flat and a closed laptop are the same
    # rectangle. Giving the detector somewhere else to put those keeps them off
    # the target's class instead of forcing a best-positive match.
    "mug": [
        "bowl",
        "can",
        "vase",
        "flower pot",
        "lid"
    ],

    "book": [
        "laptop",
        "keyboard",
        "tablet",
        "magazine",
        "notebook computer",
        "placemat"
    ],

    "bowl": [
        "plate",
        "mug",
        "lid",
        "saucer",
        "paper"
    ],

    "headphones": [
        "bag",
        "cushion",
        "shoe",
        "cable",
        "hat"
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


def _check_databases() -> None:
    """A word cannot be both what to look for and what to ignore.

    The detector hands positives and negatives to the model as one class list
    and then drops every box that landed in a negative class, so a word in both
    lists removes the target from its own detections. That happened: four
    entries meant for PROMPT_DATABASE were pasted into NEGATIVE_DATABASE as
    well -- the key `"phone": [` appears in both dictionaries and a replace
    without a count hit both -- and the mug went from being found at 0.78 to
    not being found at all. Nothing raised; it read as the detector simply
    failing on those objects.
    """
    for target, positives in PROMPT_DATABASE.items():
        clash = set(positives) & set(NEGATIVE_DATABASE.get(target, []))
        if clash:
            raise ValueError(
                f"'{target}' hem pozitif hem negatif ifade tasiyor: "
                f"{sorted(clash)}. Bu hedefi kendi tespitlerinden eler.")


_check_databases()