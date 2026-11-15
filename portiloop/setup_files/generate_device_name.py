#!/usr/bin/env python3
"""Generate a random `<adjective>-<noun>` device name for Portiloop RDK X5 boards."""
import secrets

ADJECTIVES = (
    "agile", "amber", "bold", "brave", "bright", "brisk", "calm", "clever",
    "cosmic", "crisp", "curious", "daring", "eager", "gentle", "glad", "golden",
    "happy", "humble", "jolly", "keen", "kind", "lively", "lucky", "mellow",
    "merry", "mighty", "nimble", "noble", "patient", "peaceful", "playful",
    "plucky", "proud", "quick", "quiet", "rapid", "rosy", "serene", "sharp",
    "silent", "silver", "sleek", "smart", "smooth", "snappy", "solar", "solid",
    "steady", "sunny", "swift", "tender", "tidy", "vivid", "warm", "wild",
    "wise", "witty", "zesty",
)

NOUNS = (
    "alpaca", "badger", "beaver", "bison", "bobcat", "condor", "cougar", "coyote",
    "crane", "dolphin", "eagle", "egret", "elk", "falcon", "ferret", "finch",
    "fox", "gazelle", "gecko", "gibbon", "gopher", "hawk", "heron", "ibex",
    "iguana", "jackal", "jaguar", "jay", "koala", "lemur", "leopard", "llama",
    "lynx", "macaw", "marmot", "marten", "meerkat", "mink", "moose", "narwhal",
    "newt", "ocelot", "orca", "osprey", "otter", "owl", "panda", "panther",
    "parrot", "pelican", "penguin", "puma", "quail", "rabbit", "raccoon",
    "raven", "robin", "salmon", "seal", "serval", "sloth", "sparrow", "stork",
    "swan", "tapir", "tiger", "toucan", "turtle", "viper", "walrus", "weasel",
    "whale", "wolf", "wombat", "wren", "yak", "zebra",
)


def generate_name() -> str:
    return f"{secrets.choice(ADJECTIVES)}-{secrets.choice(NOUNS)}"


if __name__ == "__main__":
    print(generate_name())
