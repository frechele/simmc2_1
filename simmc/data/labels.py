import numpy as np


def _make_mapping_table(lst):
    return lst, { k: v for v, k in enumerate(lst) }

def label_to_onehot(value, mapping_table):
    return np.eye(len(mapping_table))[mapping_table[value]]

def labels_to_vector(values, mapping_table):
    return np.sum([label_to_onehot(value, mapping_table) for value in values], axis=0)


TYPE, TYPE_MAPPING_TABLE = _make_mapping_table([
    "EndTable",  # furniture
    "CouchChair",
    "Lamp",
    "Chair",
    "AreaRug",
    "CoffeeTable",
    "Shelves",
    "Sofa",
    "Table",
    "Bed",
    "shirt",  # fashion
    "tshirt",
    "sweater",
    "skirt",
    "joggers",
    "dress",
    "blouse",
    "coat",
    "shoes",
    "trousers",
    "jeans",
    "tank top",
    "shirt, vest",
    "hoodie",
    "hat",
    "jacket",
    "vest",
    "suit"
])

BRAND, BRAND_MAPPING_TABLE = _make_mapping_table([
    'Downtown Stylists',
    'The Vegan Baker',
    'Art Den',
    'Ocean Wears',
    '212 Local',
    'Glam Nails',
    'New Fashion',
    'Nature Photographers',
    'Fancy Nails',
    'Uptown Gallery',
    'Uptown Studio',
    'Downtown Consignment',
    'Modern Arts',
    'Coats & More',
    'Yogi Fit',
    'Pedals & Gears',
    'Home Store',
    'HairDo',
    'Brain Puzzles',
    'Garden Retail',
    'Global Voyager',
    'Art News Today',
    'Cats Are Great',
    'North Lodge',
    'StyleNow Feed',
    'River Chateau'
])

# material in furniture, pattern in fashion
MATERIAL, MATERIAL_MAPPING_TABLE = _make_mapping_table([
    "natural fibers",
    "wool",
    "metal",
    "leather",
    "marble",
    "wood",
    "memory foam",
    "heavy stripes",
    "text",
    "canvas",
    "cargo",
    "denim",
    "heavy vertical stripes",
    "light spots",
    "checkered",
    "radiant",
    "light vertical stripes",
    "vertical design",
    "plain",
    "design",
    "dotted",
    "plain with stripes on side",
    "leafy design",
    "streaks",
    "camouflage",
    "floral",
    "horizontal stripes",
    "vertical stripes",
    "twin colors",
    "velvet",
    "checkered, plain",
    "stripes",
    "spots",
    "diamonds",
    "plaid",
    "leather",
    "light stripes",
    "multicolored",
    "vertical striples",
    "leapard print",
    "holiday",
    "knit",
    "star design"
])

ASSET_TYPE, ASSET_TYPE_MAPPING_TABLE = _make_mapping_table([
    "tshirt_display",
    "jacket_hanging",
    "tshirt_folded",
    "skirt",
    "shoes",
    "trousers_display",
    "jacket_display",
    "tshirt_hanging",
    "hat",
    "dress_hanging",
    "blouse_hanging"
])

COLOR, COLOR_MAPPING_TABLE = _make_mapping_table([
    "pink",
    "red",
    "white",
    "blue",
    "dark violet",
    "dirty grey",
    "grey",
    "light green",
    "black",
    "light grey",
    "light pink",
    "light red",
    "light orange",
    "dark blue",
    "orange",
    "wooden",
    "maroon",
    "beige",
    "dark brown",
    "dark red",
    "dark pink",
    "light blue",
    "brown",
    "dark yellow",
    "purple",
    "violet",
    "olive",
    "golden",
    "yellow",
    "dark green",
    "dark grey",
    "green",
    "dirty green"
])

SLEEVE_LENGTH, SLEEVE_LENGTH_MAPPING_TABLE = _make_mapping_table([
    "",
    "half",
    "sleeveless",
    "long",
    "full",
    "short"
])

ACTION, ACTION_MAPPING_TABLE = _make_mapping_table([
    "INFORM:DISAMBIGUATE",
    "REQUEST:DISAMBIGUATE",
    "REQUEST:ADD_TO_CART",
    "CONFIRM:ADD_TO_CART",
    "ASK:GET",
    "INFORM:GET",
    "REQUEST:GET",
    "INFORM:COMPARE",
    "REQUEST:COMPARE",
    "INFORM:REFINE"
])

SLOT_KEY, SLOT_KEY_MAPPING_TABLE = _make_mapping_table([
    "customerRating",
    "color",
    "sleeveLength",
    "availableSizes",
    "materials",
    "brand",
    "customerReview",
    "pattern",
    "size",
    "price",
    "type"
])

SYSTEM_UTTR_TOKEN = "<SYSU>"
USER_UTTR_TOKEN = "<USRU>"
