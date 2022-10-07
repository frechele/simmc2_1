def _make_table(strings):
    return { k: v for v, k in enumerate(strings) }

ACTION_STRINGS = [
    "NONE",
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
]
ACTION_TABLE = _make_table(ACTION_STRINGS)

SLOT_KEY_STRINGS = [
    "NONE",
    'customerRating',
    'color',
    'sleeveLength',
    'availableSizes',
    'materials',
    'brand',
    'customerReview',
    'pattern',
    'size',
    'price',
    'type'  # only used in slots (not request_slot)
]

NUM_OF_FURNITURES = 57
NUM_OF_FASHION = 288
NUM_OF_OBJECTS = NUM_OF_FURNITURES + NUM_OF_FASHION
