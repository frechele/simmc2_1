def _make_mapping_table(lst):
    return lst, { k: v for v, k in enumerate(lst) }


ACTION, ACTION_MAPPING_TABLE = _make_mapping_table([
    "INFORM:DISAMBIGUATE",
    "INFORM:GET",
    "REQUEST:GET",
    "ASK:GET",
    "INFORM:REFINE",
    "REQUEST:COMPARE",
    "REQUEST:ADD_TO_CART"
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
