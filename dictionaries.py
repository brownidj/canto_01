# -*- coding: utf-8 -*-
"""
Central dictionaries for the app.
- MINI_GLOSS: tiny built-in fallback glossary for very common forms (traditional)
- ANDYS_LIST: Andy's custom list in MINI_GLOSS style

When run directly (python dictionaries.py), this module will merge ANDYS_LIST
into MINI_GLOSS (in-memory) without duplicates and print a short summary.
"""

# ------------------------ MINI_GLOSS (very common) ------------------------ #
MINI_GLOSS = {
    u"我": ["I", "me"],
    u"你": ["you"],
    u"佢": ["he", "she", "they"],
    u"唔": ["not (Cantonese)"],
    u"有": ["to have", "there is"],
    u"冇": ["not have (Cantonese)"],
    u"去": ["to go"],
    u"食": ["to eat"],
    u"飲": ["to drink"],
    u"學": ["to learn; school (in compounds)"],
    u"講": ["to speak; to say"],
    u"睇": ["to look; to watch (Cantonese)"],
    u"聽": ["to listen"],
    u"車": ["vehicle; car"],
    u"書": ["book"],
    u"電": ["electric; electricity"],
    u"心": ["heart; mind"],
    u"頭": ["head; top"],
    u"手": ["hand"],
    u"腳": ["leg; foot"],
    u"日": ["sun; day"],
    u"月": ["moon; month"],
    u"年": ["year"],
    u"香港": ["Hong Kong"],
    u"廣東話": ["Cantonese (language)"],
    u"你好": ["hello"],
    u"然": ["thus", "so", "like that"],
    u"其": ["its", "his", "her", "their"],
    u"之": ["(classical genitive/linker)"],
    u"以": ["to use", "by means of"],
    u"於": ["at", "in", "to"],
    u"而": ["and", "and then", "but"],
    u"則": ["then", "in that case"],
    u"不": ["not"],
    u"也": ["also", "too"],
    u"的": ["(structural/possessive particle)"],
    u"了": ["(aspect particle)", "completed action"],
    u"嗎": ["(question particle)"],
    u"呢": ["(particle: how about…; continuative)"],
    u"吧": ["(particle: suggestion/softener)"],
    u"個": ["classifier (general)"],
    u"嘢": ["thing; stuff", "food/drink (in collocations like 食嘢、飲嘢)"],
    u"食嘢": ["to eat (colloquial)", "to get something to eat"],
    u"飲嘢": ["to drink (colloquial)", "to get something to drink"],
    u"一": ["one"],
    u"二": ["two"],
    u"三": ["three"],
    u"四": ["four"],
    u"六": ["six"],
    u"七": ["seven"],
    u"八": ["eight"],
    u"九": ["nine"],
    u"十": ["ten"],
    u"零": ["zero"],
    u"人": ["person"],
}

# ------------------------ Tricky Initials (examples) ------------------------ #
TRICKY_INITIALS = {
    # z initial (unaspirated /ts/)
    u"走": ["to run; to leave"],
    u"早": ["early; morning"],
    u"鐘": ["clock; bell"],
    u"張": ["classifier for flat objects; sheet"],
    u"自然": ["nature"],
    u"資料": ["data"],
    u"做運": ["to exercise"],
    u"自然界": ["nature"],
    u"資料庫": ["database"],
    u"做運動": ["to exercise"],
    u"中學生": ["middle school student"],

    # c initial (aspirated /tsʰ/)
    u"車": ["vehicle; car"],
    u"菜": ["vegetable; dish"],
    u"唱": ["to sing"],
    u"長": ["long; elder"],
    u"出門": ["to go out"],
    u"差不": ["almost; nearly"],
    u"聰明": ["smart; clever"],
    u"廚房": ["kitchen"],
    u"餐廳裡": ["in the restaurant"],
    u"傳統節": ["traditional festival"],
    u"參觀團": ["tour group"],
    u"採訪者": ["interviewer"],

    # j initial (/j/, English y in yes)
    u"有": ["to have; there is"],
    u"要": ["to want; to need"],
    u"魚": ["fish"],
    u"月": ["moon; month"],
    u"有人": ["someone"],
    u"要求": ["to request"],
    u"魚市": ["fish market"],
    u"月球": ["moon"],
    u"有人說": ["someone said"],
    u"要求高": ["high demand"],
    u"魚市場": ["fish market"],
    u"月球人": ["moon person (astronaut)"],

    # gw initial
    u"廣": ["broad; vast (as in 廣東 Guangdong)"],
    u"貴": ["expensive; noble"],
    u"國": ["country; nation"],
    u"棍": ["stick; rod"],
    u"廣播": ["broadcast"],
    u"貴族": ["aristocrat"],
    u"國際": ["international"],
    u"棍棒": ["stick"],
    u"廣播站": ["broadcast station"],
    u"貴族家": ["aristocratic family"],
    u"國際線": ["international route"],
    u"棍棒隊": ["stick-wielding team"],

    # kw initial
    u"快": ["fast; quick"],
    u"窮": ["poor"],
    u"群": ["group; crowd"],
    u"葵": ["sunflower"],
    u"快捷": ["fast"],
    u"窮學": ["poor student"],
    u"群體": ["group"],
    u"葵花": ["sunflower"],
    u"快捷鍵": ["shortcut key"],
    u"窮學生": ["poor student"],
    u"群體性": ["group nature"],
    u"葵花籽": ["sunflower seed"],

    # ng initial
    u"我": ["I; me"],
    u"五": ["five"],
    u"牙": ["tooth"],
    u"牛": ["cow; ox"],
    u"我自": ["myself"],
    u"五金": ["hardware"],
    u"牙醫": ["dentist"],
    u"牛肉": ["beef"],
    u"我自己": ["myself"],
    u"五金店": ["hardware store"],
    u"牙醫師": ["dentist"],
    u"牛肉麵": ["beef noodles"],
}

# ------------------------ Andy's List (MINI_GLOSS-style) ------------------------ #
ANDYS_LIST = {
    u"一": ["one"],
    u"二": ["two"],
    u"三": ["three"],
    u"四": ["four"],
    u"五": ["five"],
    u"六": ["six"],
    u"七": ["seven"],
    u"八": ["eight"],
    u"九": ["nine"],
    u"十": ["ten"],
    u"零": ["zero"],
    u"我": ["I", "me"],
    u"人": ["person"],
    u"者": ["he", "she", "they"],
}

TEST_WORDS = {
    u"廣播": ["broadcast"],
    u"廣播站": ["broadcast station"],
    u"廣播電台": ["broadcasting company"],
    u"廣播新聞": ["news broadcast"],
}


def merge_andys_into_mini(mini: dict, andys: dict) -> dict:
    """Merge ANDYS_LIST into MINI_GLOSS in-place, skipping keys that already exist."""
    for k, v in andys.items():
        if k not in mini:
            mini[k] = v
    return mini

if __name__ == "__main__":
    before = len(MINI_GLOSS)
    merged = merge_andys_into_mini(MINI_GLOSS, ANDYS_LIST)
    after = len(merged)
    added = after - before
    print(f"Merged ANDYS_LIST into MINI_GLOSS: added {added} new entries (total {after}).")

